import os
import time
import pickle
import argparse

import numpy as np
import pandas as pd

from operator import itemgetter
from datetime import datetime as dt

from tqdm import tqdm


# seed
SEED = 12345
# SEED = 23456
# SEED = 34567
# SEED = 45678
# SEED = 56789

# settings
DNAMES = (
    'ml1m',
    'ml20m',
    'steam2',
)
MAX_SEQUENCE_LENGTH = 200
MIN_SESSION_COUNT_PER_USER = 2
MIN_ITEM_COUNT_PER_SESSION = 2
MIN_ITEM_COUNT_PER_USER = 5
MIN_USER_COUNT_PER_ITEM = 5
SESSION_WINDOW = 24 * 60 * 60
NUM_NEGATIVE_SAMPLES = 100
NEGATIVE_SAMPLER_SEED = SEED


def parse_args():
    task2names = {
        'prepare': DNAMES,
        'count': (
            'stats',
        ),
    }
    tasks = list(task2names.keys())
    names = []
    for subnames in list(task2names.values()):
        names.extend(subnames)
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, choices=tasks, help="task to do")
    parser.add_argument('name', type=str, choices=names, help="name to do")
    parser.add_argument('--data_root', type=str, default='./data', help="data root dir")
    parser.add_argument('--rough_root', type=str, default='/roughs', help="rough root dir")

    return parser.parse_args()


def dts2ts(dts):
    dts = dts.split('.')[0]
    if 'T' in dts:
        template = "%Y-%m-%dT%H:%M:%S"
    else:
        template = "%Y-%m-%d %H:%M:%S"
    if 'Z' in dts:
        template += 'Z'
    dto = dt.strptime(dts, template)
    ts = int(time.mktime(dto.timetuple()))
    return ts


def cut_and_assign_sids_to_rows(rows):
    sid = 0
    uid2rows = {}
    for uid, iid, timestamp in tqdm(rows, desc="* organize uid2rows"):
        if uid not in uid2rows:
            uid2rows[uid] = []
        uid2rows[uid].append((iid, timestamp))
    rows = []
    uids = list(uid2rows.keys())
    for uid in tqdm(uids, desc="* cutting"):
        user_rows = sorted(uid2rows[uid], key=itemgetter(1))
        tba = []
        sid2count = {}
        if MAX_SEQUENCE_LENGTH:
            user_rows = user_rows[-MAX_SEQUENCE_LENGTH:]
        sid += 1
        _, previous_timestamp = user_rows[0]
        for iid, timestamp in user_rows:
            if timestamp - previous_timestamp > SESSION_WINDOW:
                sid += 1
            tba.append((uid, iid, sid, timestamp))
            sid2count[sid] = sid2count.get(sid, 0) + 1
            previous_timestamp = timestamp
        if MIN_SESSION_COUNT_PER_USER and len(sid2count) < MIN_SESSION_COUNT_PER_USER:
            continue
        if MIN_ITEM_COUNT_PER_SESSION and min(sid2count.values()) < MIN_ITEM_COUNT_PER_SESSION:
            continue
        rows.extend(tba)
    return rows


def do_general_preprocessing(args, df_rows):
    """
        Create `df_rows` in a right format and the rest will be done.

        Args:
            `args`: see `parse_args`.
            `df_rows`: a DataFrame with column of `(uid, iid, timestamp)`.
    """
    print("do general preprocessing")

    # check first
    data_dir = os.path.join(args.data_root, args.name)
    os.makedirs(data_dir, exist_ok=True)

    # filter out tiny items
    print("- filter out tiny items")
    df_iid2ucount = df_rows.groupby('iid').size()
    survived_iids = df_iid2ucount.index[df_iid2ucount >= MIN_USER_COUNT_PER_ITEM]
    df_rows = df_rows[df_rows['iid'].isin(survived_iids)]

    # filter out tiny users
    print("- filter out tiny users")
    df_uid2icount = df_rows.groupby('uid').size()
    survived_uids = df_uid2icount.index[df_uid2icount >= MIN_ITEM_COUNT_PER_USER]
    df_rows = df_rows[df_rows['uid'].isin(survived_uids)]

    # cut and assign sid
    print("- cut and assign sid")
    rows = cut_and_assign_sids_to_rows(df_rows.values)
    df_rows = pd.DataFrame(rows)
    df_rows.columns = ['uid', 'iid', 'sid', 'timestamp']

    # map uid -> uindex
    print("- map uid -> uindex")
    uids = set(df_rows['uid'])
    uid2uindex = {uid: index for index, uid in enumerate(set(uids), start=1)}
    df_rows['uindex'] = df_rows['uid'].map(uid2uindex)
    df_rows = df_rows.drop(columns=['uid'])
    with open(os.path.join(data_dir, 'uid2uindex.pkl'), 'wb') as fp:
        pickle.dump(uid2uindex, fp)

    # map iid -> iindex
    print("- map iid -> iindex")
    iids = set(df_rows['iid'])
    iid2iindex = {iid: index for index, iid in enumerate(set(iids), start=1)}
    df_rows['iindex'] = df_rows['iid'].map(iid2iindex)
    df_rows = df_rows.drop(columns=['iid'])
    with open(os.path.join(data_dir, 'iid2iindex.pkl'), 'wb') as fp:
        pickle.dump(iid2iindex, fp)

    # save df_rows
    print("- save df_rows")
    df_rows.to_pickle(os.path.join(data_dir, 'df_rows.pkl'))

    # split train, valid, test
    print("- split train, valid, test")
    train_data = {}
    valid_data = {}
    test_data = {}
    for uindex in tqdm(list(uid2uindex.values()), desc="* splitting"):
        df_user_rows = df_rows[df_rows['uindex'] == uindex].sort_values(by='timestamp')
        user_rows = list(df_user_rows[['iindex', 'sid', 'timestamp']].itertuples(index=False, name=None))
        train_data[uindex] = user_rows[:-2]
        valid_data[uindex] = user_rows[-2: -1]
        test_data[uindex] = user_rows[-1:]

    # save splits
    print("- save splits")
    with open(os.path.join(data_dir, 'train.pkl'), 'wb') as fp:
        pickle.dump(train_data, fp)
    with open(os.path.join(data_dir, 'valid.pkl'), 'wb') as fp:
        pickle.dump(valid_data, fp)
    with open(os.path.join(data_dir, 'test.pkl'), 'wb') as fp:
        pickle.dump(test_data, fp)


def do_general_random_negative_sampling(args, num_samples=100, seed=SEED):
    """
        The `ns_random.pkl` created here is a dict with `uindex` as a key and a list of `iindex` as a value.

        `ns_random` = `uindex` -> [list of `iindex`].
    """
    print("do general random negative sampling")

    # check first
    data_dir = os.path.join(args.data_root, args.name)
    os.makedirs(data_dir, exist_ok=True)

    # load materials
    print("- load materials")
    with open(os.path.join(data_dir, 'df_rows.pkl'), 'rb') as fp:
        df_rows = pickle.load(fp)
    with open(os.path.join(data_dir, 'uid2uindex.pkl'), 'rb') as fp:
        uid2uindex = pickle.load(fp)
        user_count = len(uid2uindex)
    with open(os.path.join(data_dir, 'iid2iindex.pkl'), 'rb') as fp:
        iid2iindex = pickle.load(fp)
        item_count = len(iid2iindex)

    # sample random negatives
    print("- sample random negatives")
    ns = {}
    np.random.seed(seed)
    for uindex in tqdm(list(range(1, user_count + 1)), desc="* sampling"):
        seen_iindices = set(df_rows[df_rows['uindex'] == uindex]['iindex'])
        sampled_iindices = set()
        for _ in range(num_samples):
            iindex = np.random.choice(item_count) + 1
            while iindex in seen_iindices or iindex in sampled_iindices:
                iindex = np.random.choice(item_count) + 1
            sampled_iindices.add(iindex)
        ns[uindex] = list(sampled_iindices)

    # save sampled random nagetives
    print("- save sampled random nagetives")
    with open(os.path.join(data_dir, 'ns_random.pkl'), 'wb') as fp:
        pickle.dump(ns, fp)


def do_general_popular_negative_sampling(args, num_samples=100):
    """
        The `ns_popular.pkl` created here is a dict with `uindex` as a key and a list of `iindex` as a value.

        `ns_popular` = `uindex` -> [list of `iindex`].
    """
    print("do general popular negative sampling")

    # check first
    data_dir = os.path.join(args.data_root, args.name)
    os.makedirs(data_dir, exist_ok=True)

    # load materials
    print("- load materials")
    with open(os.path.join(data_dir, 'df_rows.pkl'), 'rb') as fp:
        df_rows = pickle.load(fp)
    with open(os.path.join(data_dir, 'uid2uindex.pkl'), 'rb') as fp:
        uid2uindex = pickle.load(fp)
        user_count = len(uid2uindex)

    # reorder items
    print("- reorder items")
    reordered_iindices = list(df_rows.groupby(['iindex']).size().sort_values().index)[::-1]

    # sample popular negatives
    print("- sample popular negatives")
    ns = {}
    for uindex in tqdm(list(range(1, user_count + 1)), desc="* sampling"):
        seen_iindices = set(df_rows[df_rows['uindex'] == uindex]['iindex'])
        sampled_iindices = []
        for iindex in reordered_iindices:
            if len(sampled_iindices) == num_samples:
                break
            if iindex in seen_iindices:
                continue
            sampled_iindices.append(iindex)
        ns[uindex] = sampled_iindices

    # save sampled popular nagetives
    print("- save sampled popular nagetives")
    with open(os.path.join(data_dir, 'ns_popular.pkl'), 'wb') as fp:
        pickle.dump(ns, fp)


def task_prepare_ml1m(args):
    print("task: prepare ml1m")

    # check first
    data_dir = os.path.join(args.data_root, args.name)
    os.makedirs(data_dir, exist_ok=True)

    # load data
    print("- load data")
    df_rows = pd.read_csv(os.path.join(args.rough_root, args.name, 'ratings.dat'), sep='::', header=None, engine='python')
    df_rows.columns = ['uid', 'iid', 'rating', 'timestamp']

    # make implicit
    print("- make implicit")
    df_rows = df_rows[df_rows['rating'] >= 1]
    df_rows = df_rows.drop(columns=['rating'])

    # do the rest
    do_general_preprocessing(args, df_rows)
    do_general_random_negative_sampling(args)
    do_general_popular_negative_sampling(args)

    print("done")


def task_prepare_ml20m(args):
    print("task: prepare ml20m")

    # check first
    data_dir = os.path.join(args.data_root, args.name)
    os.makedirs(data_dir, exist_ok=True)

    # load data
    print("- load data")
    df_rows = pd.read_csv(os.path.join(args.rough_root, args.name, 'ratings.csv'), sep=',', header=0, engine='python')
    df_rows.columns = ['uid', 'iid', 'rating', 'timestamp']

    # make implicit
    print("- make implicit")
    df_rows = df_rows[df_rows['rating'] >= 4]
    df_rows = df_rows.drop(columns=['rating'])

    # do the rest
    do_general_preprocessing(args, df_rows)
    do_general_random_negative_sampling(args)
    do_general_popular_negative_sampling(args)

    print("done")


def task_prepare_steam2(args):
    print("task: prepare steam2")

    # check first
    data_dir = os.path.join(args.data_root, args.name)
    os.makedirs(data_dir, exist_ok=True)

    # load data
    print("- load data")
    rows = []
    with open(os.path.join(args.rough_root, 'steam', 'steam_reviews.json')) as fp:
        raw = fp.read().strip()
        lines = raw.split('\n')
        for line in tqdm(lines, desc="* loading"):
            one = eval(line)
            uid = one['username']
            iid = one['product_id']
            dto = dt.strptime(one['date'], "%Y-%m-%d")
            timestamp = int(time.mktime(dto.timetuple()))
            rows.append((uid, iid, timestamp))
    df_rows = pd.DataFrame(rows)
    df_rows.columns = ['uid', 'iid', 'timestamp']

    # do the rest
    do_general_preprocessing(args, df_rows)
    do_general_random_negative_sampling(args)
    do_general_popular_negative_sampling(args)

    print("done")


def task_count_stats(args):
    print("task: count stats")

    print('\t'.join([
        "dname",
        "#user",
        "#item",
        "#row",
        "density",
        "ic_25",
        "ic_50",
        "ic_75",
        "ic_95",
        "sc_25",
        "sc_50",
        "sc_75",
        "sc_95",
        "cc_25",
        "cc_50",
        "cc_75",
        "cc_95",
    ]))
    for dname in DNAMES:
        data_dir = os.path.join(args.data_root, dname)

        # load data
        with open(os.path.join(data_dir, 'uid2uindex.pkl'), 'rb') as fp:
            uid2uindex = pickle.load(fp)
        with open(os.path.join(data_dir, 'iid2iindex.pkl'), 'rb') as fp:
            iid2iindex = pickle.load(fp)
        with open(os.path.join(data_dir, 'df_rows.pkl'), 'rb') as fp:
            df_rows = pickle.load(fp)

        # get density
        num_users = len(uid2uindex)
        num_items = len(iid2iindex)
        num_rows = len(df_rows)
        density = num_rows / num_users / num_items

        # get item count per user
        icounts = df_rows.groupby('uindex').size().to_numpy()  # allow duplicates

        # get session count per user
        scounts = df_rows.groupby('uindex').agg({'sid': 'nunique'})['sid'].to_numpy()

        # get item count per user-session
        ccounts = df_rows.groupby(['uindex', 'sid']).size().to_numpy()

        # report
        print('\t'.join([
            dname,
            str(num_users),
            str(num_items),
            str(num_rows),
            f"{100 * density:.04f}%",
            str(int(np.percentile(icounts, 25))),
            str(int(np.percentile(icounts, 50))),
            str(int(np.percentile(icounts, 75))),
            str(int(np.percentile(icounts, 95))),
            str(int(np.percentile(scounts, 25))),
            str(int(np.percentile(scounts, 50))),
            str(int(np.percentile(scounts, 75))),
            str(int(np.percentile(scounts, 95))),
            str(int(np.percentile(ccounts, 25))),
            str(int(np.percentile(ccounts, 50))),
            str(int(np.percentile(ccounts, 75))),
            str(int(np.percentile(ccounts, 95))),
        ]))


if __name__ == '__main__':
    args = parse_args()
    globals()[f'task_{args.task}_{args.name}'](args)
