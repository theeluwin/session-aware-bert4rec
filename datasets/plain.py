# Plain: Plain User Rows

import os
import pickle

from torch import LongTensor as LT
from torch.utils.data import Dataset


__all__ = (
    'PlainTrainDataset',
    'PlainEvalDataset',
)


class PlainTrainDataset(Dataset):

    data_root = 'data'

    def __init__(self, name: str):

        # params
        self.name = name

        # load data
        with open(os.path.join(self.data_root, name, 'iid2iindex.pkl'), 'rb') as fp:
            self.iid2iindex = pickle.load(fp)
        with open(os.path.join(self.data_root, name, 'train.pkl'), 'rb') as fp:
            self.uindex2rows_train = pickle.load(fp)

        # settle down
        self.uindices = list(self.uindex2rows_train.keys())
        self.num_items = len(self.iid2iindex)

    def __len__(self):
        return len(self.uindices)

    def __getitem__(self, index):
        uindex = self.uindices[index]
        rows = self.uindex2rows_train[uindex]
        return {
            'uindex': uindex,
            'rows': rows,
        }

    @staticmethod
    def collate_fn(samples):
        uindex = [sample['uindex'] for sample in samples]
        rows = [sample['rows'] for sample in samples]
        return {
            'uindex': uindex,
            'rows': rows,
        }


class PlainEvalDataset(Dataset):

    data_root = 'data'

    def __init__(self,
                 name: str,
                 target: str,  # 'valid', 'test'
                 ns: str,  # 'random', 'popular', 'all'
                 ):

        # params
        self.name = name
        self.target = target
        self.ns = ns

        # load data
        with open(os.path.join(self.data_root, name, 'iid2iindex.pkl'), 'rb') as fp:
            self.iid2iindex = pickle.load(fp)
        with open(os.path.join(self.data_root, name, f'{target}.pkl'), 'rb') as fp:
            self.uindex2rows_eval = pickle.load(fp)

        # in case of ns
        if ns != 'all':
            with open(os.path.join(self.data_root, name, f'ns_{ns}.pkl'), 'rb') as fp:
                self.uindex2negatives = pickle.load(fp)

        # settle down
        self.uindices = list(self.uindex2rows_eval.keys())
        self.num_items = len(self.iid2iindex)

    def __len__(self):
        return len(self.uindices)

    def __getitem__(self, index):

        # data point
        uindex = self.uindices[index]
        rows_eval = self.uindex2rows_eval[uindex]

        # get eval row
        answer, _, _ = rows_eval[0]

        # candidates and labels
        if self.ns != 'all':
            negatives = self.uindex2negatives[uindex]
            cands = [answer] + negatives
            labels = [1] + [0] * len(negatives)
        else:
            cands = list(range(1, self.num_items))
            labels = [0] * self.num_items
            labels[answer - 1] = 1

        return {
            'uindex': uindex,
            'cands': LT(cands),
            'labels': LT(labels),
        }
