# Session-aware BERT4Rec

Official repository for "Exploiting Session Information in BERT-based Session-aware Sequential Recommendation", SIGIR 2022 short.

Everything in the paper is implemented (including vanilla BERT4Rec and SASRec), and can be reproduced.

## Usage

### 1. Build Docker

```bash
./scripts/build.sh
```

### 2. Download dataset

Download corresponding datasets into some directory, such as `./roughs`.

For [Steam](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data) dataset, use version 2.

Rename datasets: `'ml1m'` for MovieLens-1M, `'ml20m'` for MovieLens-2M, `'steam2'` for Steam.

### 3. Preprocess

* `--rough_root`: for original dataset files
* `--data_root`: for processed data files

```bash
python preprocess.py prepare ml1m --data_root ./data --rough_root ./roughs
python preprocess.py prepare ml20m --data_root ./data --rough_root ./roughs
python preprocess.py prepare steam2 --data_root ./data --rough_root ./roughs
```

For some stats:

```bash
python preprocess.py count stats --data_root ./data --rough_root ./roughs > dstats.tsv
```

### 4. Run

See default configuration setting in `entry.py`.

To modify configuration, make some directory under `runs/` like `./runs/ml1m/bert4rec/vanilla/`, and create `config.json`.

#### Sample Run Script

My `x0.sh` file that uses GPU No. 0:

```bash
runpy () {
    docker run \
        -it \
        --rm \
        --init \
        --gpus '"device=0"' \
        --shm-size 16G \
        --volume="$HOME/.cache/torch:/root/.cache/torch" \
        --volume="$PWD:/workspace" \
        session-aware-bert4rec \
        python "$@"
}

runpy entry.py ml1m/bert4rec/vanilla
```

## Terminologies

The `df_` prefix always means DataFrame from Pandas.

* `uid` (str|int): User ID (unique).
* `iid` (str|int): Item ID (unique).
* `sid` (str|int): Session ID (unique), used only for session separation.
* `uindex` (int): mapped index number of User ID, 1 ~ n.
* `iindex` (int): mapped index number of Item ID, 1 ~ m.
* `timestamp` (int): UNIX timestamp.

## Data Files

After preprocessing, we'll have followings in each `data/:dataset_name/` directory.

* `uid2uindex.pkl` (dict): {`uid` &rightarrow; `uindex`}.
* `iid2iindex.pkl` (dict): {`iid` &rightarrow; `iindex`}.
* `df_rows.pkl` (df): column of (`uindex`, `iindex`, `sid`, `timestamp`), with no index.
* `train.pkl` (dict): {`uindex` &rightarrow; [list of (`iindex`, `sid`, `timestamp`)]}.
* `valid.pkl` (dict): {`uindex` &rightarrow; [list of (`iindex`, `sid`, `timestamp`)]}.
* `test.pkl` (dict): {`uindex` &rightarrow; [list of (`iindex`, `sid`, `timestamp`)]}.
* `ns_random.pkl` (dict): {`uindex` -> [list of `iindex`]}.
* `ns_popular.pkl` (dict): {`uindex` -> [list of `iindex`]}.

## Code References

* [FeiSun/BERT4Rec](https://github.com/FeiSun/BERT4Rec)
* [jaywonchung/BERT4Rec-VAE-Pytorch](https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch)
