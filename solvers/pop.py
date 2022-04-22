import os
import json
import pickle

import torch

from typing import List

from tqdm import tqdm
from torch import LongTensor as LT
from torch.utils.data import DataLoader

from datasets import (
    PlainTrainDataset,
    PlainEvalDataset,
)
from tools.metrics import calc_batch_rec_metrics_per_k

from .base import BaseSolver


__all__ = (
    'POPSolver',
)


class POPSolver(BaseSolver):

    def __init__(self, config: dict):
        C = config

        # before super
        with open(os.path.join(C['envs']['DATA_ROOT'], C['dataset'], 'iid2iindex.pkl'), 'rb') as fp:
            self.iid2iindex = pickle.load(fp)
        self.num_items = len(self.iid2iindex)

        super().__init__(config)

    # over-override
    def init_device(self) -> None:
        self.device = torch.device('cpu')

    # over-override
    def init_optimizer(self) -> None:
        pass

    # over-override
    def init_scheduler(self) -> None:
        pass

    # over-override
    def load_model(self, purpose: str) -> None:
        C = self.config

        if purpose == 'test':
            self.logger.info(f"loading model: {C['name']}")
            with open(self.model_path, 'rb') as fp:
                self.iindex2popularity = pickle.load(fp)

        elif purpose == 'train':
            if os.path.isfile(self.model_path):
                self.logger.info(f"loading model: {C['name']}")
                with open(self.model_path, 'rb') as fp:
                    self.iindex2popularity = pickle.load(fp)
            else:
                self.logger.info(f"preparing new model from scratch: {C['name']}")

    # over-override
    def set_model_mode(self, purpose: str) -> None:
        pass

    # over-override
    def solve(self) -> None:
        C = self.config
        name = C['name']
        print(f"solve {name} start")

        # report new session
        self.logger.info("-- new solve session started --")

        # report model parameters
        with open(os.path.join(self.data_dir, 'meta.json'), 'w') as fp:
            json.dump({'num_params': 0}, fp)

        # solve loop
        self.load_model('train')
        self.solve_train(1)

        # solve end
        self.load_model('test')
        self.solve_test()

        # for the paper
        line = []
        with open(os.path.join(self.data_dir, 'results_random.json')) as fp:
            results = json.load(fp)
            line.append(results['Recall@10'])
            line.append(results['NDCG@10'])
        with open(os.path.join(self.data_dir, 'results_popular.json')) as fp:
            results = json.load(fp)
            line.append(results['Recall@10'])
            line.append(results['NDCG@10'])
        with open(os.path.join(self.data_dir, 'results_all.json')) as fp:
            results = json.load(fp)
            line.append(results['Recall@10'])
            line.append(results['NDCG@10'])
        with open(os.path.join(self.data_dir, 'meta.json')) as fp:
            meta = json.load(fp)
            line.append(meta['num_params'])
        self.logger.info('\n' + '\t'.join([str(value) for value in line]))

        print(f"solve {name} end")

    # over-override
    def train_one_epoch(self, epoch: int) -> None:
        self.logger.info(f"<train one epoch (epoch: {epoch})>")

        # prepare
        self.set_model_mode('train')

        # loop
        pbar = tqdm(self.train_dataloader)
        pbar.set_description(f"[epoch {epoch} train]")
        for batch in pbar:

            # get loss
            _ = self.calculate_loss(batch)

        pbar.close()

        # epoch end
        self.logger.info('\t'.join([
            f"[epoch {epoch}]",
            "train done"
        ]))

    # over-override
    def evaluate_on_valid(self, epoch: int) -> None:
        self.logger.info(f"<evaluate on valid (epoch: {epoch})>")
        C = self.config

        # prepare
        self.set_model_mode('eval')
        ks = sorted(C['metric']['ks'])
        pivot = C['metric']['pivot']

        # init
        result_values = []

        # loop
        pbar = tqdm(self.valid_dataloader)
        pbar.set_description(f"[epoch {epoch} valid]")
        with torch.no_grad():
            for batch in pbar:

                # get rankers
                rankers = self.calculate_rankers(batch)

                # evaluate
                labels = batch['labels'].to(self.device)
                batch_results = calc_batch_rec_metrics_per_k(rankers, labels, ks)  # type: ignore
                result_values.extend(batch_results['values'][pivot])

                # verbose
                pbar.set_postfix(pivot_score=f'{sum(result_values) / len(result_values):.4f}')

        pbar.close()

        # report
        pivot_score = sum(result_values) / len(result_values)
        self.logger.info('\t'.join([
            f"[epoch {epoch}]",
            f"valid pivot score: {pivot_score:.4f}"
        ]))

        # save model
        with open(self.model_path, 'wb') as fp:
            pickle.dump(self.iindex2popularity, fp)

    # override
    def init_model(self) -> None:
        self.iindex2popularity: List[int] = [0 for _ in range(self.num_items + 1)]  # type: ignore

    # override
    def init_criterion(self) -> None:
        pass

    # override
    def init_dataloader(self) -> None:
        C = self.config
        name = C['dataset']
        self.train_dataloader = DataLoader(
            PlainTrainDataset(name=name),
            batch_size=C['train']['batch_size'],
            shuffle=True,
            num_workers=C['envs']['CPU_COUNT'],
            pin_memory=False,
            drop_last=False,
            collate_fn=PlainTrainDataset.collate_fn
        )
        self.valid_dataloader = DataLoader(
            PlainEvalDataset(
                name=name,
                target='valid',
                ns='random'
            ),
            batch_size=C['train']['batch_size'],
            shuffle=False,
            num_workers=C['envs']['CPU_COUNT'],
            pin_memory=False,
            drop_last=False
        )
        self.test_ns_random_dataloader = DataLoader(
            PlainEvalDataset(
                name=name,
                target='test',
                ns='random'
            ),
            batch_size=C['train']['batch_size'],
            shuffle=False,
            num_workers=C['envs']['CPU_COUNT'],
            pin_memory=False,
            drop_last=False
        )
        self.test_ns_popular_dataloader = DataLoader(
            PlainEvalDataset(
                name=name,
                target='test',
                ns='popular'
            ),
            batch_size=C['train']['batch_size'],
            shuffle=False,
            num_workers=C['envs']['CPU_COUNT'],
            pin_memory=False,
            drop_last=False
        )
        self.test_ns_all_dataloader = DataLoader(
            PlainEvalDataset(
                name=name,
                target='test',
                ns='all'
            ),
            batch_size=C['train']['batch_size'],
            shuffle=False,
            num_workers=C['envs']['CPU_COUNT'],
            pin_memory=False,
            drop_last=False
        )

    # override (special: inplace update)
    def calculate_loss(self, batch):

        for _, rows in zip(batch['uindex'], batch['rows']):

            # just plain counting
            for iindex, _, _ in rows:
                self.iindex2popularity[iindex] = self.iindex2popularity[iindex] + 1

        return 0

    # override
    def calculate_rankers(self, batch):

        # device
        cands = batch['cands'].to(self.device)  # b x C

        # forward
        batch_size = cands.size(0)
        logits = LT(self.iindex2popularity).repeat(batch_size, 1)

        # gather
        scores = logits.gather(1, cands)  # b x C
        rankers = scores.argsort(dim=1, descending=True)

        return rankers
