import os
import pickle

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from models import BERT4Rec
from datasets import (
    MLMTrainDataset,
    MLMEvalDataset,
)

from .base import BaseSolver


__all__ = (
    'BERT4RecSolver',
)


class BERT4RecSolver(BaseSolver):

    def __init__(self, config: dict) -> None:
        C = config

        # before super
        with open(os.path.join(C['envs']['DATA_ROOT'], C['dataset'], 'iid2iindex.pkl'), 'rb') as fp:
            self.iid2iindex = pickle.load(fp)
        self.num_items = len(self.iid2iindex)

        super().__init__(config)

    # override
    def init_model(self) -> None:
        C = self.config
        cm = C['model']
        self.model = BERT4Rec(
            num_items=self.num_items,
            sequence_len=C['dataloader']['sequence_len'],
            max_num_segments=C['dataloader']['max_num_segments'],
            use_session_token=C['dataloader']['use_session_token'],
            num_layers=cm['num_layers'],
            hidden_dim=cm['hidden_dim'],
            temporal_dim=cm['temporal_dim'],
            num_heads=cm['num_heads'],
            dropout_prob=cm['dropout_prob'],
            random_seed=cm['random_seed']
        ).to(self.device)

    # override
    def init_criterion(self) -> None:
        self.ce_losser = CrossEntropyLoss(ignore_index=0)

    # override
    def init_dataloader(self) -> None:
        C = self.config
        name = C['dataset']
        sequence_len = C['dataloader']['sequence_len']
        max_num_segments = C['dataloader']['max_num_segments']
        use_session_token = C['dataloader']['use_session_token']
        self.train_dataloader = DataLoader(
            MLMTrainDataset(
                name=name,
                sequence_len=sequence_len,
                max_num_segments=max_num_segments,
                random_cut_prob=C['dataloader']['random_cut_prob'],
                mask_prob=C['dataloader']['mlm_mask_prob'],
                use_session_token=use_session_token,
                random_seed=C['dataloader']['random_seed']
            ),
            batch_size=C['train']['batch_size'],
            shuffle=True,
            num_workers=C['envs']['CPU_COUNT'],
            pin_memory=True,
            drop_last=False
        )
        self.valid_dataloader = DataLoader(
            MLMEvalDataset(
                name=name,
                target='valid',
                ns='random',
                sequence_len=sequence_len,
                max_num_segments=max_num_segments,
                use_session_token=use_session_token
            ),
            batch_size=C['train']['batch_size'],
            shuffle=False,
            num_workers=C['envs']['CPU_COUNT'],
            pin_memory=True,
            drop_last=False
        )
        self.test_ns_random_dataloader = DataLoader(
            MLMEvalDataset(
                name=name,
                target='test',
                ns='random',
                sequence_len=sequence_len,
                max_num_segments=max_num_segments,
                use_session_token=use_session_token
            ),
            batch_size=C['train']['batch_size'],
            shuffle=False,
            num_workers=C['envs']['CPU_COUNT'],
            pin_memory=True,
            drop_last=False,
        )
        self.test_ns_popular_dataloader = DataLoader(
            MLMEvalDataset(
                name=name,
                target='test',
                ns='popular',
                sequence_len=sequence_len,
                max_num_segments=max_num_segments,
                use_session_token=use_session_token
            ),
            batch_size=C['train']['batch_size'],
            shuffle=False,
            num_workers=C['envs']['CPU_COUNT'],
            pin_memory=True,
            drop_last=False,
        )
        self.test_ns_all_dataloader = DataLoader(
            MLMEvalDataset(
                name=name,
                target='test',
                ns='all',
                sequence_len=sequence_len,
                max_num_segments=max_num_segments,
                use_session_token=use_session_token
            ),
            batch_size=C['train']['batch_size'],
            shuffle=False,
            num_workers=C['envs']['CPU_COUNT'],
            pin_memory=True,
            drop_last=False
        )

    # override
    def calculate_loss(self, batch):

        # device
        tokens = batch['tokens'].to(self.device)  # b x L
        labels = batch['labels'].to(self.device)  # b x L

        # use segments
        if self.config['dataloader']['max_num_segments']:
            segments = batch['segments'].to(self.device)  # b x L
        else:
            segments = None

        # use stamps
        if self.config['model']['temporal_dim']:
            stamps = batch['stamps'].to(self.device)  # b x L
        else:
            stamps = None

        # forward
        logits = self.model(tokens, segments=segments, stamps=stamps)  # b x L x (V + 1)

        # loss
        logits = logits.view(-1, logits.size(-1))  # bL x (V + 1)
        labels = labels.view(-1)  # bL
        loss = self.ce_losser(logits, labels)

        return loss

    # override
    def calculate_rankers(self, batch):

        # device
        tokens = batch['tokens'].to(self.device)  # b x L
        cands = batch['cands'].to(self.device)  # b x C

        # use segments
        if self.config['dataloader']['max_num_segments']:
            segments = batch['segments'].to(self.device)  # b x L
        else:
            segments = None

        # use stamps
        if self.config['model']['temporal_dim']:
            stamps = batch['stamps'].to(self.device)  # b x L
        else:
            stamps = None

        # forward
        logits = self.model(tokens, segments=segments, stamps=stamps)  # b x L x (V + 1)

        # gather
        logits = logits[:, -1, :]  # b x (V + 1)
        scores = logits.gather(1, cands)  # b x C
        rankers = scores.argsort(dim=1, descending=True)

        return rankers
