import os
import json
import logging

import torch

from typing import (
    List,
    Dict,
)

from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import (
    Dataset,
    DataLoader,
)

from tools.metrics import (
    METRIC_NAMES,
    calc_batch_rec_metrics_per_k,
)
from tools.schedulers import CosineAnnealingWarmupRestarts


class BaseSolver:

    def __init__(self, config: dict):
        self.config = config
        self.init_path()
        self.init_logger()
        self.init_device()
        self.init_model()
        self.init_optimizer()
        self.init_scheduler()
        self.init_criterion()
        self.init_dataloader()

    def init_path(self) -> None:
        C = self.config

        self.log_dir = os.path.join(C['run_dir'], 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger_path = os.path.join(self.log_dir, 'solver.log')
        self.writer_path = os.path.join(self.log_dir, 'scalars.json')

        self.pth_dir = os.path.join(C['run_dir'], 'pths')
        os.makedirs(self.pth_dir, exist_ok=True)
        self.model_path = os.path.join(self.pth_dir, 'model.pth')
        self.check_path = os.path.join(self.pth_dir, 'checkpoint.pth')

        self.data_dir = os.path.join(C['run_dir'], 'data')
        os.makedirs(self.data_dir, exist_ok=True)

    def init_logger(self) -> None:
        C = self.config

        # tx writer
        self.writer = SummaryWriter(self.log_dir)

        # logging logger
        self.logger = logging.getLogger(C['name'])
        self.logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(self.logger_path)
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter("[%(asctime)s] %(message)s")
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def init_device(self) -> None:
        C = self.config
        if C['envs']['GPU_COUNT']:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def init_optimizer(self) -> None:
        CTO = self.config['train']['optimizer']
        if CTO['algorithm'] == 'sgd':
            self.optimizer = torch.optim.SGD(  # type: ignore
                self.model.parameters(),  # type: ignore
                lr=CTO['lr'],
                momentum=CTO['momentum'],
                weight_decay=CTO['weight_decay']
            )
        elif CTO['algorithm'] == 'adam':
            self.optimizer = torch.optim.Adam(  # type: ignore
                self.model.parameters(),  # type: ignore
                lr=CTO['lr'],
                betas=(
                    CTO['beta1'],
                    CTO['beta2']
                ),
                weight_decay=CTO['weight_decay'],
                amsgrad=CTO['amsgrad']
            )
        elif CTO['algorithm'] == 'adamw':
            self.optimizer = torch.optim.AdamW(  # type: ignore
                self.model.parameters(),  # type: ignore
                lr=CTO['lr'],
                betas=(
                    CTO['beta1'],
                    CTO['beta2']
                ),
                weight_decay=CTO['weight_decay'],
                amsgrad=CTO['amsgrad']
            )
        else:
            raise NotImplementedError

    def init_scheduler(self) -> None:
        CTS = self.config['train']['scheduler']
        if CTS['algorithm'] == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(  # type: ignore
                self.optimizer,
                step_size=CTS['step_size'],
                gamma=CTS['gamma']
            )
        elif CTS['algorithm'] == 'cosine':
            self.scheduler = CosineAnnealingWarmupRestarts(  # type: ignore
                self.optimizer,
                first_cycle_steps=CTS['first_cycle_steps'],
                cycle_mult=CTS['cycle_mult'],
                max_lr=CTS['max_lr'],
                min_lr=CTS['min_lr'],
                warmup_steps=CTS['warmup_steps'],
                gamma=CTS['gamma']
            )
        elif CTS['algorithm'] is None:
            self.scheduler = None  # type: ignore
        else:
            raise NotImplementedError

    def load_model(self, purpose: str) -> None:
        C = self.config

        if purpose == 'test':
            self.logger.info(f"loading model with the best score: {C['name']}")
            model_dict = torch.load(self.model_path)  # if exception, just raise it
            self.model.load_state_dict(model_dict)  # type: ignore

        elif purpose == 'train':
            if os.path.isfile(self.check_path):
                self.logger.info(f"loading model from the checkpoint: {C['name']}")
                check = torch.load(self.check_path)  # if exception, just raise it
                self.start_epoch = check['epoch'] + 1
                self.best_score = check.get('best_score', None)
                self.model.load_state_dict(check['model'])  # type: ignore
                self.optimizer.load_state_dict(check['optimizer'])
                if self.scheduler is not None and self.start_epoch > 1:
                    for _ in range(self.start_epoch - 1):
                        self.scheduler.step()
            else:
                self.logger.info(f"preparing new model from scratch: {C['name']}")
                self.start_epoch = 1
                self.best_score = None

    def set_model_mode(self, purpose: str) -> None:
        if purpose == 'eval':
            self.model = self.model.eval()  # type: ignore
        elif purpose == 'train':
            self.model = self.model.train()

    def solve(self) -> None:
        C = self.config
        name = C['name']
        print(f"solve {name} start")

        # report new session
        self.logger.info("-- new solve session started --")

        # report model parameters
        numels = []
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                numels.append(parameter.numel())
        num_params = sum(numels)
        self.logger.info(f"total parameters:\t{num_params}")
        with open(os.path.join(self.data_dir, 'meta.json'), 'w') as fp:
            json.dump({'num_params': num_params}, fp)

        # solve loop
        self.early_stop = False
        self.patience_counter = 0
        self.load_model('train')
        for epoch in range(self.start_epoch, C['train']['epoch'] + 1):
            self.solve_train(epoch)
            if self.scheduler is not None:
                self.scheduler.step()
            if self.early_stop:
                break

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

        # save writer
        self.writer.export_scalars_to_json(self.writer_path)
        self.writer.close()

        print(f"solve {name} end")

    def solve_train(self, epoch: int) -> None:
        self.train_one_epoch(epoch)
        self.evaluate_on_valid(epoch)

    def solve_test(self) -> None:
        self.evaluate_on_test('random')
        self.evaluate_on_test('popular')
        self.evaluate_on_test('all')

    def train_one_epoch(self, epoch: int) -> None:
        self.logger.info(f"<train one epoch (epoch: {epoch})>")

        # prepare
        self.set_model_mode('train')
        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        # loop
        pbar = tqdm(self.train_dataloader)
        pbar.set_description(f"[epoch {epoch} train]")
        for batch in pbar:

            # get loss
            loss = self.calculate_loss(batch)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # step end
            epoch_loss_sum += float(loss.data)
            epoch_loss_count += 1
            pbar.set_postfix(loss=f'{epoch_loss_sum / epoch_loss_count:.4f}')

        pbar.close()

        # epoch end
        epoch_loss = epoch_loss_sum / epoch_loss_count
        self.logger.info('\t'.join([
            f"[epoch {epoch}]",
            f"train loss: {epoch_loss:.4f}"
        ]))
        self.writer.add_scalar('train_loss', epoch_loss, epoch)

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
        self.writer.add_scalar('valid_pivot_score', pivot_score, epoch)

        # save best
        if self.best_score is None or self.best_score < pivot_score:
            self.best_score = pivot_score
            torch.save(self.model.state_dict(), self.model_path)
            self.logger.info(f"updated best model at epoch {epoch} with pivot score {pivot_score}")
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        if self.patience_counter >= C['train']['patience']:
            self.logger.info(f"no update for {self.patience_counter} epochs, thus early stopping")
            self.early_stop = True

        # save checkpoint
        torch.save({
            'epoch': epoch,
            'best_score': self.best_score,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, self.check_path)

    def evaluate_on_test(self, ns: str = 'random') -> None:
        self.logger.info(f"<evaluate on test ns {ns}>")
        C = self.config

        # prepare
        self.set_model_mode('eval')
        ks = sorted(C['metric']['ks'])

        # init
        results_values: Dict[str, List] = {}
        for k in ks:
            for metric_name in METRIC_NAMES:
                results_values[f'{metric_name}@{k}'] = []
        metric_keys = list(results_values.keys())
        if ns == 'random':
            dataloader = self.test_ns_random_dataloader
        elif ns == 'popular':
            dataloader = self.test_ns_popular_dataloader
        elif ns == 'all':
            dataloader = self.test_ns_all_dataloader

        # loop
        pbar = tqdm(dataloader)
        pbar.set_description(f"[test ns {ns}]")
        with torch.no_grad():
            for batch in pbar:

                # get rankers
                rankers = self.calculate_rankers(batch)

                # evaluate
                labels = batch['labels'].to(self.device)
                batch_results = calc_batch_rec_metrics_per_k(rankers, labels, ks)  # type: ignore
                for metric_key in metric_keys:
                    results_values[metric_key].extend(batch_results['values'][metric_key])

        pbar.close()

        # average
        results_mean: Dict[str, float] = {}
        for metric_key in metric_keys:
            values = results_values[metric_key]
            results_mean[metric_key] = sum(values) / len(values)

        # report
        reports = ["metric report:"]
        reports.append('\t'.join(['k'] + list(METRIC_NAMES)))
        for k in ks:
            row = [str(k)]
            for metric_name in METRIC_NAMES:
                result = results_mean[f'{metric_name}@{k}']
                row.append(f"{result:.05f}",)
            reports.append('\t'.join(row))
        self.logger.info('\n'.join(reports))
        with open(os.path.join(self.data_dir, f'results_{ns}.json'), 'w') as fp:
            json.dump(results_mean, fp)

    # override this
    def init_model(self) -> None:
        self.model = torch.nn.Module()
        raise NotImplementedError

    # override this
    def init_criterion(self) -> None:
        raise NotImplementedError

    # override this
    def init_dataloader(self) -> None:
        self.train_dataloader = DataLoader(Dataset())  # type: ignore
        self.valid_dataloader = DataLoader(Dataset())  # type: ignore
        self.test_ns_random_dataloader = DataLoader(Dataset())  # type: ignore
        self.test_ns_popular_dataloader = DataLoader(Dataset())  # type: ignore
        self.test_ns_all_dataloader = DataLoader(Dataset())  # type: ignore
        raise NotImplementedError

    # override this
    def calculate_loss(self, batch):
        raise NotImplementedError

    # override this
    def calculate_rankers(self, batch):
        raise NotImplementedError
