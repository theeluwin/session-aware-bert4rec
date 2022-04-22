import torch

from typing import (
    Any,
    List,
    Dict,
)


__all__ = (
    'METRIC_NAMES',
    'calc_batch_rec_metrics_per_k',
)


# METRIC_NAMES = ('HR', 'Recall', 'NDCG', 'AUROC')
METRIC_NAMES = ('HR', 'Recall', 'NDCG')


def calc_batch_rec_metrics_per_k(rankers: torch.LongTensor,
                                 labels: torch.LongTensor,
                                 ks: List[int]
                                 ) -> Dict[str, Any]:
    """
        Args:
            rankers: LongTensor, (b x C), pos per rank (0 to C-1)
            labels: LongTensor, (b x C), binary per pos (0 or 1)
            ks: list of top-k values

        Returns:
            a dict of various metrics.
            keys are 'count', 'mean', 'values'.
            names are 'HR', 'Recall', 'NDCG'.

        put'em all in the same device.
    """

    # prepare
    batch_size = rankers.size(0)
    metrics: Dict[str, Any] = {
        'count': batch_size,
        'values': {},
        'mean': {},
    }
    answer_count = labels.sum(1)
    device = labels.device
    ks = sorted(ks, reverse=True)

    # for each k
    for k in ks:
        rankers_at_k = rankers[:, :k]
        hit_per_pos = labels.gather(1, rankers_at_k)

        # hr
        hrs = hit_per_pos.sum(1).bool().float()
        hrs_list = list(hrs.detach().cpu().numpy())
        metrics['values'][f'HR@{k}'] = hrs_list
        metrics['mean'][f'HR@{k}'] = sum(hrs_list) / batch_size

        # recall
        divisor = torch.min(
            torch.Tensor([k]).to(device),
            answer_count,
        )
        recalls = (hit_per_pos.sum(1) / divisor.float())
        recalls_list = list(recalls.detach().cpu().numpy())
        metrics['values'][f'Recall@{k}'] = recalls_list
        metrics['mean'][f'Recall@{k}'] = sum(recalls_list) / batch_size

        # ndcg
        positions = torch.arange(1, k + 1).to(device).float()
        weights = 1 / (positions + 1).log2()
        dcg = (hit_per_pos * weights).sum(1)
        idcg = torch.Tensor([weights[:min(n, k)].sum() for n in answer_count]).to(device)
        ndcgs = dcg / idcg
        ndcgs_list = list(ndcgs.detach().cpu().numpy())
        metrics['values'][f'NDCG@{k}'] = ndcgs_list
        metrics['mean'][f'NDCG@{k}'] = sum(ndcgs_list) / batch_size

        """
        # auroc
        positions = torch.arange(k)
        flag_per_pos = hit_per_pos.detach().cpu().bool()
        aurocs = []
        for b in range(batch_size):
            poss = positions[flag_per_pos[b]]
            negs = positions[~flag_per_pos[b]]
            poss_count = poss.size(0)
            negs_count = negs.size(0)
            total_count = poss_count * negs_count
            if not negs_count:
                auroc = 1.0
            elif not poss_count:
                auroc = 0.0
            else:
                rocs = [(pos < negs).int().sum() for pos in poss]
                auroc = sum(rocs) / total_count
                auroc = float(auroc)
            aurocs.append(auroc)
        metrics['values'][f'AUROC@{k}'] = aurocs
        metrics['mean'][f'AUROC@{k}'] = sum(aurocs) / batch_size
        """

    return metrics
