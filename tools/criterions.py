import torch as t
import torch.nn as nn


__all__ = (
    'ContrastiveCriterion',
)


class ContrastiveCriterion(nn.Module):

    def __init__(self,
                 ce_losser: nn.CrossEntropyLoss,
                 tau: float = 0.1
                 ):
        super(ContrastiveCriterion, self).__init__()
        self.ce_losser = ce_losser
        self.tau = tau

    def forward(self,
                vec_anchor: t.Tensor,  # (b, v)
                vec_pos: t.Tensor,  # (b, v)
                vec_negs: t.Tensor  # (b, k, v)
                ) -> t.Tensor:

        # vec_pos: (b, v) -> (b, 1, v)
        # vec_negs: (b, k, v)
        # vec_others: (b, 1 + k, v)
        vec_others = t.cat([vec_pos.unsqueeze(dim=1), vec_negs], dim=1)

        # vec_anchor: (b, v) -> (b, v, 1)
        # bmm: (b, 1 + k, v) @ (b, v, 1) -> (b, 1 + k, 1)
        # squeeze: (b, 1 + k, 1) -> (b, 1 + k)
        logit = t.bmm(vec_others, vec_anchor.unsqueeze(dim=2)).squeeze(dim=2)

        # target is always 0
        with t.no_grad():
            batch_size = vec_anchor.size(0)
            target = t.LongTensor([0 for _ in range(batch_size)]).cuda()  # like, class 0

        # get loss with given CE
        loss = self.ce_losser(logit / self.tau, target)
        return loss
