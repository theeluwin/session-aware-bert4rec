import torch
import torch.nn as nn

from typing import Optional

from math import (
    pi,
    sqrt,
)

from torch import Tensor
from torch.nn.functional import softmax


__all__ = (
    'GELU',
    'LayerNorm',
    'Attention',
    'MultiHeadedAttention',
    'SublayerConnection',
    'PositionWiseFeedForward',
    'Transformer',
)


class GELU(nn.Module):

    def forward(self, x: Tensor):
        return 0.5 * x * (1 + torch.tanh(sqrt(2 / pi) * (x + 0.044715 * torch.pow(x, 3))))


class LayerNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        # params
        self.dim = dim
        self.eps = eps

        # layers
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor):
        mu = x.mean(-1, keepdim=True)
        sigma = x.std(-1, keepdim=True)
        return self.alpha * (x - mu) / (sigma + self.eps) + self.beta


class Attention(nn.Module):

    def forward(self,
                Q: Tensor,
                K: Tensor,
                V: Tensor,
                mask: Optional[Tensor] = None,
                dropout: Optional[nn.Module] = None
                ):
        """
            Q: (b x ? x L x dim_Q)
            K: (b x ? x L x dim_K)
            V: (b x ? x L x dim_V)
            ?: 1 (squeezed) or h (multi-head)

            mask: (b x ? x L x L)
            dropout: nn.Module

            assuming dim_Q = dim_K
        """

        dim_Q = Q.size(-1)

        # A: (b x ? x L x L)
        A = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(dim_Q)

        # apply mask (the logit value of a padding token should be minus infinity)
        if mask is not None:
            A = A.masked_fill(mask == 1, -1e9)  # tip: `mask is False` does not invoke broadcasting

        # getting normalized(probability) weights through softmax (when padding token, it'll be 0)
        # P: (b x ? x L x L)
        P = softmax(A, dim=-1)

        # apply dropout (with given dropout)
        if dropout is not None:
            P = dropout(P)

        # (b x ? x L x L) @ (b x ? x L x dim_V) -> (b x ? x L x dim_V)
        x = torch.matmul(P, V)

        return x, P


class MultiHeadedAttention(nn.Module):

    def __init__(self,
                 num_heads: int,
                 dim_model: int,
                 dim_temp: int = 0,
                 dropout_prob: float = 0.1
                 ):
        """
            dim_V should be equal to dim_model / num_heads

            we assume dim_Q = dim_K = dim_V
        """

        super().__init__()
        assert dim_model % num_heads == 0

        # params
        self.dim_model = dim_model
        self.dim_temp = dim_temp
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

        # splitted dim_V
        self.dim_V = dim_model // num_heads

        # layers
        self.W_Q = nn.Linear(dim_model, dim_model)
        self.W_K = nn.Linear(dim_model, dim_model)
        self.W_V = nn.Linear(dim_model, dim_model)
        self.W_M = nn.Linear(dim_model, dim_model)
        self.W_T = nn.Linear(dim_temp, dim_temp)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self,
                Q: Tensor,
                K: Tensor,
                V: Tensor,
                T: Optional[Tensor] = None,
                mask: Optional[Tensor] = None
                ):

        b = Q.size(0)

        # 1) Do all the linear projections in a batch from dim_model, then split into (num_heads x dim_V)
        # [process]
        # (1) linear(W): (b x L x dim_model) -> (b x L x dim_model)
        # (2) view: (b x L x dim_model) -> (b x L x num_heads x dim_V)
        # (3) transpose: (b x L x num_heads x dim_V) -> (b x num_heads x L x dim_V)
        Q = self.W_Q(Q).view(b, -1, self.num_heads, self.dim_V).transpose(1, 2)
        K = self.W_K(K).view(b, -1, self.num_heads, self.dim_V).transpose(1, 2)
        V = self.W_V(V).view(b, -1, self.num_heads, self.dim_V).transpose(1, 2)

        # only if temporal attention
        if self.dim_temp:
            # [process]
            # (1) linear(W): (b x L x dim_temp) -> (b x L x dim_temp)
            # (2) unsqueeze: (b x L x dim_temp) -> (b x 1 x L x dim_temp)
            # (2) repeat: (b x 1 x L x dim_temp) -> (b x num_heads x L x dim_temp)
            T = self.W_T(T).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            Q = torch.cat((Q, T), -1)  # type: ignore
            K = torch.cat((K, T), -1)  # type: ignore

        # 2) Apply attention to the projected vectors in the batch
        # note that attenion only cares about the last two dimensions
        # X: (b x num_heads x L x dim_V)
        X, _ = self.attention(Q, K, V, mask=mask, dropout=self.dropout)

        # 3) "concat" those heads using view
        # [process]
        # (1) transpose: (b x num_heads x L x dim_V) -> (b x L x num_heads x dim_V)
        # (2) contiguous: reorder memory inside GPU (no dimension change)
        # (3) view: (b x L x num_heads x dim_V) -> (b x L x dim_model)
        X = X.transpose(1, 2).contiguous().view(b, -1, self.dim_model)

        # 4) apply the final linear
        # X: (b x L x dim_model)
        X = self.W_M(X)

        return X


class SublayerConnection(nn.Module):

    def __init__(self, dim: int = 256, dropout_prob: float = 0.1):
        super().__init__()

        # params
        self.dim = dim
        self.dropout_prob = dropout_prob

        # layers
        self.layernorm = LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self,
                sublayer: nn.Module,
                num_args: int,
                x: Tensor,
                t: Optional[Tensor] = None
                ):
        r = self.layernorm(x)
        if num_args == 1:
            r = sublayer(r)
        elif num_args == 2:
            r = sublayer(r, t)
        r = self.dropout(r)
        return x + r


class PositionWiseFeedForward(nn.Module):

    def __init__(self,
                 dim_model: int = 256,
                 dim_ff: int = 1024,
                 dropout_prob: float = 0.1
                 ):
        super().__init__()

        # params
        self.dim_model = dim_model
        self.dim_ff = dim_ff
        self.dropout_prob = dropout_prob

        # layers
        self.W_in = nn.Linear(dim_model, dim_ff)
        self.W_out = nn.Linear(dim_ff, dim_model)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.gelu = GELU()

    def forward(self, x: Tensor):
        x = self.W_in(x)  # (b x dim_model) -> (b x dim_ff)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.W_out(x)  # (b x dim_ff) -> (b x dim_model)
        return x


class Transformer(nn.Module):

    def __init__(self,
                 dim_model: int = 256,
                 dim_temp: int = 0,
                 num_heads: int = 4,
                 dim_ff: int = 1024,
                 dropout_prob: float = 0.1
                 ):
        super().__init__()

        # params
        self.dim_model = dim_model
        self.dim_temp = dim_temp
        self.num_heads = num_heads
        self.dim_ff = dim_ff
        self.dropout_prob = dropout_prob

        # layers
        self.attention = MultiHeadedAttention(num_heads=num_heads, dim_model=dim_model, dim_temp=dim_temp, dropout_prob=dropout_prob)
        self.attention_sublayer = SublayerConnection(dim=dim_model, dropout_prob=dropout_prob)
        self.pwff = PositionWiseFeedForward(dim_model=dim_model, dim_ff=dim_ff, dropout_prob=dropout_prob)
        self.pwff_sublayer = SublayerConnection(dim=dim_model, dropout_prob=dropout_prob)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x: Tensor, t: Optional[Tensor] = None, mask: Optional[Tensor] = None):

        # we need dynamic mask for the attention forward (sublayer module also has parameters, namely layernorm)
        # x: (b x L x dim_model)
        # mask: (b x L x L), set False to ignore that point
        x = self.attention_sublayer(lambda z, w: self.attention(z, z, z, w, mask=mask), 2, x, t)
        x = self.pwff_sublayer(self.pwff, 1, x)
        x = self.dropout(x)

        return x
