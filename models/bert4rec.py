import torch.nn as nn

from typing import Optional

from torch import (
    Tensor,
    LongTensor,
)

from tools.utils import fix_random_seed

from .layers import (
    TemporalEmbedding,
    InputEmbedding,
    Transformer,
)


__all__ = (
    'BERT4Rec',
)


class BERT4Rec(nn.Module):

    def __init__(self,
                 num_items: int,
                 sequence_len: int,
                 max_num_segments: int,
                 use_session_token: bool = False,
                 num_layers: int = 2,
                 hidden_dim: int = 256,
                 temporal_dim: int = 0,
                 num_heads: int = 4,
                 dropout_prob: float = 0.1,
                 random_seed: Optional[int] = None
                 ):
        """
            Note that item index starts from 1.
            Use 0 label (ignore index in CE) to avoid learning unmasked(context, known) items.
        """
        super().__init__()

        # params
        self.num_items = num_items
        self.sequence_len = sequence_len
        self.max_num_segments = max_num_segments
        self.use_session_token = use_session_token
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.temporal_dim = temporal_dim
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        self.random_seed = random_seed

        # set seed
        if random_seed is not None:
            fix_random_seed(random_seed)

        # 0: padding token
        # 1 ~ V: item tokens
        # V + 1: mask token
        # V + 2: session token
        if use_session_token:
            vocab_size = num_items + 3
        else:
            vocab_size = num_items + 2

        # consider padding segment
        num_segments = max_num_segments + 1

        # layers
        self.input_embedding = InputEmbedding(
            vocab_size=vocab_size,
            sequence_len=sequence_len,
            num_segments=num_segments,
            embedding_dim=hidden_dim,
            dropout_prob=dropout_prob
        )
        self.temporal_embedding = TemporalEmbedding(
            embedding_dim=temporal_dim
        )
        self.transformers = nn.ModuleList([
            Transformer(
                dim_model=hidden_dim,
                dim_temp=temporal_dim,
                num_heads=num_heads,
                dim_ff=hidden_dim * 4,
                dropout_prob=dropout_prob
            ) for _ in range(num_layers)
        ])
        self.clf = nn.Linear(hidden_dim, num_items + 1)

    def forward(self,
                tokens: LongTensor,
                segments: Optional[LongTensor] = None,
                stamps: Optional[Tensor] = None
                ):

        L = tokens.size(1)

        # mask for whether padding token or not in attention matrix (True if padding token)
        # [process] (b x L) -> (b x 1 x L) -> (b x L x L) -> (b x 1 x L x L)
        token_mask = ~(tokens > 0).unsqueeze(1).repeat(1, L, 1).unsqueeze(1)

        # get embedding
        # [process] (b x L) -> (b x L x d)
        x = self.input_embedding(tokens, segments)
        if stamps is not None:
            t = self.temporal_embedding(stamps)
        else:
            t = None

        # apply multi-layered transformers
        # [process] (b x L x d) -> ... -> (b x L x d)
        for transformer in self.transformers:
            x = transformer(x, t, token_mask)

        # classifier
        # [process] (b x L x d) -> (b x L x (V + 1))
        logits = self.clf(x)

        return logits
