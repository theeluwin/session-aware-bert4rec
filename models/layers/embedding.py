import torch
import torch.nn as nn
import numpy as np

from typing import Optional

from torch import (
    Tensor,
    LongTensor,
)


__all__ = (
    'TokenEmbedding',
    'PositionalEmbedding',
    'SegmentEmbedding',
    'TemporalEmbedding',
    'InputEmbedding',
)


class TokenEmbedding(nn.Embedding):

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 256
                 ):

        # params
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # (V x d)
        super().__init__(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)


class PositionalEmbedding(nn.Module):

    def __init__(self,
                 sequence_len: int,
                 embedding_dim: int = 256
                 ):
        super().__init__()

        # params
        self.sequence_len = sequence_len
        self.embedding_dim = embedding_dim

        # layers
        self.embedding = nn.Embedding(sequence_len, embedding_dim)

    def forward(self, tokens: LongTensor):

        b = tokens.size(0)

        # [process]
        # (1) unsqueeze: (L x d) -> (1 x L x d)
        # (2) repeat: (1 x L x d) -> (b x L x d)
        x = self.embedding.weight.unsqueeze(0).repeat(b, 1, 1)

        return x


class SegmentEmbedding(nn.Embedding):

    def __init__(self,
                 num_segments: int,
                 embedding_dim: int = 256
                 ):

        # params
        self.num_segments = num_segments
        self.embedding_dim = embedding_dim

        # (S x d)
        super().__init__(num_embeddings=num_segments, embedding_dim=embedding_dim, padding_idx=0)


class TemporalEmbedding(nn.Module):

    def __init__(self, embedding_dim: int = 32):
        super().__init__()

        # params
        self.embedding_dim = embedding_dim

        # init (see TGSRec)
        temporal_init = torch.from_numpy(1 / 10 ** np.linspace(0, 9, embedding_dim))

        # layers
        self.weight = nn.Parameter(temporal_init.float())
        self.bias = nn.Parameter(torch.zeros(embedding_dim).float())

    def forward(self, stamps: Tensor):

        # (d) -> (1 x 1 x d)
        weight = self.weight.view(1, 1, -1)
        bias = self.bias.view(1, 1, -1)

        # (b x L) -> (b x L x 1)
        stamps = stamps.unsqueeze(-1)

        # (b x L x 1) times (1 x 1 x d) -> (b x L x d)
        span = stamps * weight + bias
        harmonic = torch.cos(span)

        return harmonic


class InputEmbedding(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 sequence_len: int = 0,
                 num_segments: int = 0,
                 embedding_dim: int = 256,
                 dropout_prob: float = 0.1
                 ):
        super().__init__()

        # params
        self.vocab_size = vocab_size
        self.sequence_len = sequence_len
        self.num_segments = num_segments
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout_prob

        # layers
        self.token_embedding = TokenEmbedding(vocab_size=vocab_size, embedding_dim=embedding_dim)
        self.position_embedding = PositionalEmbedding(sequence_len=sequence_len, embedding_dim=embedding_dim)
        self.segment_embedding = SegmentEmbedding(num_segments=num_segments, embedding_dim=embedding_dim)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self,
                tokens: LongTensor,
                segments: Optional[LongTensor] = None
                ):

        # basic embedding
        x = self.token_embedding(tokens)

        # add positions
        if self.sequence_len:
            x = x + self.position_embedding(tokens)

        # add segments
        if self.num_segments and segments is not None:
            x = x + self.segment_embedding(segments)

        x = self.dropout(x)
        return x
