from typing import Literal, OrderedDict

import torch
from torch import nn


class EmbeddingPositionalEncoding(nn.Module):
    def __init__(
        self,
        transform: Literal["add", "concat", "dry"],
        num_tokens: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.transform = transform
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=num_tokens, embedding_dim=embedding_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : `(seq_len, batch_size, D)` Tensor

        Returns
        -------
        `(seq_len, batch_size, D or D + embedding_dim)` Tensor
        """
        positions = torch.arange(0, x.size(0), device=x.device).reshape(1, -1)
        pos_emb = self.embedding(positions).permute(1, 0, 2)
        if self.transform == "add":
            return x + pos_emb
        elif self.transform == "concat":
            assert False  # TODO
            return torch.concatenate([x, pos_emb], dim=2)
        return pos_emb


class TransformerLayer(nn.Module):
    def __init__(
        self, num_heads: int, n_features: int, dropout: float, layer_norm: bool = True
    ):
        # NOTE: got rid of layer norms
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=n_features, num_heads=num_heads, dropout=dropout
        )  # TODO: zrobić lepiej xd
        self.sa_dropout = nn.Dropout(p=dropout)
        self.feed_forward = nn.Sequential(
            OrderedDict(
                [
                    ("fc0", nn.Linear(n_features, n_features)),
                    ("softplus", nn.Softplus()),
                    ("dropout0", nn.Dropout(p=dropout)),
                    ("fc1", nn.Linear(n_features, n_features)),
                    ("dropout1", nn.Dropout(p=dropout)),
                ]
            )
        )
        self.layer_norm = (
            nn.LayerNorm(n_features) if layer_norm else None
        )  # ! TODO: eps -> 0.01

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """(seq_len, batch_size, n_features) -> (seq_len, batch_size, n_features)"""
        x = src + self.sa_dropout(self.self_attn(src, src, src)[0])
        x = x + self.feed_forward(x)
        if self.layer_norm:
            x = self.layer_norm(x)
        return x