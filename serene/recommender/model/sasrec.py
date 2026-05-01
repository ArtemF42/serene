import logging
from typing import Literal

import torch
import torch.nn as nn

from .attention import SelfAttention
from .feed_forward_network import FeedForwardNetwork
from .model import Model


class SASRecBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout_p: float,
        hidden_dim: int,
        activation: Literal["relu", "gelu", "silu", "swiglu"] = "swiglu",
    ) -> None:
        super().__init__()

        self.pre_attn_rms_norm = nn.RMSNorm(embedding_dim)
        self.attn = SelfAttention(embedding_dim, num_heads, dropout_p, is_causal=True)

        self.pre_ffn_rms_norm = nn.RMSNorm(embedding_dim)
        self.ffn = FeedForwardNetwork(embedding_dim, dropout_p, hidden_dim, activation)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.pre_attn_rms_norm(x), padding_mask)
        x = x + self.ffn(self.pre_ffn_rms_norm(x))

        return x


class SASRecModel(Model):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout_p: float,
        max_length: int,
        hidden_dim: int | None = None,
        activation: Literal["relu", "gelu", "silu", "swiglu"] = "swiglu",
        padding_idx: int = 0,
    ) -> None:
        super().__init__(padding_idx=padding_idx)

        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.max_length = max_length

        if hidden_dim is None:
            hidden_dim = embedding_dim * 4
            logging.info(f"`hidden_dim` was not provided, set to {hidden_dim}.")

        self.hidden_dim = hidden_dim
        self.activation = activation

        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=padding_idx)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout_p)

        self.blocks = nn.ModuleList()

        for _ in range(num_blocks):
            block = SASRecBlock(
                embedding_dim,
                num_heads,
                dropout_p,
                hidden_dim,
                activation,
            )
            self.blocks.append(block)

        self.out_rms_norm = nn.RMSNorm(embedding_dim)

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.item_embedding(inputs)

    def head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(hidden_states, self.item_embedding.weight)

    def _forward(
        self,
        inputs_embeddings: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if inputs_embeddings.shape[1] > self.max_length:
            raise ValueError("input length exceeds `max_length`.")

        x = inputs_embeddings  # convenience alias

        x = x * self.embedding_dim**0.5
        x = x + self.position_embedding(torch.arange(x.shape[1], dtype=torch.long, device=x.device))
        x = self.embedding_dropout(x)

        for block in self.blocks:
            x = block(x, padding_mask)

        return self.out_rms_norm(x)
