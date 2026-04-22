from typing import Literal

import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    """Optimized implementation of SwiGLU activation."""

    def __init__(self, embedding_dim: int, intermediate_dim: int) -> None:
        super().__init__()

        self.wv_proj = nn.Linear(embedding_dim, intermediate_dim * 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_w, x_v = self.wv_proj(x).chunk(2, dim=-1)
        return nn.functional.silu(x_w) * x_v


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        intermediate_dim: int,
        dropout_p: float,
        activation: Literal["relu", "gelu", "silu", "swiglu"] = "swiglu",
    ) -> None:
        super().__init__()

        if activation == "swiglu":
            intermediate_dim = intermediate_dim * 2 // 3
            activation_block = SwiGLU(embedding_dim, intermediate_dim)
        else:
            activation_block = nn.Sequential(
                nn.Linear(embedding_dim, intermediate_dim, bias=False),
                {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[activation](),
            )

        self.layers = nn.Sequential(
            activation_block,
            nn.Dropout(dropout_p),
            nn.Linear(intermediate_dim, embedding_dim, bias=False),
            nn.Dropout(dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
