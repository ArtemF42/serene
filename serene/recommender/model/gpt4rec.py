from typing import Literal

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

from .model import HuggingFaceModel


class GPT4RecModel(HuggingFaceModel):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout_p: float,
        max_length: int,
        intermediate_dim: int | None = None,
        activation: Literal["relu", "gelu", "gelu_new", "silu", "tanh"] = "gelu_new",
        padding_idx: int = 0,
    ) -> None:
        super().__init__()

        self._gpt2_model = GPT2Model(
            GPT2Config(
                vocab_size=num_items,
                n_positions=max_length,
                n_embd=embedding_dim,
                n_layer=num_blocks,
                n_head=num_heads,
                n_inner=intermediate_dim,
                activation_function=activation,
                resid_pdrop=dropout_p,
                embd_pdrop=dropout_p,
                attn_pdrop=dropout_p,
                pad_token_id=padding_idx,
            )
        )

    def _forward(
        self,
        inputs_embeddings: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return self._gpt2_model(
            *args,
            attention_mask=padding_mask,
            inputs_embeds=inputs_embeddings,
            **kwargs,
        ).last_hidden_state

    @property
    def item_embedding(self) -> nn.Embedding:
        return self._gpt2_model.wte
