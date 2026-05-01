import logging

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel

from .model import HuggingFaceModel


class Llama4Rec(HuggingFaceModel):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout_p: float,
        max_length: int,
        hidden_dim: int | None = None,
        activation: str = "silu",
        padding_idx: int = 0,
    ) -> None:
        super().__init__(padding_idx=padding_idx)

        if hidden_dim is None:
            hidden_dim = embedding_dim * 4
            logging.info(f"`hidden_dim` was not provided, set to {hidden_dim}.")

        self._llama_model = LlamaModel(
            LlamaConfig(
                vocab_size=num_items,
                hidden_size=embedding_dim,
                intermediate_size=hidden_dim,
                num_hidden_layers=num_blocks,
                num_attention_heads=num_heads,
                hidden_act=activation,
                max_position_embeddings=max_length,
                pad_token_id=padding_idx,
                attention_dropout=dropout_p,
            )
        )

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._llama_model.embed_tokens(inputs)

    def head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(hidden_states, self._llama_model.embed_tokens.weight)

    def _forward(
        self,
        inputs_embeds: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return self._llama_model(
            *args,
            attention_mask=padding_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        ).last_hidden_state
