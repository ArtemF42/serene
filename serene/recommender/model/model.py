from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Model(nn.Module, ABC):
    @property
    @abstractmethod
    def item_embedding(self) -> nn.Embedding: ...

    @abstractmethod
    def _forward(
        self,
        inputs_embeddings: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor: ...

    def forward(
        self,
        inputs: torch.Tensor | None = None,
        inputs_embeddings: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if inputs is not None:
            if inputs_embeddings is not None:
                raise ValueError("exactly one of `inputs` or `inputs_embeddings` must be specified.")

            inputs_embeddings = self.item_embedding(inputs)

        if inputs_embeddings is None:
            raise ValueError("exactly one of `inputs` or `inputs_embeddings` must be specified.")

        return self._forward(inputs_embeddings, padding_mask, *args, **kwargs)


class HuggingFaceModel(Model):
    def _convert_padding_mask(self, padding_mask: torch.Tensor) -> torch.Tensor:
        return padding_mask.float()

    def forward(
        self,
        inputs: torch.Tensor | None = None,
        inputs_embeddings: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if padding_mask is not None:
            padding_mask = self._convert_padding_mask(padding_mask)

        return super().forward(inputs, inputs_embeddings, padding_mask, *args, **kwargs)
