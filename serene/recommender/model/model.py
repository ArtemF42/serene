from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Model(nn.Module, ABC):
    def __init__(self, padding_idx: int) -> None:
        super().__init__()

        self.padding_idx = padding_idx

    @abstractmethod
    def embed(self, inputs: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def head(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def _forward(
        self,
        inputs_embeds: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor: ...

    def forward(
        self,
        inputs: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if inputs is not None:
            if inputs_embeds is not None:
                raise ValueError("exactly one of `inputs` or `inputs_embeds` must be specified.")

            inputs_embeds = self.embed(inputs)

        if inputs_embeds is None:
            raise ValueError("exactly one of `inputs` or `inputs_embeds` must be specified.")

        return self._forward(inputs_embeds, padding_mask, *args, **kwargs)


class HuggingFaceModel(Model):
    def _convert_padding_mask(self, padding_mask: torch.Tensor) -> torch.Tensor:
        return padding_mask.float()

    def forward(
        self,
        inputs: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if padding_mask is not None:
            padding_mask = self._convert_padding_mask(padding_mask)

        return super().forward(inputs, inputs_embeds, padding_mask, *args, **kwargs)
