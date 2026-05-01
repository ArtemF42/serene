from typing import Any

import torch
import torch.nn as nn
from lightning import LightningModule

from .model.model import Model
from ..data.sampler import RandomSampler


class SequentialRecommender(LightningModule):
    def __init__(
        self,
        model: Model,
        optimizer_params: dict[str, Any],
        scheduler_params: dict[str, Any],
        sampler: RandomSampler | None = None,
    ) -> None:
        super().__init__()

        self.model = model
        self.sampler = sampler

        # optimization params
        optimizer_params = optimizer_params.copy()

        self.learning_rate = optimizer_params.pop("lr", 1e-3)

        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params

        self.loss_fn = nn.CrossEntropyLoss()

    def training_step(self, batch: dict[str, Any], batch_idx: int | None = None) -> torch.Tensor:
        inputs: torch.Tensor = batch["inputs"]

        # shift sequences for causal modeling
        labels = inputs[:, 1:].clone()
        inputs = inputs[:, :-1]

        # mask padding
        labels = labels.masked_fill(labels == self.model.padding_idx, -100)

        # forward pass
        padding_mask = inputs != self.model.padding_idx
        hidden_states = self.model(inputs=inputs, padding_mask=padding_mask)

        if self.sampler is None:
            logits = self.model.head(hidden_states)
        else:
            # compute logits for positive samples
            pos_labels = torch.where(labels != -100, labels, self.model.padding_idx)
            pos_embeds = self.model.embed(pos_labels)  # (B, L, E)
            pos_logits = (hidden_states * pos_embeds).sum(dim=-1)  # (B, L)

            # compute logits for negative samples
            neg_labels = self.sampler()
            neg_embeds = self.model.embed(neg_labels)  # (N, E)
            neg_logits = hidden_states @ neg_embeds.T  # (B, L, N)

            logits = torch.cat((pos_logits.unsqueeze(-1), neg_logits), dim=-1)  # (B, L, 1 + N)
            labels = torch.where(labels == -100, labels, 0)

        loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
        self.log("train-loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            **self.scheduler_params,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
