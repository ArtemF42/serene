import logging
from typing import Self

import polars as pl
import torch
import torch.nn as nn


class _AliasTable(nn.Module):
    def __init__(self, weights: torch.Tensor) -> None:
        super().__init__()

        assert weights.ndim == 1
        assert weights.numel() > 0
        assert (weights >= 0).all()
        assert weights.sum() > 0

        device = weights.device

        weights = weights.to(torch.float64)
        weights = weights / weights.sum()

        n = weights.numel()
        average = 1 / n

        probs = torch.zeros(n, dtype=torch.float64, device=device)
        alias = torch.zeros(n, dtype=torch.long, device=device)

        small: list[int] = []
        large: list[int] = []

        for i, w in enumerate(weights):
            (small if w < average else large).append(i)

        while small and large:
            small_i, large_i = small.pop(), large.pop()

            probs[small_i] = weights[small_i] * n
            alias[small_i] = large_i

            weights[large_i] += weights[small_i] - average

            (small if weights[large_i] < average else large).append(large_i)

        while small:
            probs[small.pop()] = 1

        while large:
            probs[large.pop()] = 1

        self.register_buffer("_probs", probs)
        self.register_buffer("_alias", alias)

        self._n = n

    def forward(self, size: int = 1) -> torch.Tensor:
        i = torch.randint(0, self._n, size=(size,), dtype=torch.long, device=self.device)
        return torch.where(
            torch.rand(size=(size,), dtype=torch.float64, device=self.device) < self._probs[i],
            i, self._alias[i],
        )  # fmt: skip

    @property
    def device(self) -> torch.device:
        return self._probs.device


class RandomSampler(nn.Module):
    def __init__(
        self,
        items: torch.Tensor,
        freqs: torch.Tensor,
        num_samples: int = 1,
        alpha: float | None = None,
    ) -> None:
        super().__init__()

        assert items.ndim == freqs.ndim == 1
        assert items.numel() == freqs.numel()

        if alpha is not None:
            if not 0 <= alpha <= 1:
                logging.warning("`alpha` should be in range [0, 1].")

            freqs = freqs**alpha

        self.register_buffer("items", items)
        self.register_buffer("freqs", freqs)

        assert num_samples > 0

        self.num_samples = num_samples
        self.alpha = alpha

        self._alias_table = _AliasTable(freqs)

    def forward(self) -> torch.Tensor:
        return self.items[self._alias_table(self.num_samples)]

    @classmethod
    def from_events(
        cls,
        events: pl.DataFrame,
        item_key: str = "item_id",
        num_samples: int = 1,
        alpha: float | None = None,
    ) -> Self:
        items, freqs = events[item_key].value_counts().to_torch().T
        return cls(items=items, freqs=freqs, num_samples=num_samples, alpha=alpha)
