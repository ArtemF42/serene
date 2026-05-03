from abc import ABC, abstractmethod
from typing import Iterable

import torch
import torch.nn as nn

from .functional import compute_hits, compute_hit_rate, compute_mrr, compute_ndcg


class TorchMetric(nn.Module, ABC):
    def __init__(self, top_k: int | Iterable[int]) -> None:
        super().__init__()

        top_k = (top_k,) if isinstance(top_k, int) else tuple(top_k)

        if not top_k:
            raise ValueError()

        if not all(k > 0 for k in top_k):
            raise ValueError("all `top_k` values must be positive.")

        max_k = max(top_k)

        self.top_k = top_k
        self.max_k = max_k

        self.register_buffer("k_idx", torch.tensor(top_k, dtype=torch.long) - 1)

    @abstractmethod
    def _forward(self, hits: torch.Tensor) -> torch.Tensor: ...

    def forward(
        self,
        recs: torch.Tensor | None = None,
        actuals: torch.Tensor | None = None,
        hits: torch.Tensor | None = None,
    ) -> dict[str, float]:
        if hits is None and not (recs is None or actuals is None):
            hits = compute_hits(recs, actuals)
        elif hits is not None and (recs is None and actuals is None):
            pass
        else:
            raise ValueError("either `hits` or both `recs` and `actuals` must be specified.")

        if hits.shape[1] < self.max_k:
            raise ValueError()

        values = self._forward(hits[:, : self.max_k])[:, self.k_idx].mean(dim=0)

        return {f"{self.name}@{k}": value.item() for k, value in zip(self.top_k, values)}

    @property
    def name(self) -> str:
        return self.__class__.__name__


class HitRate(TorchMetric):
    """Implementation of HitRate."""

    def _forward(self, hits: torch.Tensor) -> torch.Tensor:
        return compute_hit_rate(hits)


class MRR(TorchMetric):
    """Implementation of Mean Reciprocal Rank (MRR)."""

    def __init__(self, top_k: int | Iterable[int]) -> None:
        super().__init__(top_k)

        self.register_buffer("_reciprocal_ranks", 1 / (torch.arange(self.max_k) + 1))

    def _forward(self, hits: torch.Tensor) -> torch.Tensor:
        return compute_mrr(hits, self._reciprocal_ranks)


class NDCG(TorchMetric):
    """Implementation of Normalized Discounted Cumulative Gain (NDCG).

    Note:
        Formally computes DCG, but result is equivalent to NDCG since ground truth contains only one relevant item.
    """

    def __init__(self, top_k: int | Iterable[int]) -> None:
        super().__init__(top_k)

        self.register_buffer("_discount_factors", 1 / torch.log2(torch.arange(self.max_k) + 2))

    def _forward(self, hits: torch.Tensor) -> torch.Tensor:
        return compute_ndcg(hits, self._discount_factors)
