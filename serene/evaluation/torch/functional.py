import torch


def compute_hits(recs: torch.Tensor, actuals: torch.Tensor) -> torch.Tensor:
    assert recs.ndim == 2
    assert actuals.ndim == 1

    return recs.eq(actuals.unsqueeze(-1)).float()


def compute_ranks(hits: torch.Tensor) -> torch.Tensor:
    assert hits.ndim == 2

    return torch.arange(hits.shape[1], dtype=torch.long, device=hits.device).unsqueeze(0).expand_as(hits) + 1


def compute_hit_rate(hits: torch.Tensor) -> torch.Tensor:
    assert hits.ndim == 2

    return torch.cummax(hits, dim=1).values


def compute_mrr(hits: torch.Tensor, reciprocal_ranks: torch.Tensor | None = None) -> torch.Tensor:
    assert hits.ndim == 2

    if reciprocal_ranks is None:
        reciprocal_ranks = 1 / compute_ranks(hits)
    else:
        assert reciprocal_ranks.shape == hits.shape

    return torch.cummax(hits * reciprocal_ranks, dim=1).values


def compute_ndcg(hits: torch.Tensor, discount_factors: torch.Tensor | None = None) -> torch.Tensor:
    assert hits.ndim == 2

    if discount_factors is None:
        discount_factors = 1 / torch.log2(compute_ranks(hits) + 1)
    else:
        assert discount_factors.shape == hits.shape

    return torch.cumsum(hits * discount_factors, dim=1)
