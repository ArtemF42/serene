import logging
from collections.abc import Sequence
from typing import Literal

import polars as pl


def apply_min_count_filter(events: pl.DataFrame, min_count: int, key: str) -> pl.DataFrame:
    """Filter out interactions whose group has fewer than `min_count` occurrences.

    Args:
        events (pl.DataFrame): interactions to filter.
        min_count (int): minimum number of interactions required in each group.
        key (str): name of the column used to group interactions.

    Returns:
        pl.DataFrame: filtered interactions.
    """
    return events.filter(pl.len().over(key) >= min_count)


def apply_min_rating_filter(events: pl.DataFrame, min_rating: int | float, rating_key: str = "rating") -> pl.DataFrame:
    """Filter out interactions whose rating is below `min_rating`.

    Args:
        events (pl.DataFrame): interactions to filter.
        min_rating (int | float): minimum rating required to keep an interaction.
        rating_key (str, optional): name of the rating column in `events`. Defaults to "rating".

    Returns:
        pl.DataFrame: filtered interactions.
    """
    return events.filter(pl.col(rating_key) >= min_rating)


def apply_consecutive_duplicates_filter(
    events: pl.DataFrame,
    aggregation: Literal["first", "last"] | pl.Expr | Sequence[pl.Expr] = "first",
    user_key: str = "user_id",
    item_key: str = "item_id",
    time_key: str = "timestamp",
) -> pl.DataFrame:
    if isinstance(aggregation, str):
        aggregation = {"first": pl.all().first(), "last": pl.all().last()}[aggregation]

    return (
        events.sort(user_key, time_key)
        .with_columns(
            ((pl.col(item_key) != pl.col(item_key).shift()).fill_null(True).cum_sum())
            .over(user_key)
            .alias("__consecutive_series_idx__")
        )
        .group_by(user_key, "__consecutive_series_idx__", maintain_order=True)
        .agg(aggregation)
        .drop("__consecutive_series_idx__")
    )


def apply_n_core_filter(
    events: pl.DataFrame,
    min_count: int | None = None,
    user_min_count: int | None = None,
    item_min_count: int | None = None,
    user_key: str = "user_id",
    item_key: str = "item_id",
) -> pl.DataFrame:
    if min_count is None:
        if user_min_count is None or item_min_count is None:
            raise ValueError("if `min_count` is not specified, both `user_min_count` and `item_min_count` must be provided.")  # fmt: skip
    else:
        if user_min_count is not None or item_min_count is not None:
            logging.warning("`user_min_count` and `item_min_count` are overridden by `min_count`.")

        user_min_count = item_min_count = min_count

    height = -1

    while events.height != height:
        height = events.height

        events = apply_min_count_filter(events, user_min_count, user_key)
        events = apply_min_count_filter(events, item_min_count, item_key)

    return events
