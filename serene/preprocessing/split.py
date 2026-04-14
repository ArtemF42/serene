import logging
from datetime import datetime

import polars as pl


def apply_global_time_splitter(
    events: pl.DataFrame,
    time_threshold: int | datetime | str | float,
    time_format: str | None = None,
    keep_past: bool = True,
    user_key: str = "user_id",
    time_key: str = "timestamp",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if isinstance(time_threshold, str):
        if time_format is None:
            raise ValueError("string `time_threshold` requires `time_format`")

        time_threshold = datetime.strptime(time_threshold, time_format)
    elif isinstance(time_threshold, float):
        if not 0 <= time_threshold <= 1:
            raise ValueError("float `time_threshold` must be between 0.0 and 1.0")

        time_threshold = events[time_key].quantile(time_threshold)

    events = events.with_columns((pl.col(time_key) <= time_threshold).alias("__cond__"))

    return (
        events.filter(pl.col("__cond__")).drop("__cond__"),
        events.filter(
            pl.col("__cond__").not_().any().over(user_key)
            if keep_past else ~pl.col("__cond__")
        ).drop("__cond__"),
    )  # fmt: skip


def apply_random_user_splitter(
    events: pl.DataFrame,
    frac: float = 0.9,
    seed: int = 42,
    user_key: str = "user_id",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    events = events.with_columns(
        pl.col(user_key)
        .is_in(
            events[user_key]
            .unique(maintain_order=True)
            .sample(fraction=frac, seed=seed)
            .implode()
        )
        .alias("__cond__")
    )  # fmt: skip

    return (
        events.filter(pl.col("__cond__")).drop("__cond__"),
        events.filter(~pl.col("__cond__")).drop("__cond__"),
    )


def apply_last_n_splitter(
    events: pl.DataFrame,
    n: int = 1,
    user_key: str = "user_id",
    time_key: str = "timestamp",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if not n > 0:
        raise ValueError

    if events[user_key].value_counts(name="__count__")["__count__"].min() < n:
        logging.warning("some users have fewer than `n` interactions and will only appear in the target set")

    events = events.with_columns(
        pl.col(time_key)
        .rank(method="ordinal", descending=True)
        .over(user_key)
        .gt(n)  # rank > n
        .alias("__cond__")
    )  # fmt: skip

    return (
        events.filter(pl.col("__cond__")).drop("__cond__"),
        events.filter(~pl.col("__cond__")).drop("__cond__"),
    )
