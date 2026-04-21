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
    """Split interactions into past and future events by a global time threshold.

    Args:
        events (pl.DataFrame): interactions to split.
        time_threshold (int | datetime | str | float): global time threshold used to split events. Used directly
            if `int` or `datetime` is provided. If `str`, it is parsed into a datetime value using `time_format`.
            If `float`, it must be between 0.0 and 1.0 and is interpreted as a quantile of the `time_key` column.
        time_format (str | None, optional): format used to parse string `time_threshold`. Defaults to None.
        keep_past (bool, optional): whether to include user histories from the past split in the future split. Defaults to True.
        user_key (str, optional): name of the user column in `events`. Defaults to "user_id".
        time_key (str, optional): name of the time column in `events`. Defaults to "timestamp".

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: past and future splits.
    """
    if isinstance(time_threshold, str):
        if time_format is None:
            raise ValueError("string `time_threshold` requires `time_format`.")

        time_threshold = datetime.strptime(time_threshold, time_format)
    elif isinstance(time_threshold, float):
        if not 0 <= time_threshold <= 1:
            raise ValueError("float `time_threshold` must be between 0.0 and 1.0.")

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
    num_users: int | None = None,
    fraction: float | None = None,
    seed: int = 42,
    user_key: str = "user_id",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split interactions by randomly sampling users.

    Note:
        Exactly one of `num_users` or `fraction` must be provided.

    Args:
        events (pl.DataFrame): interactions to split.
        num_users (int | None, optional): number of unique users to sample. Defaults to None.
        fraction (float | None, optional): fraction of unique users to sample. Defaults to None.
        seed (int, optional): random seed used for sampling. Defaults to 42.
        user_key (str, optional): name of the user column in `events`. Defaults to "user_id".

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: sampled and remaining user splits.
    """
    if (num_users is None) == (fraction is None):
        raise ValueError("exactly one of `num_users` or `fraction` must be provided.")

    events = events.with_columns(
        pl.col(user_key)
        .is_in(
            events[user_key]
            .unique(maintain_order=True)
            .sample(n=num_users, fraction=fraction, seed=seed)
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
    """Split interactions by holding out the last `n` interactions for each user.

    Args:
        events (pl.DataFrame): interactions to split.
        n (int, optional): number of last interactions to include in the future split for each user. Defaults to 1.
        user_key (str, optional): name of the user column in `events`. Defaults to "user_id".
        time_key (str, optional): name of the time column in `events`. Defaults to "timestamp".

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: past and future splits.
    """
    if not n > 0:
        raise ValueError("`n` must be positive.")

    if events[user_key].value_counts(name="__count__")["__count__"].min() < n:
        logging.warning("some users have fewer than `n` interactions and will only appear in the future split.")

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
