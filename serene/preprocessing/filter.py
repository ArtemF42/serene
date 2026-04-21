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
