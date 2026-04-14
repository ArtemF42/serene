import polars as pl


def apply_min_count_filter(events: pl.DataFrame, min_count: int, key: str) -> pl.DataFrame:
    return events.filter(pl.len().over(key) >= min_count)


def apply_min_rating_filter(events: pl.DataFrame, min_rating: int | float, rating_key: str = "rating") -> pl.DataFrame:
    return events.filter(pl.col(rating_key) >= min_rating)
