from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
import polars.selectors as cs
import torch
from torch.utils.data import Dataset


class SequentialDataset(Dataset):
    def __init__(
        self,
        events: pl.DataFrame,
        max_length: int,
        min_length: int,
        user_key: str = "user_id",
        item_key: str = "item_id",
        time_key: str = "timestamp",
        feature_keys: Sequence[str] | None = None,
        random_slice: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        if not max_length > 0:
            raise ValueError("`max_length` must be > 0.")

        if not 0 <= min_length <= max_length:
            raise ValueError("`min_length` must be <= `max_length` and >= 0.")

        self.max_length = max_length
        self.min_length = min_length

        schema = events.schema

        if not schema[item_key].is_integer():
            raise ValueError(f"{item_key!r} must be an integer column, found {schema[item_key]}.")

        feature_keys = () if feature_keys is None else tuple(feature_keys)

        for key in feature_keys:
            dtype = schema[key]

            if not (dtype.is_integer() or dtype.is_float() or dtype == pl.Boolean):
                raise ValueError(f"only integer, float, and boolean features are supported, found {dtype} for feature {key!r}.")  # fmt: skip

        self.user_key = user_key
        self.item_key = item_key
        self.time_key = time_key
        self.feature_keys = feature_keys

        self.random_slice = random_slice
        self.seed = seed

        events = (
            events.select(user_key, item_key, time_key, *feature_keys)
            .with_columns(
                cs.integer().cast(pl.Int64),
                cs.float().cast(pl.Float32),
            )
            .filter(pl.len().over(user_key) >= min_length)
            .sort(user_key, time_key)
        )
        counts = events.group_by(user_key, maintain_order=True).len(name="__count__")

        self._item_ids: np.ndarray = events[item_key].to_numpy()
        self._features: dict[str, np.ndarray] = {key: events[key].to_numpy() for key in feature_keys}

        self._users: list[Any] = counts[user_key].to_list()

        self._offsets: np.ndarray = np.zeros(len(counts) + 1, dtype=np.int64)
        self._offsets[1:] = np.cumsum(counts["__count__"])

        if random_slice:
            self._rng = np.random.default_rng(seed=seed)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if not 0 <= idx < len(self):
            raise IndexError(f"index {idx} is out of range for dataset of size {len(self)}.")

        start, end = self._offsets[idx], self._offsets[idx + 1]
        length = end - start

        if length <= self.max_length:
            _slice = slice(None)
        elif self.random_slice:
            shift = self._rng.integers(0, length - self.max_length + 1)
            _slice = slice(shift, shift + self.max_length)
        else:
            _slice = slice(-self.max_length, None)

        return {
            "user_id": self._users[idx],
            "history": self._item_ids[start:end],
            "inputs": torch.from_numpy(self._item_ids[start:end][_slice].copy()),
        } | {
            f"feature.{key}": torch.from_numpy(value[start:end][_slice].copy())
            for key, value in self._features.items()
        }  # fmt: skip

    def __len__(self) -> int:
        return len(self._users)
