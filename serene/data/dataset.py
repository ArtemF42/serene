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
            raise ValueError

        if not 0 <= min_length <= max_length:
            raise ValueError

        self.max_length = max_length
        self.min_length = min_length

        if not events.schema[item_key].is_integer():
            raise ValueError

        feature_keys = () if feature_keys is None else tuple(feature_keys)

        for key in feature_keys:
            dtype = events.schema[key]

            if not (dtype.is_integer() or dtype.is_float() or dtype == pl.Boolean):
                raise ValueError

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

        self._events: dict[str, np.ndarray] = {
            key: events[key].to_numpy()
            for key in (item_key, *feature_keys)
        }  # fmt: skip
        self._users: list[Any] = counts[user_key].to_list()

        self._offsets: np.ndarray = np.zeros(len(counts) + 1, dtype=np.int64)
        self._offsets[1:] = np.cumsum(counts["__count__"])

        if random_slice:
            self._rng = np.random.default_rng(seed=seed)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if not 0 <= idx < len(self):
            raise IndexError

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
            self.user_key: self._users[idx],
            self.item_key: self._events[self.item_key][start:end],
        } | {
            f"inputs.{key}": torch.from_numpy(value[start:end][_slice].copy())
            for key, value in self._events.items()
        }  # fmt: skip

    def __len__(self) -> int:
        return len(self._users)
