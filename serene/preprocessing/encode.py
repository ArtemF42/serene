from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Self

import orjson
import polars as pl


class Encoder:
    def __init__(self, key: str, shift: int = 0, mapping: dict[Any, int] | None = None) -> None:
        if mapping is not None and not mapping:
            raise ValueError()

        self._key = key
        self._shift = shift
        self._mapping = mapping

    def fit(self, events: pl.DataFrame) -> Self:
        if events.is_empty():
            raise ValueError()

        self._mapping = self._build_mapping(self._unique_values(events), shift=self._shift)

        return self

    def fit_encode(self, events: pl.DataFrame) -> pl.DataFrame:
        return self.fit(events).encode(events)

    def update(self, events: pl.DataFrame) -> Self:
        self._ensure_fitted()

        unique_values = self._unique_values(events)
        unseen_values = unique_values.filter(~unique_values.is_in(self._mapping))

        self._mapping |= self._build_mapping(unseen_values, shift=max(self._mapping.values()) + 1)

        return self

    def update_encode(self, events: pl.DataFrame) -> pl.DataFrame:
        return self.update(events).encode(events)

    def encode(self, events: pl.DataFrame) -> pl.DataFrame:
        self._ensure_fitted()

        return events.with_columns(pl.col(self._key).replace_strict(self._mapping))

    def decode(self, events: pl.DataFrame) -> pl.DataFrame:
        self._ensure_fitted()

        inverse_mapping = {value: idx for idx, value in self._mapping.items()}

        return events.with_columns(pl.col(self._key).replace_strict(inverse_mapping))

    def save(self, filepath: str | os.PathLike) -> None:
        self._ensure_fitted()

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, mode="wb") as file:
            file.write(orjson.dumps(self.to_dict()))

    @classmethod
    def load(cls, filepath: str | os.PathLike) -> Encoder:
        if os.path.exists(filepath):
            with open(filepath, mode="rb") as file:
                return cls.from_dict(orjson.loads(file.read()))

        raise FileNotFoundError()

    @property
    def key(self) -> str:
        return self._key

    @property
    def shift(self) -> int:
        return self._shift

    @property
    def mapping(self) -> dict[Any, int] | None:
        if self._mapping is None:
            return None

        return self._mapping.copy()

    def to_dict(self) -> dict[str, Any]:
        self._ensure_fitted()

        return {"key": self._key, "shift": self._shift, "mapping": list(self._mapping.items())}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Encoder:
        return cls(key=data["key"], shift=data["shift"], mapping=dict(data["mapping"]))

    def _ensure_fitted(self) -> None:
        if self._mapping is None:
            raise RuntimeError()

    def _unique_values(self, events: pl.DataFrame) -> pl.Series:
        return events[self._key].unique().sort()

    def _build_mapping(self, unique_values: pl.Series, shift: int) -> dict[Any, int]:
        return {value: idx for idx, value in enumerate(unique_values, start=shift)}


class EncoderCollection:
    def __init__(self, encoders: Iterable[Encoder]) -> None:
        self._encoders = tuple(encoders)

    def fit(self, events: pl.DataFrame) -> Self:
        for encoder in self._encoders:
            encoder.fit(events)

        return self

    def fit_encode(self, events: pl.DataFrame) -> pl.DataFrame:
        return self.fit(events).encode(events)

    def update(self, events: pl.DataFrame) -> Self:
        for encoder in self._encoders:
            encoder.update(events)

        return self

    def update_encode(self, events: pl.DataFrame) -> pl.DataFrame:
        return self.update(events).encode(events)

    def encode(self, events: pl.DataFrame) -> pl.DataFrame:
        for encoder in self._encoders:
            events = encoder.encode(events)

        return events

    def decode(self, events: pl.DataFrame) -> pl.DataFrame:
        for encoder in self._encoders:
            events = encoder.decode(events)

        return events

    def save(self, filepath: str | os.PathLike) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, mode="wb") as file:
            file.write(orjson.dumps(self.to_dict()))

    @classmethod
    def load(cls, filepath: str | os.PathLike) -> EncoderCollection:
        if os.path.exists(filepath):
            with open(filepath, mode="rb") as file:
                return cls.from_dict(orjson.loads(file.read()))

        raise FileNotFoundError()

    @property
    def encoders(self) -> tuple[Encoder]:
        return self._encoders

    def to_dict(self) -> dict[str, Any]:
        return {"encoders": [encoder.to_dict() for encoder in self._encoders]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EncoderCollection:
        return cls(map(Encoder.from_dict, data["encoders"]))
