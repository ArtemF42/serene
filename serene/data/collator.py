from functools import partial
from typing import Any, Literal

from torch.nn.utils.rnn import pad_sequence


class SimpleCollator:
    def __init__(
        self,
        padding_idx: int = 0,
        padding_side: Literal["left", "right"] = "left",
        return_padding_mask: bool = False,
    ) -> None:
        self.padding_idx = padding_idx
        self.padding_side = padding_side
        self.return_padding_mask = return_padding_mask

        self._pad_sequence = partial(pad_sequence, batch_first=True, padding_side=padding_side)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        batch = {key: [example[key] for example in batch] for key in batch[0].keys()}

        batch["inputs"] = self._pad_sequence(batch["inputs"], padding_value=self.padding_idx)

        # NOTE: features are padded with the default `padding_value=0`,
        # which is cast to `0` (int), `0.0` (float), or `False` (bool) depending on tensor dtype.
        for key in batch.keys():
            if key.startswith("feature"):
                batch[key] = self._pad_sequence(batch[key])

        if self.return_padding_mask:
            batch["padding_mask"] = batch["inputs"] != self.padding_idx

        return batch
