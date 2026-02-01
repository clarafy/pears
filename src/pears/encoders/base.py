from collections.abc import Sequence


class SequentialEncoder:
    """
    Concrete encoder that maps N items to a contiguous range of integers
    starting at 0.
    """

    def __init__(self, items: Sequence[str]):
        self._items = sorted(set(items))
        self._to_idx = {label: idx for idx, label in enumerate(self._items)}
        self._to_label = dict(enumerate(self._items))

    def encode(self, label: str) -> int:
        return self._to_idx[label]

    def decode(self, index: int) -> str:
        return self._to_label[index]

    def __len__(self) -> int:
        return len(self._items)

    def is_valid_label(self, label: str) -> bool:
        return label in self._to_idx

    def is_valid_idx(self, idx: int) -> bool:
        return idx in self._to_label

    def items(self) -> Sequence[str]:
        return self._items
