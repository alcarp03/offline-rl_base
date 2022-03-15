from typing import (
    Callable,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeVar,
)
import numpy as np

T = TypeVar("T")


class FIFOQueue(Generic[T]):
    """Simple FIFO queue implementation.

    Random access of this queue object is O(1).

    """

    _maxlen: int
    _drop_callback: Optional[Callable[[T], None]]
    _buffer: List[Optional[T]]
    _cursor: int
    _size: int
    _index: int

    def __init__(
        self, maxlen: int, drop_callback: Optional[Callable[[T], None]] = None
    ):
        self._maxlen = maxlen
        self._drop_callback = drop_callback
        self._buffer = [None for _ in range(maxlen)]
        self._cursor = 0
        self._size = 0
        self._index = 0

    def append(self, item: T) -> None:
        # call drop callback if necessary
        cur_item = self._buffer[self._cursor]
        if cur_item and self._drop_callback:
            self._drop_callback(cur_item)

        self._buffer[self._cursor] = item

        # increment cursor
        self._cursor += 1
        if self._cursor == self._maxlen:
            self._cursor = 0
        self._size = min(self._size + 1, self._maxlen)

    def extend(self, items: Sequence[T]) -> None:
        for item in items:
            self.append(item)

    def __getitem__(self, index: int) -> T:
        assert index < self._size

        # handle negative indexing
        if index < 0:
            index = self._size + index

        item = self._buffer[index]
        assert item is not None
        return item

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[T]:
        self._index = 0
        return self

    def __next__(self) -> T:
        if self._index >= self._size:
            raise StopIteration
        item = self._buffer[self._index]
        assert item is not None
        self._index += 1
        return item


class SegmentTree(Generic[T]):

    _max_size: int
    _drop_callback: Optional[Callable[[T], None]]
    _buffer: List[Optional[T]]
    _cursor: int
    _size: int
    _index: int

    def __init__(
        self, max_size: int, drop_callback: Optional[Callable[[T], None]] = None
    ):
        self._maxlen = max_size
        self._drop_callback = drop_callback
        self._buffer = [None for _ in range(max_size)]
        self._cursor = 0
        self._size = 0
        self._index = 0

        self._transitions = np.zeros((max_size, 1))
        self.index = 0
        self.max_size = max_size
        self.full = False
        self.tree_start = 2 ** (max_size - 1).bit_length() - 1
        self.sum_tree = np.zeros((self.tree_start + self.max_size), dtype=np.float32)

        self.max = 1.0

    def __getitem__(self, index: int) -> T:
        assert index < self.sum_tree[0]

        # handle negative indexing
        if index < 0:
            index = self.max_size + index

        item = self.sum_tree[index]
        assert item is not None
        return item

    def __len__(self) -> int:
        return self.sum_tree[0]

    def __iter__(self) -> Iterator[T]:
        self._index = 0
        return self

    def __next__(self) -> T:
        if self._index >= self.sum_tree[0]:
            raise StopIteration
        item = self.sum_tree[self._index]
        assert item is not None
        self._index += 1
        return item

    def _update_nodes(self, indices):
        children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
        # [0,1,2,3] -> [1,3,5,7; 2,4,6,8]
        self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)

    def _propagate(self, indices):
        parents = (indices - 1) // 2
        unique_parents = np.unique(parents)
        self._update_nodes(unique_parents)
        if parents[0] != 0:
            self._propagate(parents)

    def _propagate_index(self, index):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate_index(parent)

    def update(self, indices, values):
        self.sum_tree[indices] = values
        self._propagate(indices)
        current_max_value = np.max(values)
        self.max = max(current_max_value, self.max)

    # update single value given a tree index for efficiency
    def _update_index(self, index, value):
        self.sum_tree[index] = value  # set new value
        self._propagate_index(index)  # propagate value
        self.max = max(value, self.max)

    def append(
        self,
        transition,
        value,
    ):
        self._transitions[self.index] = transition

        self._update_index(self.index + self.tree_start, value)
        self.index = (self.index + 1) % self.max_size
        self.full = self.full or self.index == 0
        self.max = max(value, self.max)

    def _retrieve(self, indices, values):
        children_indices = indices * 2 + np.expand_dims(
            [1, 2], axis=1
        )  # Make matrix of children indices
        if children_indices[0, 0] >= self.sum_tree.shape[0]:
            return indices
        left_children_values = self.sum_tree[children_indices[0]]
        successor_choices = np.greater(values, left_children_values).astype(
            np.int32
        )  # Classify which values are in left or right branches
        successor_indices = children_indices[
            successor_choices, np.arange(indices.size)
        ]  # Use classification to index into the indices matrix
        successor_values = (
            values - successor_choices * left_children_values
        )  # Subtract the left branch values when searching in the right branch
        return self._retrieve(successor_indices, successor_values)

    def find(self, values):
        indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
        data_index = indices - self.tree_start
        return self.sum_tree[indices], data_index, indices

    def get(self, data_index):
        return self._transitions[data_index]

    def total(self):
        return self.sum_tree[0]
