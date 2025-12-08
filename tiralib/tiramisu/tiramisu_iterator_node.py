from typing import Tuple

IteratorIdentifier = Tuple[str, int]


class IteratorNode:
    def __init__(
        self,
        name: str,
        parent_iterator: IteratorIdentifier | None,
        lower_bound: int | str,
        upper_bound: int | str,
        child_iterators: list[IteratorIdentifier],
        computations_list: list[str],
        level: int,
        id: IteratorIdentifier,
    ):
        self.name = name
        self.id: IteratorIdentifier = id
        self.parent_iterator = parent_iterator
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.child_iterators = child_iterators
        self.computations_list = computations_list
        self.level = level

    def add_child(self, child: IteratorIdentifier) -> None:
        self.child_iterators.append(child)

    def add_computation(self, comp: str) -> None:
        self.computations_list.append(comp)

    def has_non_rectangular(self) -> bool:
        return (
            isinstance(self.lower_bound, str) or isinstance(self.upper_bound, str)
        ) and (self.lower_bound != "UNK" and self.upper_bound != "UNK")

    def has_unkown_bounds(self) -> bool:
        return self.lower_bound == "UNK" or self.upper_bound == "UNK"

    def has_integer_bounds(self) -> bool:
        return isinstance(self.lower_bound, int) and isinstance(self.upper_bound, int)

    def __str__(self) -> str:
        return f"{self.name}(id={self.id}, lower_bound={self.lower_bound}, upper_bound={self.upper_bound}, child_iterators={self.child_iterators}, computations_list={self.computations_list}, level={self.level})"  # noqa: E501

    def __repr__(self) -> str:
        return self.__str__()
