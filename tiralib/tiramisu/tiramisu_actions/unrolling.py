from __future__ import annotations

import copy
from typing import List

from tiralib.tiramisu.tiramisu_iterator_node import IteratorIdentifier
from tiralib.tiramisu.tiramisu_tree import TiramisuTree

from tiralib.tiramisu.tiramisu_actions.tiramisu_action import (
    TiramisuAction,
    TiramisuActionType,
)


class Unrolling(TiramisuAction):
    """
    Unrolling optimization command.
    """

    def __init__(
        self,
        params: List[IteratorIdentifier | int],
        comps: List[str] = [],
    ):
        # Unrolling takes 2 parameters: the iterator to unroll and the
        # unrolling factor
        assert len(params) == 2
        assert isinstance(params[0], tuple) and isinstance(params[1], int), (
            f"Invalid unrolling parameters: {params}"
        )
        self.iterator_id = params[0]
        self.unrolling_factor = params[1]

        self.params = params
        self.comps = comps

        super().__init__(type=TiramisuActionType.UNROLLING, params=params, comps=comps)

    def initialize_action_for_tree(self, tiramisu_tree: TiramisuTree):
        # clone the tree to be able to restore it later
        self.tree = copy.deepcopy(tiramisu_tree)
        if self.iterator_id not in tiramisu_tree.iterators:
            self.iterator_id = self.tree.get_iterator_of_computation(
                *self.iterator_id
            ).id

        if not self.comps:
            iterator = tiramisu_tree.iterators[self.iterator_id]

            # Get the computations that are in the loop to be unrolled
            self.comps = tiramisu_tree.get_iterator_subtree_computations(iterator.id)
            # order the computations by their absolute order
            self.comps.sort(
                key=lambda comp: tiramisu_tree.computations_absolute_order[comp]
            )

        self.set_string_representations(tiramisu_tree)

    def set_string_representations(self, tiramisu_tree: TiramisuTree):
        assert self.iterator_id is not None
        assert self.unrolling_factor is not None
        assert self.comps is not None

        self.tiramisu_optim_str = ""
        loop_level = self.iterator_id[1]
        unrolling_factor = self.unrolling_factor
        # for comp in self.comps:
        self.tiramisu_optim_str = "\n    ".join(
            [f"{comp}.unroll({loop_level},{unrolling_factor});" for comp in self.comps]
        )
        self.str_representation = (
            f"U(L{str(loop_level)},{str(unrolling_factor)},comps={self.comps})"
        )

        self.legality_check_string = f"prepare_schedules_for_legality_checks(true);\n    is_legal &= loop_unrolling_is_legal({loop_level}, {{{', '.join([f'&{comp}' for comp in self.comps])}}});\n    {self.tiramisu_optim_str}"  # noqa: E501

    @classmethod
    def get_candidates(cls, program_tree: TiramisuTree) -> List[IteratorIdentifier]:
        candidates: List[IteratorIdentifier] = []

        for iterator in program_tree.iterators:
            iterator_node = program_tree.iterators[iterator]
            if not iterator_node.child_iterators and iterator_node.computations_list:
                candidates.append(program_tree.iterators[iterator].id)

        return candidates
