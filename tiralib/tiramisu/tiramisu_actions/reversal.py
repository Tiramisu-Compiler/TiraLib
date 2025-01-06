from __future__ import annotations

import copy
from typing import Dict, List

from tiralib.tiramisu.tiramisu_iterator_node import IteratorIdentifier
from tiralib.tiramisu.tiramisu_tree import TiramisuTree

from tiralib.tiramisu.tiramisu_actions.tiramisu_action import (
    TiramisuAction,
    TiramisuActionType,
)


class Reversal(TiramisuAction):
    """
    Reversal optimization command.
    """

    def __init__(
        self, params: List[IteratorIdentifier], comps: List[str] | None = None
    ):
        # Reversal takes one parameter of the loop to reverse
        assert len(params) == 1

        self.iterator_id = params[0]
        self.params = params
        self.comps = comps
        super().__init__(type=TiramisuActionType.REVERSAL, params=params, comps=comps)

    def initialize_action_for_tree(self, tiramisu_tree: TiramisuTree):
        # clone the tree to be able to restore it later
        self.tree = copy.deepcopy(tiramisu_tree)

        # user passed a different iteratorId than the main one
        if self.iterator_id not in tiramisu_tree.iterators:
            self.iterator_id = self.tree.get_iterator_of_computation(
                *self.iterator_id
            ).id

        if self.comps is None:
            iterator = tiramisu_tree.iterators[self.iterator_id]

            self.comps = tiramisu_tree.get_iterator_subtree_computations(iterator.id)
            # order the computations by their absolute order
            self.comps.sort(
                key=lambda comp: tiramisu_tree.computations_absolute_order[comp]
            )

        self.set_string_representations(tiramisu_tree)

    def set_string_representations(self, tiramisu_tree: TiramisuTree):
        assert self.iterator_id is not None
        assert self.comps is not None

        self.tiramisu_optim_str = ""
        level = self.iterator_id[1]
        for comp in self.comps:
            self.tiramisu_optim_str += f"{comp}.loop_reversal({level});\n"

        self.str_representation = f"R(L{level},comps={self.comps})"

        self.legality_check_string = self.tiramisu_optim_str

    @classmethod
    def get_candidates(cls, program_tree: TiramisuTree) -> Dict[str, List[str]]:
        candidates: Dict[str, List[str]] = {}
        for root in program_tree.roots:
            rootId = program_tree.iterators[root].id
            candidates[rootId] = [rootId] + [
                program_tree.iterators[iterator].id
                for iterator in program_tree.iterators[root].child_iterators
            ]
            nodes_to_visit = program_tree.iterators[root].child_iterators.copy()

            while nodes_to_visit:
                node = nodes_to_visit.pop(0)
                node_children = program_tree.iterators[node].child_iterators
                nodes_to_visit.extend(node_children)
                candidates[rootId].extend(
                    [program_tree.iterators[node].id for node in node_children]
                )

        return candidates

    def transform_tree(self, program_tree: TiramisuTree):
        node = program_tree.iterators[self.params[0]]

        # Reverse the loop bounds
        if isinstance(node.lower_bound, int) and isinstance(node.upper_bound, int):
            # Halide way of reversing to keep increment 1
            node.lower_bound, node.upper_bound = (
                -node.upper_bound,
                -node.lower_bound,
            )
        else:
            node.lower_bound, node.upper_bound = (
                node.upper_bound,
                node.lower_bound,
            )
