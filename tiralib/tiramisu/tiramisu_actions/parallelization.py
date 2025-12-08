from __future__ import annotations

import copy

from tiralib.tiramisu.tiramisu_iterator_node import IteratorIdentifier
from tiralib.tiramisu.tiramisu_tree import TiramisuTree

from tiralib.tiramisu.tiramisu_actions.tiramisu_action import (
    TiramisuAction,
    TiramisuActionType,
)


class Parallelization(TiramisuAction):
    """
    Parallelization optimization command.
    """

    def __init__(
        self,
        params: list[IteratorIdentifier],
        comps: list[str] = [],
    ):
        # Parallelization only takes one parameter the loop to
        # parallelize specified by a tuple (computation_name, iterator_level)
        assert len(params) == 1
        self.params = params
        self.comps = comps
        self.iterator_id = self.params[0]
        super().__init__(
            type=TiramisuActionType.PARALLELIZATION,
            params=params,
            comps=comps,
        )

    def initialize_action_for_tree(self, tiramisu_tree: TiramisuTree):
        # we save a copy of the tree to be able to restore it later
        self.tree = copy.deepcopy(tiramisu_tree)

        # user passed a different iteratorId than the main one
        if self.iterator_id not in tiramisu_tree.iterators:
            self.iterator_id = self.tree.get_iterator_of_computation(
                *self.iterator_id
            ).id
        if not self.comps:
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

        level = self.iterator_id[1]
        first_comp = self.comps[0]
        self.tiramisu_optim_str = f"{first_comp}.tag_parallel_level({level});\n"

        self.str_representation = f"P(L{level},comps={self.comps})"

        self.legality_check_string = f"prepare_schedules_for_legality_checks(true);\n    is_legal &= loop_parallelization_is_legal({level}, {{{', '.join([f'&{comp}' for comp in self.comps])}}});\n    {self.tiramisu_optim_str}"  # noqa: E501

    @classmethod
    def _get_candidates_of_node(
        cls, node_id: IteratorIdentifier, program_tree: TiramisuTree
    ) -> list[list[IteratorIdentifier]]:
        candidates: list[list[IteratorIdentifier]] = []
        node = program_tree.iterators[node_id]

        if node.child_iterators:
            candidates.append(
                [program_tree.iterators[child].id for child in node.child_iterators]
            )

            for child in node.child_iterators:
                candidates += cls._get_candidates_of_node(child, program_tree)

        return candidates

    @classmethod
    def get_candidates(
        cls, program_tree: TiramisuTree
    ) -> dict[IteratorIdentifier, list[list[IteratorIdentifier]]]:
        """Get the list of candidates for parallelization.

        Parameters:
        ----------
        `program_tree`: `TiramisuTree`
            The Tiramisu tree of the program.

        Returns:
        -------
        `dict[IteratorIdentifier, list[list[IteratorIdentifier]]]`
            Dictionary of candidates for parallelization of each root.
        """

        candidates: dict[IteratorIdentifier, list[list[IteratorIdentifier]]] = {}

        for root in program_tree.roots:
            rootId = program_tree.iterators[root].id
            candidates[rootId] = [[rootId]] + cls._get_candidates_of_node(
                root, program_tree
            )

        return candidates
