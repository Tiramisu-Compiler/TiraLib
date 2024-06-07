from __future__ import annotations

import copy
import itertools
from typing import TYPE_CHECKING, Dict, List, Tuple

from tiralib.tiramisu.compiling_service import CompilingService
from tiralib.tiramisu.tiramisu_iterator_node import IteratorIdentifier
from tiralib.tiramisu.tiramisu_tree import TiramisuTree

if TYPE_CHECKING:
    from tiralib.tiramisu.schedule import Schedule

from tiralib.tiramisu.tiramisu_actions.tiramisu_action import (
    TiramisuAction,
    TiramisuActionType,
)


class Skewing(TiramisuAction):
    """
    Skewing optimization command.
    """

    def __init__(
        self,
        params: List[IteratorIdentifier | int],
        comps: List[str] | None = None,
    ):
        # Skewing takes four parameters of the form L1, L2, F3, F4
        # 1. L1 and L2 are the levels of the iterators to skew
        # 2. F3 and F4 are the factors of the skewing

        assert len(params) == 4
        self.params = params
        self.comps = comps

        self.iterators = params[:2]
        self.factors = params[2:]

        super().__init__(
            type=TiramisuActionType.SKEWING,
            params=params,
            comps=comps,
        )

    def initialize_action_for_tree(self, tiramisu_tree: TiramisuTree):
        # clone the tree to be able to restore it later
        self.tree = copy.deepcopy(tiramisu_tree)

        if self.comps is None:
            outermost_iterator_id = self.iterators[0]
            outermost_iterator = self.tree.get_iterator_of_computation(
                *outermost_iterator_id
            )

            # get the computations of the outermost iterator subtree
            # (includes the innermost iterator)
            self.comps = self.tree.get_iterator_subtree_computations(
                outermost_iterator.name
            )
            # sort the computations according to the absolute order
            self.comps.sort(
                key=lambda comp: self.tree.computations_absolute_order[comp]
            )

        self.set_string_representations(self.tree)

    def set_string_representations(self, tiramisu_tree: TiramisuTree):
        assert self.iterators is not None
        assert self.comps is not None
        assert len(self.params) == 4
        assert isinstance(self.iterators[0], tuple) and isinstance(
            self.iterators[1], tuple
        )

        self.tiramisu_optim_str = ""
        for comp in self.comps:
            self.tiramisu_optim_str += f"{comp}.skew({self.iterators[0][1]}, {self.iterators[1][1]}, {self.factors[0]}, {self.factors[1]});\n"  # noqa: E501

        self.str_representation = f"S(L{self.iterators[0][1]},L{self.iterators[1][1]},{self.factors[0]},{self.factors[1]},comps={self.comps})"  # noqa: E501

        self.legality_check_string = self.tiramisu_optim_str

    @classmethod
    def get_candidates(
        cls, program_tree: TiramisuTree
    ) -> Dict[str, List[Tuple[str, str]]]:
        candidates: Dict[str, List[Tuple[str, str]]] = {}

        candidate_sections = program_tree.get_candidate_sections()

        for root in candidate_sections:
            rootId = program_tree.iterators[root].id
            candidates[rootId] = []
            for section in candidate_sections[root]:
                # Only consider sections with more than one iterator
                if len(section) > 1:
                    # Get all possible combinations of 2 successive iterators
                    candidates[rootId].extend(
                        [
                            (
                                program_tree.iterators[comb[0]].id,
                                program_tree.iterators[comb[1]].id,
                            )
                            for comb in itertools.pairwise(section)
                        ]
                    )
        return candidates

    @classmethod
    def get_factors(
        cls,
        schedule: Schedule,
        loop_levels: List[int],
        comps_skewed_loops: List[str],
    ) -> Tuple[int, int]:
        factors = CompilingService.call_skewing_solver(
            schedule, loop_levels, comps_skewed_loops
        )
        if factors is not None:
            return factors
        else:
            return None
