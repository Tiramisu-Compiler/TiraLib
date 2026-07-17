from __future__ import annotations

import copy
import itertools
from typing import TYPE_CHECKING, List, Tuple

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
    """Skew two consecutive iterators ``i`` and ``j``.

    The transformed iterators are::

        [i']   [alpha  beta ] [i]
        [j'] = [gamma  sigma] [j]

    With two factors, ``[f_i, f_j]`` supplies the first matrix row and
    Tiramisu computes ``gamma`` and ``sigma`` such that
    ``f_i * sigma - f_j * gamma = 1``. With four factors,
    ``[alpha, beta, gamma, sigma]`` supplies the complete matrix directly;
    its determinant must have absolute value one.
    """

    def __init__(
        self,
        params: List[IteratorIdentifier | int],
        comps: List[str] = [],
    ):
        # Skewing takes either four parameters of the form L1, L2, f_i, f_j
        # or six parameters of the form L1, L2, alpha, beta, gamma, sigma.
        # 1. L1 and L2 are the levels of the iterators to skew
        # 2. Two factors fill the matrix's first row; four fill it row-major.

        assert len(params) in (4, 6)
        super().__init__(
            type=TiramisuActionType.SKEWING,
            params=params,
            comps=comps,
        )
        self.params = params
        self.comps = comps
        assert isinstance(params[0], tuple) and isinstance(params[1], tuple), (
            "The first two parameters must be tuples"
        )
        assert all(isinstance(factor, int) for factor in params[2:]), (
            "The skewing factors must be integers"
        )

        self.iterators: list[IteratorIdentifier] = params[:2]  # type: ignore
        self.factors: list[int] = params[2:]  # type: ignore

    def initialize_action_for_tree(self, tiramisu_tree: TiramisuTree):
        # clone the tree to be able to restore it later
        self.tree = copy.deepcopy(tiramisu_tree)
        for idx, iterator in enumerate(self.iterators):
            if iterator not in tiramisu_tree.iterators:
                self.iterators[idx] = self.tree.get_iterator_of_computation(
                    *iterator
                ).id

        if not self.comps:
            outermost_iterator_id = self.iterators[0]
            outermost_iterator = self.tree.iterators[outermost_iterator_id]

            # get the computations of the outermost iterator subtree
            # (includes the innermost iterator)
            self.comps = self.tree.get_iterator_subtree_computations(
                outermost_iterator.id
            )
            # sort the computations according to the absolute order
            self.comps.sort(
                key=lambda comp: self.tree.computations_absolute_order[comp]
            )

        self.set_string_representations(self.tree)

    def set_string_representations(self, tiramisu_tree: TiramisuTree):
        assert self.iterators is not None
        assert self.comps is not None
        assert len(self.params) in (4, 6)
        assert isinstance(self.iterators[0], tuple) and isinstance(
            self.iterators[1], tuple
        )

        self.tiramisu_optim_str = ""
        factors_str = ", ".join(str(factor) for factor in self.factors)
        for comp in self.comps:
            self.tiramisu_optim_str += f"{comp}.skew({self.iterators[0][1]}, {self.iterators[1][1]}, {factors_str});\n"  # noqa: E501

        self.str_representation = f"S(L{self.iterators[0][1]},L{self.iterators[1][1]},{','.join(str(factor) for factor in self.factors)},comps={self.comps})"  # noqa: E501

        self.legality_check_string = self.tiramisu_optim_str

    @classmethod
    def get_candidates(
        cls, program_tree: TiramisuTree
    ) -> dict[IteratorIdentifier, list[Tuple[IteratorIdentifier, IteratorIdentifier]]]:
        candidates: dict[
            IteratorIdentifier, list[Tuple[IteratorIdentifier, IteratorIdentifier]]
        ] = {}

        candidate_sections = program_tree.get_candidate_sections()

        for root_id in candidate_sections:
            candidates[root_id] = []
            for section in candidate_sections[root_id]:
                # Only consider sections with more than one iterator
                if len(section) > 1:
                    # Get all possible combinations of 2 successive iterators
                    candidates[root_id].extend(
                        [
                            (
                                comb[0],
                                comb[1],
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
    ) -> Tuple[int, int] | None:
        """
        Get the factors of the skewing optimization.
        This function calls the CompilingService to get the factors
        of the skewing optimization.
        Args:
            schedule (Schedule): The schedule of the program.
            loop_levels (List[int]): The levels of the loops to skew.
            comps_skewed_loops (List[str]): The computations of the loops to skew.
        Returns:
            Tuple[int, int] | None: The factors of the skewing optimization.
        """
        factors = CompilingService.call_skewing_solver(
            schedule, loop_levels, comps_skewed_loops
        )
        if factors is not None:
            return factors
        else:
            return None
