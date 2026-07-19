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
        self.legality_comps: List[str] = []

        super().__init__(type=TiramisuActionType.UNROLLING, params=params, comps=comps)

    def initialize_action_for_tree(self, tiramisu_tree: TiramisuTree):
        # clone the tree to be able to restore it later
        self.tree = copy.deepcopy(tiramisu_tree)

        comp_name, level = self.iterator_id
        if level == -1:
            # L-1 means "innermost loop after previously applied
            # transformations"; resolve against the current tree state.
            def innermost_level_of(comp: str) -> int:
                levels = [
                    it.level
                    for it in tiramisu_tree.iterators.values()
                    if comp in it.computations_list
                ]
                if not levels:
                    raise ValueError(
                        f"Cannot resolve L-1 for unrolling: computation "
                        f"{comp!r} is not in the current tree."
                    )
                return max(levels)

            reference_comps = self.comps or [comp_name]
            innermost_levels = {c: innermost_level_of(c) for c in reference_comps}
            unique_levels = set(innermost_levels.values())
            if len(unique_levels) > 1:
                raise ValueError(
                    "U(L-1,...) requires all target computations to share "
                    f"the same innermost loop depth; got {innermost_levels}. "
                    "Split non-perfectly-nested computations into separate "
                    "U(...) actions."
                )
            self.iterator_id = (comp_name, unique_levels.pop())

        if self.iterator_id not in tiramisu_tree.iterators:
            self.iterator_id = self.tree.get_iterator_of_computation(
                *self.iterator_id
            ).id

        iterator = tiramisu_tree.iterators[self.iterator_id]

        # The specialized Tiramisu legality check operates on every computation
        # in the selected loop.  Keep that historical behavior even when a
        # serialized schedule explicitly unrolls only a subset of them.
        self.legality_comps = tiramisu_tree.get_iterator_subtree_computations(
            iterator.id
        )
        self.legality_comps.extend(
            comp for comp in self.comps if comp not in self.legality_comps
        )
        self.legality_comps.sort(
            key=lambda comp: tiramisu_tree.computations_absolute_order[comp]
        )

        if not self.comps:
            # Direct API calls that omit `comps` retain the original behavior:
            # unroll the complete iterator subtree.
            self.comps = self.legality_comps.copy()

        self.set_string_representations(tiramisu_tree)

    def set_string_representations(self, tiramisu_tree: TiramisuTree):
        assert self.iterator_id is not None
        assert self.unrolling_factor is not None
        assert self.comps is not None

        loop_level = self.iterator_id[1]
        unrolling_factor = self.unrolling_factor

        def make_optim_str(comps: List[str]) -> str:
            unroll_lines = [
                f"{comp}.unroll({loop_level},{unrolling_factor});" for comp in comps
            ]
            # Unrolling splits a shared loop independently for each computation.
            # Re-issue the ordering for a multi-computation group so the legality
            # path retains TiraLib's original full-subtree behavior.
            if len(comps) > 1:
                first = comps[0]
                innermost = (
                    f"(isl_map_dim({first}.get_schedule(), isl_dim_out) - 2) / 2 - 1"
                )
                refusion_lines = [f"int __unroll_innermost = {innermost};"]
                for prev, cur in zip(comps, comps[1:]):
                    refusion_lines.append(f"{cur}.after({prev}, __unroll_innermost);")
                return (
                    "{\n    " + "\n    ".join(unroll_lines + refusion_lines) + "\n    }"
                )
            return "\n    ".join(unroll_lines)

        self.tiramisu_optim_str = make_optim_str(self.comps)
        self.str_representation = (
            f"U(L{str(loop_level)},{str(unrolling_factor)},comps={self.comps})"
        )
        legality_check = f"prepare_schedules_for_legality_checks(true);\n    is_legal &= loop_unrolling_is_legal({loop_level}, {{{', '.join([f'&{comp}' for comp in self.legality_comps])}}});"  # noqa: E501
        if self.comps == self.legality_comps:
            # Full-loop unrolling keeps the original TiraLib legality path,
            # including applying the transformation before the final global
            # dependency check.
            self.legality_str_representation = self.str_representation
            self.legality_check_string = (
                f"{legality_check}\n    {self.tiramisu_optim_str}"
            )
        else:
            # Tiramisu's transformed-schedule legality machinery cannot
            # represent a subset-unrolled fused loop.  Its dedicated unrolling
            # check can: validate the complete loop group without mutating the
            # legality schedule, then apply the exact subset only for execution.
            self.legality_str_representation = (
                f"UCheck(L{str(loop_level)},{str(unrolling_factor)},"
                f"comps={self.legality_comps})"
            )
            self.legality_check_string = legality_check

    @classmethod
    def get_candidates(cls, program_tree: TiramisuTree) -> List[IteratorIdentifier]:
        candidates: List[IteratorIdentifier] = []

        for iterator in program_tree.iterators:
            iterator_node = program_tree.iterators[iterator]
            if not iterator_node.child_iterators and iterator_node.computations_list:
                candidates.append(program_tree.iterators[iterator].id)

        return candidates
