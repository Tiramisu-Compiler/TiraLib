from __future__ import annotations

import copy
import itertools
import random
from typing import Tuple

from tiralib.tiramisu.tiramisu_iterator_node import IteratorIdentifier
from tiralib.tiramisu.tiramisu_tree import TiramisuTree


from tiralib.tiramisu.tiramisu_actions.tiramisu_action import (
    TiramisuAction,
    TiramisuActionType,
)


class TilingGeneral(TiramisuAction):
    """
    General Tiling for non perfectly nested loops optimization command.
    """

    def __init__(
        self,
        params: list[IteratorIdentifier | int],
        comps: list[str] | None = None,
    ):
        # General Tiling takes the iterator to tile and the tile size as
        # parameters: [iterator1, iterator2, ..., iteratorN,
        #  tile_size1, tile_size2, ..., tile_sizeN]

        # assert len(params) == 4
        assert len(params) % 2 == 0

        self.nbr_iterators = len(params) // 2
        for i in range(self.nbr_iterators):
            assert isinstance(params[i], tuple)
            assert isinstance(params[self.nbr_iterators + i], int)
        self.params = params
        self.comps = comps

        self.iterators: list[IteratorIdentifier] = params[0 : self.nbr_iterators]
        self.tile_sizes: list[int] = params[self.nbr_iterators :]

        super().__init__(
            type=TiramisuActionType.TILING_GENERAL,
            params=params,
            comps=comps,
        )

    def initialize_action_for_tree(self, tiramisu_tree: TiramisuTree):
        # clone the tree to be able to restore it later
        self.tree = copy.deepcopy(tiramisu_tree)
        for idx, iterator in enumerate(self.iterators):
            if iterator not in tiramisu_tree.iterators:
                self.iterators[idx] = self.tree.get_iterator_of_computation(
                    *iterator
                ).id

        self.tile_sizes_dict = {
            iterator: self.params[self.nbr_iterators + i]
            for i, iterator in enumerate(self.iterators)
        }

        if self.comps is None:
            outermost_iterator_id = min(self.iterators, key=lambda x: x[1])

            outermost_iterator = self.tree.iterators[outermost_iterator_id]
            # get the computations of the outermost iterator to tile
            # which include the computations of the other iterators
            self.comps = self.tree.get_iterator_subtree_computations(
                outermost_iterator.id
            )

            # sort the computations according to the absolute order
            self.comps.sort(
                key=lambda comp: self.tree.computations_absolute_order[comp]
            )

        self.set_string_representations(self.tree)

    def set_string_representations(self, tiramisu_tree: TiramisuTree):
        assert self.comps is not None
        assert self.iterators is not None
        assert self.tile_sizes_dict is not None
        assert self.iterators is not None

        all_comps = tiramisu_tree.computations
        if len(all_comps) > 1:
            all_comps.sort(
                key=lambda comp: tiramisu_tree.computations_absolute_order[comp]
            )

        self.tiramisu_optim_str = ""

        for comp in self.comps:
            loop_levels = []
            tile_sizes = []
            comp_depth = tiramisu_tree.get_iterator_of_computation(comp).level

            # assuming that self.iterators are consecutive and sorted by level
            for iterator_to_tile in self.iterators:
                # if the comp is not that deep, no need for further checks
                if comp_depth < iterator_to_tile[1]:
                    break
                # get the canonical iterator id at the tiling level an check if it matches the
                # iterators that need to be tiled
                comp_iterator_id = tiramisu_tree.get_iterator_of_computation(
                    comp, iterator_to_tile[1]
                ).id
                if comp_iterator_id == iterator_to_tile:
                    loop_levels.append(iterator_to_tile[1])
                    tile_sizes.append(self.tile_sizes_dict[iterator_to_tile])
                else:
                    # if the comp's branch diverges at this level, no need to check the upcomming levels
                    break
            assert len(loop_levels) >= 1

            loop_levels_and_factors = [str(loop_level) for loop_level in loop_levels]
            loop_levels_and_factors.extend([str(tile_size) for tile_size in tile_sizes])

            self.tiramisu_optim_str += (
                f"{comp}.tile({', '.join(loop_levels_and_factors)});\n"
            )

        str_levels_and_sizes = [
            f"L{iterator[1]}" if isinstance(iterator, tuple) else str(iterator)
            for iterator in self.params
        ]
        self.str_representation = (
            f"TG({','.join(str_levels_and_sizes)},comps={self.comps})"
        )

        self.legality_check_string = self.tiramisu_optim_str

    @classmethod
    def get_candidates(
        cls, program_tree: TiramisuTree
    ) -> dict[str, list[Tuple[str, str]]]:
        candidates: dict[str, list[Tuple[str, str]]] = {}

        candidate_sections = cls.get_imperfect_candidate_sections(program_tree)

        for root in candidate_sections:
            rootId = program_tree.iterators[root].id
            candidates[rootId] = []
            for section in candidate_sections[root]:
                # Only consider sections with more than one iterator
                if len(section) > 1:
                    first_node = program_tree.iterators[section[0]]
                    if len(first_node.child_iterators) > 1:
                        candidates[rootId].append(
                            [
                                program_tree.iterators[iterator].id
                                for iterator in section
                            ]
                        )
                    else:
                        # Get all possible combinations of
                        # 2 or 3 successive iterators
                        tmp_candidates = []
                        tmp_candidates.extend(list(itertools.pairwise(section)))
                        successive_3_iterators = [
                            tuple(section[i : i + 3])  # noqa: E203
                            for i in range(len(section) - 2)
                        ]
                        tmp_candidates.extend(successive_3_iterators)

                        for candidate in tmp_candidates:
                            perfect = True
                            for i in range(len(candidate) - 1):
                                tmp_iterator = program_tree.iterators[candidate[i]]
                                if tmp_iterator.computations_list:
                                    perfect = False
                                    break
                            if not perfect:
                                candidates[rootId].append(
                                    [
                                        program_tree.iterators[iterator].id
                                        for iterator in candidate
                                    ]
                                )

        return candidates

    def get_fusion_levels(
        self,
        ordered_computations: list[str],
        tiramisu_tree: TiramisuTree,
    ):
        fusion_levels: list[int] = []
        # for every pair of successive computations
        # get the shared iterator level
        for comp1, comp2 in itertools.pairwise(ordered_computations):
            # get the shared iterator level
            iter_comp_1 = tiramisu_tree.get_iterator_of_computation(comp1)
            iter_comp_2 = tiramisu_tree.get_iterator_of_computation(comp2)
            fusion_level: int | None = None

            # get the shared iterator level
            while iter_comp_1.name != iter_comp_2.name:
                if iter_comp_1.level > iter_comp_2.level:
                    # parent is None ->
                    # the iterators don't have a common parent
                    if iter_comp_1.parent_iterator is None:
                        fusion_level = -1
                        break
                    else:
                        iter_comp_1 = tiramisu_tree.iterators[
                            iter_comp_1.parent_iterator
                        ]
                else:
                    if iter_comp_2.parent_iterator is None:
                        fusion_level = -1
                        break
                    else:
                        iter_comp_2 = tiramisu_tree.iterators[
                            iter_comp_2.parent_iterator
                        ]

            # same iterator
            if fusion_level is None:
                fusion_level = iter_comp_1.level

            if comp1 in self.comps and comp2 in self.comps:
                nbr_addition = 0
                tmp_iterator = iter_comp_1
                while tmp_iterator is not None:
                    if tmp_iterator.id in self.iterators:
                        nbr_addition += 1
                    if tmp_iterator.parent_iterator is None:
                        tmp_iterator = None
                    else:
                        tmp_iterator = tiramisu_tree.iterators[
                            tmp_iterator.parent_iterator
                        ]
                fusion_level += nbr_addition

            fusion_levels.append(fusion_level)

        return fusion_levels

    @classmethod
    def get_imperfect_candidate_sections(
        cls,
        tiramisu_tree: TiramisuTree,
    ) -> dict[IteratorIdentifier, list[list[IteratorIdentifier]]]:
        """
        Returns a dictionary with lists of candidate sections for
        each root iterator.

        Returns:
        -------

        `candidate_sections`: `dict[str, list[list[str]]]`
            Dictionary with lists of candidate sections for each root iterator.
        """

        candidate_sections = {}
        for root in tiramisu_tree.roots:
            nodes_to_visit = [root]
            list_candidate_sections = []
            for node in nodes_to_visit:
                (
                    candidate_section,
                    new_nodes_to_visit,
                ) = cls._get_imperfect_section_of_node(tiramisu_tree, node)
                list_candidate_sections.append(candidate_section)
                nodes_to_visit.extend(new_nodes_to_visit)
            candidate_sections[root] = list_candidate_sections
        return candidate_sections

    @classmethod
    def _get_imperfect_section_of_node(
        cls, tiramisu_tree: TiramisuTree, node_name: str
    ) -> Tuple[list[str], list[str]]:
        candidate_section = [node_name]
        current_node = tiramisu_tree.iterators[node_name]

        if len(current_node.child_iterators) > 1:
            candidate_section.extend(current_node.child_iterators)
            nodes_to_visit = current_node.child_iterators.copy()
            while nodes_to_visit:
                node = nodes_to_visit.pop(0)
                tmp_node = tiramisu_tree.iterators[node]
                candidate_section.extend(tmp_node.child_iterators)
                nodes_to_visit.extend(tmp_node.child_iterators)
        else:
            while len(current_node.child_iterators) == 1:
                next_node_name = current_node.child_iterators[0]
                candidate_section.append(next_node_name)
                current_node = tiramisu_tree.iterators[next_node_name]

        if current_node.child_iterators:
            return candidate_section, current_node.child_iterators
        return candidate_section, []

    @classmethod
    def from_candidate(
        cls,
        candidate: list[str],
        tiramisu_tree: TiramisuTree,
        random_tile_sizes: list[int] = [2, 4, 8, 10, 16, 32, 64],
    ):
        iterators = [
            tiramisu_tree.get_iterator_id_from_name(iterator_name)
            for iterator_name in candidate
        ]
        tile_sizes = [random.choice(random_tile_sizes) for _ in iterators]
        return TilingGeneral(iterators + tile_sizes)
