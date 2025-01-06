import re
from typing import Tuple

from tiralib.tiramisu.tiramisu_iterator_node import (
    IteratorIdentifier,
    IteratorNode,
)


class TiramisuTree:
    """This class represents the tree structure of a Tiramisu program.
    It is composed of a list of IteratorNode objects, each of which represents
    an iterator in the Tiramisu program. Each IteratorNode object contains
    information about its parent iterator, child iterators, lower and upper
    bounds, and the computations that it is associated with.

    Attributes:
    ----------
    `roots`: `list[IteratorIdentifier]`
        list of the identifiers of the root iterators in the Tiramisu program.
    `iterators`: `dict[IteratorIdentifier, IteratorNode]`
        Dictionary of IteratorNode objects, indexed by the id of
        the iterator.
    `computations`: `list[str]`
        list of names of the computations in the Tiramisu program.
    """

    def __init__(self) -> None:
        self.roots: list[IteratorIdentifier] = []
        self.iterators: dict[IteratorIdentifier, IteratorNode] = {}
        self.computations: list[str] = []
        self.computations_absolute_order: dict[str, int] = {}

    def add_root(self, root: IteratorIdentifier) -> None:
        self.roots.append(root)

    def add_computation(self, comp: str) -> None:
        self.computations.append(comp)

    @classmethod
    def from_annotations(cls, annotations: dict) -> "TiramisuTree":
        """
        Creates a TiramisuTree object from the annotations of a
        Tiramisu program.

        Parameters:
        ----------
        `annotations`: `dict`
            Annotations of a Tiramisu program.

        Returns:
        -------
        `tiramisu_space`: `TiramisuTree`
        """
        tiramisu_space = cls()

        iterators = annotations["iterators"]

        tiramisu_space.computations_absolute_order = {
            comp: annotations["computations"][comp]["absolute_order"]
            for comp in annotations["computations"]
        }

        # order keys of computations_absolute_order by their values
        tiramisu_space.computations = [
            comp
            for comp, _ in sorted(
                tiramisu_space.computations_absolute_order.items(),
                key=lambda item: item[1],
            )
        ]

        iterator_name_to_id = {}
        for iterator in iterators:
            for computation in tiramisu_space.computations:
                # get index of the iterator in the computation iterators
                comp_iterators: list[str] = annotations["computations"][computation][
                    "iterators"
                ]
                try:
                    iterator_index = comp_iterators.index(iterator)
                except ValueError:
                    continue
                iterator_name_to_id[iterator] = (computation, iterator_index)
                break

        for iterator in iterators:
            iterator_id = iterator_name_to_id[iterator]
            parent_iterator = iterators[iterator]["parent_iterator"]
            iterator_level = None
            if parent_iterator is None:
                tiramisu_space.add_root(iterator_id)
                iterator_level = 0
            else:
                iterator_level = (
                    tiramisu_space.iterators[iterator_name_to_id[parent_iterator]].level
                    + 1
                )

            # get the computations that are associated with this iterator
            # ordered by their absolute order
            ordered_node_comps = [
                comp
                for comp in tiramisu_space.computations
                if comp in iterators[iterator]["computations_list"]
            ]

            try:
                # integer bounds
                lower_bound = int(iterators[iterator]["lower_bound"])
            except ValueError:
                # non rectangular bounds
                lower_bound = iterators[iterator]["lower_bound"]

            try:
                # integer bounds
                upper_bound = int(iterators[iterator]["upper_bound"])
            except ValueError:
                # non rectangular bounds
                upper_bound = iterators[iterator]["upper_bound"]

            tiramisu_space.iterators[iterator_id] = IteratorNode(
                name=iterator,
                id=iterator_id,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                child_iterators=[
                    iterator_name_to_id[child_iterator]
                    for child_iterator in iterators[iterator]["child_iterators"]
                ],
                computations_list=ordered_node_comps,
                parent_iterator=iterator_name_to_id.get(parent_iterator),
                level=iterator_level,
            )

        # order the roots by their first comp's absolute order
        root_with_order = []
        for root in tiramisu_space.roots:
            first_comp = root[0]
            first_comp_order = tiramisu_space.computations_absolute_order[first_comp]
            root_with_order.append((root, first_comp_order))

        tiramisu_space.roots = [
            root for root, _ in sorted(root_with_order, key=lambda item: item[1])
        ]
        return tiramisu_space

    @classmethod
    def from_isl_ast_string_list(cls, isl_ast_string_list: list[str]) -> "TiramisuTree":
        tiramisu_tree = cls()
        tiramisu_tree.computations_absolute_order = {}
        tiramisu_tree.computations = []
        tiramisu_tree.iterators = {}
        tiramisu_tree.roots = []

        name_to_iterator_identifier: dict[str, IteratorIdentifier] = {}

        level_iterator_map: dict[int, list[str]] = {}
        current_absolute_order = 1
        upper_bound_regex = r".*<=\s*(.*)"
        iterator_duplicates: dict[str, int] = {}
        for line_idx, str_line in enumerate(isl_ast_string_list):
            if "|iterator|" in str_line:
                (
                    iterator_level_str,
                    _,
                    iterator_name,
                    lower_bound_str,
                    loop_condition,
                    increment,
                ) = str_line.split("|")
                iterator_level = int(iterator_level_str)
                try:
                    lower_bound = int(lower_bound_str)
                except ValueError:
                    # Lower bound is not an integer so we keep it string
                    pass

                # Get the upper bound from the loop condition
                matched_upper_bound = re.match(upper_bound_regex, loop_condition)
                if matched_upper_bound:
                    upper_bound = matched_upper_bound.group(1)
                else:
                    upper_bound = loop_condition
                try:
                    upper_bound = int(upper_bound)
                except ValueError:
                    # Upper bound is not an integer so we keep it string
                    pass
                if iterator_name in name_to_iterator_identifier:
                    iterator_duplicates[iterator_name] = (
                        iterator_duplicates[iterator_name] + 1
                    )
                    iterator_name = (
                        iterator_name + "_" + str(iterator_duplicates[iterator_name])
                    )
                else:
                    iterator_duplicates[iterator_name] = 0

                first_comp: str = ""
                for j in range(line_idx + 1, len(isl_ast_string_list)):
                    if "|computation|" in isl_ast_string_list[j]:
                        first_comp = isl_ast_string_list[j].split("|")[2]
                        break
                iterator_id = (first_comp, iterator_level)
                name_to_iterator_identifier[iterator_name] = iterator_id

                tiramisu_tree.iterators[iterator_id] = IteratorNode(
                    name=iterator_name,
                    id=iterator_id,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    child_iterators=[],
                    computations_list=[],
                    parent_iterator=(
                        None
                        if iterator_level == 0
                        else level_iterator_map[iterator_level - 1][-1]
                    ),
                    level=iterator_level,
                )
                if iterator_level not in level_iterator_map:
                    level_iterator_map[iterator_level] = []
                level_iterator_map[iterator_level].append(iterator_id)

                if iterator_level == 0:
                    tiramisu_tree.roots.append(iterator_id)
                else:
                    # Add the iterator to its parent's child iterators
                    # (the last iterator we added in the previous level)
                    tiramisu_tree.iterators[
                        level_iterator_map[iterator_level - 1][-1]
                    ].child_iterators.append(iterator_id)

            elif "|computation|" in str_line:
                level_str, _, comp_name = str_line.split("|")
                line_idx = int(level_str)
                tiramisu_tree.computations.append(comp_name)

                # Add the computation to its iterator's computations list
                # (the last iterator we added in the previous level)
                # After making sure that the computation is in an actual loop.
                # Some computations might occur in a single iteration loop which
                # would be romved by ISL
                if line_idx != 0:
                    tiramisu_tree.iterators[
                        level_iterator_map[line_idx - 1][-1]
                    ].computations_list.append(comp_name)

                # Add the computation to the absolute order dict
                tiramisu_tree.computations_absolute_order[comp_name] = (
                    current_absolute_order
                )
                current_absolute_order += 1
        return tiramisu_tree

    def _get_subtree_representation(self, node_id: IteratorIdentifier) -> str:
        representation = ""
        representation += (
            "   " * self.iterators[node_id].level
            + "-> "
            + repr(self.iterators[node_id])
            + "\n"
        )
        comps_and_iterators = [
            (comp, "comp") for comp in self.iterators[node_id].computations_list
        ]
        comps_and_iterators += [
            (iterator, "iterator")
            for iterator in self.iterators[node_id].child_iterators
        ]

        # sort them by computations_absolute_order
        comps_and_iterators = sorted(
            comps_and_iterators,
            key=lambda item: (
                self.computations_absolute_order[item[0]]
                if item[1] == "comp"
                else self.computations_absolute_order[item[0][0]]
            ),
        )

        for comp_or_iterator, type in comps_and_iterators:
            if type == "comp":
                representation += (
                    "   " * (self.iterators[node_id].level + 1)
                    + "- "
                    + comp_or_iterator
                    + "\n"
                )
            else:
                representation += self._get_subtree_representation(comp_or_iterator)
        return representation

    def get_candidate_sections(
        self,
    ) -> dict[IteratorIdentifier, list[list[IteratorIdentifier]]]:
        """
        Returns a dictionary with lists of candidate sections for each
        root iterator.

        Returns:
        -------

        `candidate_sections`: `dict[IteratorIdentifier, list[list[IteratorIdentifier]]]`
            Dictionary with lists of candidate sections for each root iterator.
        """

        candidate_sections = {}
        for root in self.roots:
            nodes_to_visit = [root]
            list_candidate_sections = []
            for node in nodes_to_visit:
                candidate_section, new_nodes_to_visit = self._get_section_of_node(node)
                list_candidate_sections.append(candidate_section)
                nodes_to_visit.extend(new_nodes_to_visit)
            candidate_sections[self.iterators[root].id] = list_candidate_sections
        return candidate_sections

    def _get_section_of_node(self, node_name: str) -> Tuple[list[str], list[str]]:
        candidate_section = [self.iterators[node_name].id]
        current_node = self.iterators[node_name]

        while (
            len(current_node.child_iterators) == 1
            and len(current_node.computations_list) == 0
        ):
            next_node_name = current_node.child_iterators[0]
            current_node = self.iterators[next_node_name]
            candidate_section.append(current_node.id)

        if current_node.child_iterators:
            return candidate_section, current_node.child_iterators
        return candidate_section, []

    def get_iterator_subtree_computations(
        self, candidate_node_id: IteratorIdentifier
    ) -> list[str]:
        """Get the list of computations impacted by this node

        Parameters:
        ----------
        `candidate_node`: `IteratorNode`
            The candidate node for parallelization.

        `program_tree`: `TiramisuTree`
            The Tiramisu tree of the program.

        Returns:
        -------
        `list`
            list of computations impacted by the node
        """

        computations: list[str] = []
        candidate_node = self.iterators[candidate_node_id]

        computations += candidate_node.computations_list

        for child in candidate_node.child_iterators:
            computations += self.get_iterator_subtree_computations(child)

        return computations

    def get_iterator_levels(
        self, iterators_list: list[IteratorIdentifier]
    ) -> list[int]:
        """
        This function returns the levels of the iterators in the computation
        """
        return [self.iterators[iterator].level for iterator in iterators_list]

    def get_root_of_node(self, iterator_id: IteratorIdentifier) -> IteratorIdentifier:
        # Get the root node of the iterator
        current_node_id = iterator_id

        while self.iterators[current_node_id].parent_iterator:
            current_node_id = self.iterators[current_node_id].parent_iterator

        if current_node_id is None:
            raise ValueError("The iterator has no root node")

        return current_node_id

    def get_iterator_of_computation(
        self, computation_name: str, level: int | None = None
    ):
        """
        This function returns the iterator of the computation
        """
        computation_iterator = None
        for iterator in self.iterators:
            if computation_name in self.iterators[iterator].computations_list:
                computation_iterator = self.iterators[iterator]
                break

        if computation_iterator is None:
            raise ValueError("The computation is not in the tree")

        if level is not None:
            while computation_iterator.level != level:
                computation_iterator = self.iterators[
                    computation_iterator.parent_iterator
                ]

        return computation_iterator

    def get_iterator_id_from_name(self, iterator_name: str) -> IteratorIdentifier:
        """
        This function returns the id of the iterator
        """
        for iterator in self.iterators:
            if self.iterators[iterator].name == iterator_name:
                return iterator

    def set_iterator_ids(self) -> None:
        for iterator in self.iterators.values():
            iterator.id = self.get_iterator_id_from_name(iterator.name)

    def __repr__(self) -> str:
        representation = ""

        for root in self.roots:
            representation += self._get_subtree_representation(root)

        return representation
