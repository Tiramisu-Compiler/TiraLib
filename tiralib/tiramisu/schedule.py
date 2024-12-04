from __future__ import annotations

import ast
import re
from copy import deepcopy
from typing import TYPE_CHECKING, List

from tiralib.tiramisu.compiling_service import CompilingService
from tiralib.tiramisu.tiramisu_actions.tiramisu_action import TiramisuActionType
from tiralib.tiramisu.tiramisu_tree import TiramisuTree

if TYPE_CHECKING:
    from .tiramisu_actions.tiramisu_action import TiramisuAction

from tiralib.tiramisu import tiramisu_actions
from tiralib.tiramisu.tiramisu_program import TiramisuProgram


class Schedule:
    """
    A schedule is a list of optimizations to be applied to a Tiramisu program.

    Parameters
    ----------
    `tiramisu_program` : TiramisuProgram
        The Tiramisu program to which the schedule will be applied.
    `optims_list` : List[TiramisuAction]
        The list of optimizations to be applied to the Tiramisu program.
    """

    def __init__(self, tiramisu_program: TiramisuProgram | None = None) -> None:
        self.tiramisu_program = tiramisu_program
        self.optims_list: List[TiramisuAction] = []
        if tiramisu_program:
            self.tree = deepcopy(tiramisu_program.tree)
        else:
            self.tree = None
        self.legality: bool | None = None

    def set_tiramisu_program(self, tiramisu_program: TiramisuProgram) -> None:
        self.tiramisu_program = tiramisu_program
        self.tree = deepcopy(tiramisu_program.tree)

    def add_optimizations(self, list_optim_cmds: List[TiramisuAction]) -> None:
        """
        Adds a list of optimizations to the schedule while maintaining the
        schedule tree. The order of the optimizations in the list is important.

        Parameters
        ----------
        `list_optim_cmds` : `List[TiramisuAction]`
            The list of optimizations to be added to the schedule.
        """
        if self.tree is None:
            raise Exception("No Tiramisu program to apply the schedule to")

        self.legality = None

        for optim_cmd in list_optim_cmds:
            # initialize action for the schedule tree
            optim_cmd.initialize_action_for_tree(self.tree)

            self.optims_list.append(optim_cmd)

            # Fusion, distribution and tiling are special cases,
            # we need to get the new tree with the new fusion levels
            if (
                optim_cmd.is_fusion()
                or optim_cmd.is_distribution()
                or optim_cmd.is_any_tiling()
            ):
                self.update_tree_from_isl_ast()

    def pop_optimization(self) -> TiramisuAction:
        """
        Removes the last optimization from the schedule and returns it.
        """
        action = self.optims_list.pop()
        self.update_tree_from_isl_ast()
        return action

    def execute(
        self,
        min_runs: int = 1,
        max_runs: int | None = None,
        time_budget: float | None = None,
        delete_files: bool = True,
    ) -> List[float]:
        """
        Applies the schedule to the Tiramisu program and performs a sequence
        of execution time measurements.

        Parameters
        ----------
        `min_runs` : int
            Defines the minimal number of times the schedule will be executed.
            The parameter guarantees that the execution time will be measured
            at least min_run times regardless of the set time budget.
        `time_budget` : float or None
            Defines the duration (in milliseconds) allocated for execution
            time measurements.
            Once the time_budget is consumed, no more runs will be performed
            and ongoing execution will be aborted.
            Measurements completed within the time_budget (if any) will be saved.
            The min_runs first measurements are not abortable and will execute
            till completion even if time_budget is fully consumed.
        `max_runs` : int or None
            Defines the maximal number of times the schedule will be executed.
            This allows to stop taking measurements before time_budget is
            fully consumed. max_runs is only taken into consideration if
            time_budget is defined. If time_budget is defined and max_runs is
            None, max_run will be set to infinity.
        `delete_files` : bool
            Defines whether to delete the temporary generated files or not.
        Returns
        -------
        The execution time of the Tiramisu program after applying the schedule.
        """
        if self.tiramisu_program is None:
            raise Exception("No Tiramisu program to apply the schedule to")

        if self.tiramisu_program.server:
            result = self.tiramisu_program.server.run(
                operation="execution",
                schedule=self,
                nbr_executions=min_runs,
            )
            if result.legality is False:
                raise Exception("Schedule is not legal")

            return result.exec_times

        if self.legality is None and self.optims_list:
            self.is_legal()

        if self.legality is False:
            raise Exception("Schedule is not legal")

        return CompilingService.get_cpu_exec_times(
            self.tiramisu_program,
            self.optims_list,
            min_runs,
            max_runs,
            time_budget,
            delete_files,
        )

    def is_legal(self, with_ast: bool = False) -> bool:
        """
        Checks if the schedule is legal.

        Returns
        -------
        Boolean indicating if the schedule is legal.
        """

        if self.tiramisu_program is None:
            raise Exception("No Tiramisu program to apply the schedule to")

        if self.tiramisu_program.server:
            result = self.tiramisu_program.server.run("legality", self)
            self.tree = TiramisuTree.from_isl_ast_string_list(
                isl_ast_string_list=result.isl_ast.split("\n")
            )
            self.legality = result.legality

            # Update the skewing factors if they are not set
            if result.additional_info:
                if "skewing_factors" in result.additional_info:
                    for action in self.optims_list:
                        if action.type == TiramisuActionType.SKEWING:
                            if action.params[2] == 0:
                                factors = result.additional_info.replace(
                                    "skewing_factors:", ""
                                ).split(",")
                                factors = [int(factor) for factor in factors]
                                action.params[2] = factors[0]
                                action.params[3] = factors[1]
                                action.factors = factors
                                action.set_string_representations(self.tree)
            return result.legality

        legality, new_tree = CompilingService.compile_legality(self, with_ast=with_ast)

        assert isinstance(legality, bool)
        self.legality = legality
        if with_ast:
            assert new_tree
            self.tree = new_tree
        return self.legality

    def update_tree_from_isl_ast(self):
        """
        Updates the schedule tree from the isl ast.
        """
        if self.tiramisu_program is None:
            raise Exception("No Tiramisu program to apply the schedule to")

        if self.tiramisu_program.server:
            result = self.tiramisu_program.server.run("legality", self)
            self.tree = TiramisuTree.from_isl_ast_string_list(
                isl_ast_string_list=result.isl_ast.split("\n")
            )
        else:
            isl_ast_str = CompilingService.compile_isl_ast_tree(
                tiramisu_program=self.tiramisu_program, schedule=self
            )
            self.tree = TiramisuTree.from_isl_ast_string_list(isl_ast_str.split("\n"))

    @classmethod
    def from_sched_str(
        cls, sched_str: str, tiramisu_program: TiramisuProgram
    ) -> "Schedule":
        schedule = cls(tiramisu_program)
        assert schedule.tree
        for optimization_str in sched_str.split("|"):
            if optimization_str == "":
                continue
            if optimization_str[0] == "P":
                # extract loop level and comps using P\(L(\d),comps=\[([\w',]*)
                regex = r"P\(L(\d),comps=\[([\w', ]*)\]\)"
                match = re.match(regex, optimization_str)
                if match:
                    loop_level = int(match.group(1))
                    comps = match.group(2).split(",")
                    comps = [comp.strip("' ") for comp in comps]
                    schedule.add_optimizations(
                        [
                            tiramisu_actions.Parallelization(
                                [(comps[0], loop_level)],
                            )
                        ]
                    )

            elif optimization_str[0] == "U":
                # extract loop level, factor and comps using
                # U\(L(\d),(\d+),comps=\[([\w',]*)\]\)
                regex = r"U\(L(\d),(\d+),comps=\[([\w', ]*)\]\)"
                match = re.match(regex, optimization_str)
                if match:
                    loop_level = int(match.group(1))
                    factor = int(match.group(2))
                    comps = match.group(3).split(",")
                    comps = [comp.strip("' ") for comp in comps]
                    schedule.add_optimizations(
                        [
                            tiramisu_actions.Unrolling(
                                [
                                    (comps[0], loop_level),
                                    factor,
                                ],
                            )
                        ]
                    )
            elif optimization_str[0] == "I":
                regex = r"I\(L(\d),L(\d),comps=\[([\w', ]*)\]\)"
                match = re.match(regex, optimization_str)
                if match:
                    first_loop_level = int(match.group(1))
                    second_loop_level = int(match.group(2))
                    comps = match.group(3).split(",")
                    comps = [comp.strip("' ") for comp in comps]
                    schedule.add_optimizations(
                        [
                            tiramisu_actions.Interchange(
                                [
                                    (comps[0], first_loop_level),
                                    (comps[0], second_loop_level),
                                ],
                            )
                        ]
                    )
            elif optimization_str[0] == "R":
                regex = r"R\(L(\d),comps=\[([\w', ]*)\]\)"
                match = re.match(regex, optimization_str)
                if match:
                    loop_level = int(match.group(1))
                    comps = match.group(2).split(",")
                    comps = [comp.strip("' ") for comp in comps]
                    schedule.add_optimizations(
                        [
                            tiramisu_actions.Reversal(
                                [(comps[0], loop_level)],
                            )
                        ]
                    )
            elif optimization_str[:2] == "T2":
                regex = r"T2\(L(\d),L(\d),(\d+),(\d+),comps=\[([\w', ]*)\]\)"
                match = re.match(regex, optimization_str)
                if match:
                    outer_loop_level = int(match.group(1))
                    inner_loop_level = int(match.group(2))
                    outer_loop_factor = int(match.group(3))
                    inner_loop_factor = int(match.group(4))
                    comps = match.group(5).split(",")
                    comps = [comp.strip("' ").strip() for comp in comps]
                    schedule.add_optimizations(
                        [
                            tiramisu_actions.Tiling2D(
                                [
                                    (comps[0], outer_loop_level),
                                    (comps[0], inner_loop_level),
                                    outer_loop_factor,
                                    inner_loop_factor,
                                ],
                            )
                        ]
                    )
            elif optimization_str[:2] == "T3":
                regex = (
                    r"T3\(L(\d),L(\d),L(\d),(\d+),(\d+),(\d+),comps=\[([\w', ]*)\]\)"  # noqa: E501
                )
                match = re.match(regex, optimization_str)
                if match:
                    outer_loop_level = int(match.group(1))
                    middle_loop_level = int(match.group(2))
                    inner_loop_level = int(match.group(3))
                    outer_loop_factor = int(match.group(4))
                    middle_loop_factor = int(match.group(5))
                    inner_loop_factor = int(match.group(6))
                    comps = match.group(7).split(",")
                    comps = [comp.strip("' ").strip() for comp in comps]
                    schedule.add_optimizations(
                        [
                            tiramisu_actions.Tiling3D(
                                [
                                    (comps[0], outer_loop_level),
                                    (comps[0], middle_loop_level),
                                    (comps[0], inner_loop_level),
                                    outer_loop_factor,
                                    middle_loop_factor,
                                    inner_loop_factor,
                                ],
                            )
                        ]
                    )
            elif optimization_str[0] == "S":
                regex = r"S\(L(\d),L(\d),(-?\d+),(-?\d+),comps=\[([\w', ]*)\]\)"
                match = re.match(regex, optimization_str)
                if match:
                    outer_loop_level = int(match.group(1))
                    inner_loop_level = int(match.group(2))
                    outer_loop_factor = int(match.group(3))
                    inner_loop_factor = int(match.group(4))
                    comps = match.group(5).split(",")
                    comps = [comp.strip("' ").strip() for comp in comps]
                    schedule.add_optimizations(
                        [
                            tiramisu_actions.Skewing(
                                [
                                    (comps[0], outer_loop_level),
                                    (comps[0], inner_loop_level),
                                    outer_loop_factor,
                                    inner_loop_factor,
                                ],
                            )
                        ]
                    )
            elif optimization_str[0] == "F":
                regex = r"F\(L(\d),comps=\[([\w', ]*)\]\)"
                match = re.match(regex, optimization_str)
                if match:
                    loop_level = int(match.group(1))
                    comps = match.group(2).split(",")
                    comps = [comp.strip("' ").strip() for comp in comps]
                    schedule.add_optimizations(
                        [
                            tiramisu_actions.Fusion(
                                [
                                    (comps[0], loop_level),
                                    (comps[1], loop_level),
                                ],
                            )
                        ]
                    )
            elif optimization_str[0] == "D":
                regex = r"D\(L(\d),comps=\[([\w', ]*)\],distribution=([\[\]'\w, ]*)\)"  # noqa: E501
                match = re.match(regex, optimization_str)
                if match:
                    loop_level = int(match.group(1))
                    comps = match.group(2).split(",")
                    comps = [comp.strip("' ").strip() for comp in comps]
                    distribution = ast.literal_eval(match.group(3))
                    assert isinstance(distribution, list)
                    schedule.add_optimizations(
                        [
                            tiramisu_actions.Distribution(
                                [
                                    (comps[0], loop_level),
                                ],
                                distribution,
                            )
                        ]
                    )
            elif optimization_str[0] == "E":
                regex = r"E\(comps=\[([\w', ]*)\]\)"
                match = re.match(regex, optimization_str)
                if match:
                    comp = match.group(1).strip("' ")
                    schedule.add_optimizations(
                        [
                            tiramisu_actions.Expansion(
                                [comp],
                            )
                        ]
                    )
            elif optimization_str[:2] == "TG":
                raise NotImplementedError
                # regex =

        return schedule

    def __str__(self) -> str:
        """
        Generates a string representation of the schedule.
        """

        sched_str = "|".join([str(optim) for optim in self.optims_list])

        return sched_str

    def __repr__(self) -> str:
        return self.__str__()

    def copy(self) -> Schedule:
        """
        Returns a copy of the schedule.
        """
        new_schedule = Schedule(self.tiramisu_program)
        new_schedule.add_optimizations(self.optims_list)
        return new_schedule

    def __len__(self) -> int:
        return len(self.optims_list)

    def __bool__(self) -> bool:
        return bool(self.optims_list)
