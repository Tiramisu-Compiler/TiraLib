from tiralib.tiramisu.schedule import Schedule
from tiralib.tiramisu.tiramisu_actions.parallelization import Parallelization
from tiralib.tiramisu.tiramisu_program import TiramisuProgram


def parallelize_first_legal_outermost(
    tiramisu_program: TiramisuProgram,
) -> Schedule:
    schedule = Schedule(tiramisu_program)
    candidates_per_root = Parallelization.get_candidates(tiramisu_program.tree)

    for root in tiramisu_program.tree.roots:
        tmp_schedule = schedule.copy()
        for candidate in candidates_per_root[tiramisu_program.tree.iterators[root].id]:
            for node in candidate:
                tmp_schedule.add_optimizations(
                    [
                        Parallelization(
                            [
                                (
                                    node[0],
                                    node[1],
                                )
                            ]
                        )
                    ]
                )
            if tmp_schedule.is_legal():
                schedule = tmp_schedule
                break

    if not schedule.optims_list:
        return None
    return schedule
