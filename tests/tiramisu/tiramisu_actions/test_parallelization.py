import tests.utils as test_utils
from tiralib.tiramisu.schedule import Schedule
from tiralib.tiramisu.tiramisu_actions.parallelization import Parallelization
from tiralib.config import BaseConfig


def test_parallelization_init():
    parallelization = Parallelization([("comp01", 1)])
    assert parallelization.iterator_id == ("comp01", 1)
    assert parallelization.comps == []

    parallelization = Parallelization([("comp01", 1)], comps=["comp01"])
    assert parallelization.iterator_id == ("comp01", 1)
    assert parallelization.comps == ["comp01"]

    parallelization = Parallelization([("comp01", 1)], comps=["comp01", "comp01"])
    assert parallelization.comps == ["comp01"]


def test_initialize_action_for_tree():
    t_tree = test_utils.tree_test_sample()
    parallelization = Parallelization([("comp01", 1)])
    parallelization.initialize_action_for_tree(t_tree)

    assert parallelization.iterator_id == ("comp01", 1)
    assert parallelization.comps == ["comp01"]


def test_set_string_representations():
    BaseConfig.init()
    sample = test_utils.benchmark_program_test_sample()
    parallelization = Parallelization([("comp02", 0)])
    schedule = Schedule(sample)
    schedule.add_optimizations([parallelization])

    assert parallelization.tiramisu_optim_str == "comp02.tag_parallel_level(0);\n"


def test_set_string_representations_tags_each_distinct_computation():
    t_tree = test_utils.tree_test_sample_2()
    parallelization = Parallelization(
        [("comp05", 1)],
        comps=["comp05", "comp06", "comp06", "comp07"],
    )
    parallelization.initialize_action_for_tree(t_tree)

    assert parallelization.comps == ["comp05", "comp06", "comp07"]
    assert parallelization.tiramisu_optim_str == (
        "comp05.tag_parallel_level(1);\n"
        "comp06.tag_parallel_level(1);\n"
        "comp07.tag_parallel_level(1);\n"
    )
    assert parallelization.legality_check_string == (
        "prepare_schedules_for_legality_checks(true);\n"
        "    is_legal &= loop_parallelization_is_legal(1, "
        "{&comp05, &comp06, &comp07});\n"
        "    comp05.tag_parallel_level(1);\n"
        "comp06.tag_parallel_level(1);\n"
        "comp07.tag_parallel_level(1);\n"
    )


def test_get_candidates():
    BaseConfig.init()
    sample = test_utils.benchmark_program_test_sample()
    candidates = Parallelization.get_candidates(sample.tree)
    assert candidates == {
        ("comp02", 0): [
            [("comp02", 0)],
            [("comp02", 1)],
            [("comp02", 2)],
        ]
    }


def test_legality_check():
    BaseConfig.init()

    sample = test_utils.benchmark_program_test_sample()
    schedule = Schedule(sample)
    assert schedule.tree
    parallelization = Parallelization([("comp02", 0)])

    schedule.add_optimizations([parallelization])
    legality_string = schedule.optims_list[0].legality_check_string
    assert (
        legality_string
        == "prepare_schedules_for_legality_checks(true);\n    is_legal &= loop_parallelization_is_legal(0, {&comp02});\n    comp02.tag_parallel_level(0);\n"  # noqa: E501
    )
