import pytest

import tests.utils as test_utils
from tiralib.tiramisu import tiramisu_actions
from tiralib.tiramisu.schedule import Schedule
from tiralib.tiramisu.tiramisu_actions.parallelization import Parallelization
from tiralib.config import BaseConfig
from tests.utils import benchmark_program_test_sample


def test_execute():
    BaseConfig.init()
    test_program = benchmark_program_test_sample()

    schedule = Schedule(test_program)
    assert schedule.tree
    schedule.add_optimizations([Parallelization(params=[("comp02", 0)])])
    results = schedule.execute(min_runs=10)

    assert results is not None
    assert len(results) == 10


def test_min_runs_execution():
    """Test to ensure the schedule executes at least 'min_runs' times even if the 'time_budget' is insufficient"""
    BaseConfig.init()
    test_program = benchmark_program_test_sample()
    schedule = Schedule(test_program)

    exec_time = schedule.execute(1)[0]
    results = schedule.execute(min_runs=10, max_runs=20, time_budget=2 * exec_time)
    assert results is not None
    assert len(results) >= 10


def test_time_budget_enforcement():
    """Test to verify that execution stops once the 'time_budget' is exhausted, with partial measurements saved."""
    BaseConfig.init()
    test_program = benchmark_program_test_sample()
    schedule = Schedule(test_program)

    exec_time = schedule.execute(1)[0]
    time_budget = 5 * exec_time
    results = schedule.execute(min_runs=0, max_runs=20, time_budget=time_budget)

    assert results is not None
    assert 0 < len(results) < 20
    assert 0 < sum(results) <= time_budget


def test_max_runs_limitation():
    """Test to ensure the schedule stops executing after 'max_runs' even if the time budget allows more."""
    BaseConfig.init()
    test_program = benchmark_program_test_sample()
    schedule = Schedule(test_program)

    exec_time = schedule.execute(1)[0]
    time_budget = 50 * exec_time
    results = schedule.execute(min_runs=2, max_runs=5, time_budget=time_budget)

    assert results is not None
    assert len(results) == 5
    assert sum(results) < time_budget


def test_zero_min_runs():
    """Test to ensure proper behavior when 'min_runs' is set to 0, expecting no compulsory runs."""
    BaseConfig.init()
    test_program = benchmark_program_test_sample()
    schedule = Schedule(test_program)

    exec_time = schedule.execute(1)[0]
    time_budget = 0.1 * exec_time
    results = schedule.execute(min_runs=0, max_runs=5, time_budget=time_budget)

    assert results is not None
    assert len(results) == 0


def test_unlimited_max_runs():
    """Test to ensure that, when 'max_runs' is not set, we get as many runs as possible within the time budget."""
    BaseConfig.init()
    test_program = benchmark_program_test_sample()
    schedule = Schedule(test_program)

    exec_time = schedule.execute(1)[0]
    time_budget = 10 * exec_time
    results = schedule.execute(min_runs=0, time_budget=time_budget)

    assert results is not None
    assert len(results) > 1
    assert sum(results) < time_budget


def test_is_legal():
    BaseConfig.init()
    test_program = benchmark_program_test_sample()

    schedule = Schedule(test_program)
    assert schedule.tree

    schedule.add_optimizations([Parallelization(params=[("comp02", 0)])])
    legality = schedule.is_legal()

    assert legality is True


def test_copy():
    BaseConfig.init()
    original = Schedule(benchmark_program_test_sample())
    assert original.tree

    original.add_optimizations([Parallelization(params=[("comp02", 0)])])

    copy = original.copy()

    assert original is not copy
    assert original.tiramisu_program is copy.tiramisu_program
    assert original.optims_list is not copy.optims_list
    assert len(original.optims_list) == len(copy.optims_list)
    for optim in original.optims_list:
        assert optim in copy.optims_list


def test_str_representation():
    BaseConfig.init()
    test_program = benchmark_program_test_sample()

    schedule = Schedule(test_program)
    assert schedule.tree

    schedule.add_optimizations([Parallelization(params=[("comp02", 0)])])

    assert str(schedule) == "P(L0,comps=['comp02'])"


def test_from_sched_str_unrolling_l_neg_1():
    BaseConfig.init()

    # Pre-tiling: innermost level of `w` in gemver is 1 (it's a 2D loop).
    test_program = test_utils.multiple_roots_sample()
    pre_schedule = Schedule.from_sched_str("U(L-1,4,comps=['w'])", test_program)
    assert pre_schedule is not None
    pre_unrolling = pre_schedule.optims_list[0]
    pre_level = pre_unrolling.iterator_id[1]
    assert pre_level == 1
    # Serialization carries the resolved level, not L-1.
    assert "L-1" not in str(pre_schedule)
    assert "L1" in str(pre_schedule)

    # Post-tiling: T2 on (w,0)(w,1) introduces 2 new loops; the new
    # innermost level for `w` should be deeper than before.
    post_schedule = Schedule.from_sched_str(
        "T2(L0,L1,32,32,comps=['w'])|U(L-1,4,comps=['w'])", test_program
    )
    assert post_schedule is not None
    post_unrolling = post_schedule.optims_list[1]
    post_level = post_unrolling.iterator_id[1]
    assert post_level > pre_level


def test_from_sched_str():
    BaseConfig.init()

    test_program = test_utils.multiple_roots_sample()

    schedule = Schedule(test_program)
    assert schedule.tree

    schedule.add_optimizations(
        [
            Parallelization(params=[("A_hat", 0)]),
            tiramisu_actions.Interchange(params=[("x_temp", 0), ("x_temp", 1)]),
            tiramisu_actions.Fusion(params=[("A_hat", 0), ("x_temp", 0)]),
            tiramisu_actions.Tiling2D(params=[("w", 0), ("w", 1), 4, 4]),
            tiramisu_actions.Unrolling(params=[("x", 0), 4]),
            tiramisu_actions.Reversal(params=[("x", 0)]),
        ]
    )

    sched_str = str(schedule)

    new_schedule = Schedule.from_sched_str(sched_str, test_program)

    assert new_schedule is not None

    assert len(new_schedule.optims_list) == len(schedule.optims_list)

    for idx, optim in enumerate(schedule.optims_list):
        assert optim == new_schedule.optims_list[idx]

    schedule = Schedule(test_program)
    assert schedule.tree

    schedule.add_optimizations(
        [
            tiramisu_actions.Skewing([("x_temp", 0), ("x_temp", 1), 1, 1]),
        ]
    )

    sched_str = str(schedule)

    new_schedule = Schedule.from_sched_str(sched_str, test_program)

    assert new_schedule is not None
    assert len(new_schedule.optims_list) == len(schedule.optims_list)

    for idx, optim in enumerate(schedule.optims_list):
        assert optim == new_schedule.optims_list[idx]

    schedule = Schedule(test_program)
    assert schedule.tree

    schedule.add_optimizations(
        [
            tiramisu_actions.Skewing(
                [("x_temp", 0), ("x_temp", 1), 1, 0, 1, 1]
            ),
        ]
    )

    sched_str = str(schedule)
    new_schedule = Schedule.from_sched_str(sched_str, test_program)

    assert new_schedule is not None
    assert len(new_schedule.optims_list) == len(schedule.optims_list)
    assert schedule.optims_list == new_schedule.optims_list

    test_program = test_utils.tiling_3d_sample()

    schedule = Schedule(test_program)
    assert schedule.tree

    schedule.add_optimizations(
        [
            tiramisu_actions.Tiling3D(
                [("comp00", 0), ("comp00", 1), ("comp00", 2), 4, 4, 4]
            ),
        ]
    )

    sched_str = str(schedule)

    new_schedule = Schedule.from_sched_str(sched_str, test_program)

    assert new_schedule is not None
    assert len(new_schedule.optims_list) == len(schedule.optims_list)

    for idx, optim in enumerate(schedule.optims_list):
        assert optim == new_schedule.optims_list[idx]


def test_from_sched_str_preserves_parallelization_computations():
    BaseConfig.init()
    test_program = test_utils.multiple_roots_sample()

    schedule = Schedule.from_sched_str(
        "P(L0,comps=['A_hat','x_temp','A_hat'])", test_program
    )

    parallelization = schedule.optims_list[0]
    assert parallelization.comps == ["A_hat", "x_temp"]
    assert parallelization.tiramisu_optim_str == (
        "A_hat.tag_parallel_level(0);\n"
        "x_temp.tag_parallel_level(0);\n"
    )


def test_from_sched_str_unknown_action_raises():
    BaseConfig.init()
    test_program = benchmark_program_test_sample()
    with pytest.raises(ValueError, match="Failed to parse schedule token"):
        Schedule.from_sched_str("Z(L0,comps=['comp02'])", test_program)


def test_from_sched_str_malformed_body_raises():
    BaseConfig.init()
    test_program = benchmark_program_test_sample()
    # Known leading char "P" but missing the closing paren.
    with pytest.raises(ValueError, match="Failed to parse schedule token"):
        Schedule.from_sched_str("P(L0,comps=['comp02']", test_program)
