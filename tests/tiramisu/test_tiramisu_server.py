from athena.tiramisu import tiramisu_actions
from athena.tiramisu.schedule import Schedule
from athena.tiramisu.tiramisu_program import TiramisuProgram
from athena.utils.config import BaseConfig
import tests.utils as test_utils


def test_init_server():
    BaseConfig.init()

    sample = TiramisuProgram.init_server(
        "examples/function_gemver_MINI_generator.cpp",
        load_isl_ast=True,
        load_tree=True,
        from_file=True,
    )

    assert sample.name == "function_gemver_MINI"
    assert sample.isl_ast_string is not None
    assert sample.original_str is not None
    assert sample.comps is not None and len(sample.comps) > 0


def test_init_server_annotations():
    BaseConfig.init()

    sample = TiramisuProgram.init_server(
        "examples/function_gemver_MINI_generator.cpp",
        load_annotations=True,
        load_tree=True,
        from_file=True,
        reuseServer=True,
    )

    assert sample.name == "function_gemver_MINI"
    assert sample.annotations is not None


def test_get_legality():
    BaseConfig.init()

    sample = TiramisuProgram.init_server(
        "examples/function_gemver_MINI_generator.cpp",
        load_isl_ast=True,
        load_tree=True,
        from_file=True,
        reuseServer=True,
    )

    schedule = Schedule(sample)

    schedule.add_optimizations(
        [
            tiramisu_actions.Interchange(
                params=[("x_temp", 0), ("x_temp", 1)]
            ),
        ]
    )

    assert schedule.is_legal() is True


def test_get_exec_times():
    BaseConfig.init()

    sample = TiramisuProgram.init_server(
        "examples/function_gemver_MINI_generator.cpp",
        load_isl_ast=True,
        load_tree=True,
        from_file=True,
        reuseServer=True,
    )

    schedule = Schedule(sample)

    schedule.add_optimizations(
        [
            tiramisu_actions.Interchange(
                params=[("x_temp", 0), ("x_temp", 1)]
            ),
        ]
    )

    len(schedule.execute()) > 0


def test_get_skewing_factors():
    BaseConfig.init()

    test_data, test_cpps = test_utils.load_test_data()

    tiramisu_func = TiramisuProgram.init_server(
        # data=test_data["function550013"],
        original_code=test_cpps["function550013"],
        load_annotations=True,
        load_tree=True,
        reuseServer=True,
    )

    schedule = Schedule(tiramisu_func)

    schedule.add_optimizations(
        [tiramisu_actions.Skewing([("comp00", 0), ("comp00", 1), 0, 0])]
    )

    assert schedule.is_legal()

    assert schedule.optims_list[0].factors == [1, 1]
