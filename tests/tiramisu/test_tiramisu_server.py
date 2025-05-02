from pathlib import Path
import tests.utils as test_utils
from tiralib.config import BaseConfig
from tiralib.tiramisu import tiramisu_actions
from tiralib.tiramisu.schedule import Schedule
from tiralib.tiramisu.tiramisu_program import TiramisuProgram


def test_init_server():
    BaseConfig.init()

    cpp_code = Path("examples/function_gemver_MINI_generator.cpp").read_text()
    sample = TiramisuProgram.init_server(
        cpp_code=cpp_code,
        load_isl_ast=True,
        load_tree=True,
    )

    assert sample.name == "function_gemver_MINI"
    assert sample.isl_ast_string is not None
    assert sample.cpp_code is not None
    assert len(sample.tree.computations) > 0


def test_init_server_annotations():
    BaseConfig.init()

    cpp_code = Path("examples/function_gemver_MINI_generator.cpp").read_text()
    sample = TiramisuProgram.init_server(
        cpp_code=cpp_code,
        load_annotations=True,
        load_tree=True,
        reuse_server=True,
    )

    assert sample.name == "function_gemver_MINI"
    assert sample.annotations is not None


def test_get_legality():
    BaseConfig.init()

    cpp_code = Path("examples/function_gemver_MINI_generator.cpp").read_text()
    sample = TiramisuProgram.init_server(
        cpp_code=cpp_code,
        load_isl_ast=True,
        load_tree=True,
        reuse_server=True,
    )

    schedule = Schedule(sample)

    schedule.add_optimizations(
        [
            tiramisu_actions.Interchange(params=[("x_temp", 0), ("x_temp", 1)]),
        ]
    )

    assert schedule.is_legal() is True


def test_get_exec_times():
    BaseConfig.init()

    cpp_code = Path("examples/function_gemver_MINI_generator.cpp").read_text()
    sample = TiramisuProgram.init_server(
        cpp_code=cpp_code,
        load_isl_ast=True,
        load_tree=True,
        reuse_server=True,
    )

    schedule = Schedule(sample)

    schedule.add_optimizations(
        [
            tiramisu_actions.Interchange(params=[("x_temp", 0), ("x_temp", 1)]),
        ]
    )

    assert len(schedule.execute()) > 0


def test_get_skewing_factors():
    BaseConfig.init()

    _, test_cpps = test_utils.load_test_data()

    tiramisu_func = TiramisuProgram.init_server(
        cpp_code=test_cpps["function550013"],
        load_annotations=True,
        load_tree=True,
        reuse_server=True,
    )

    schedule = Schedule(tiramisu_func)

    schedule.add_optimizations(
        [tiramisu_actions.Skewing([("comp00", 0), ("comp00", 1), 0, 0])]
    )

    assert schedule.is_legal()
    assert isinstance(schedule.optims_list[0], tiramisu_actions.Skewing)
    assert schedule.optims_list[0].factors == [1, 1]
