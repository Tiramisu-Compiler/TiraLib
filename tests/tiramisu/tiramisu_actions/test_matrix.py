from tests.utils import interchange_example, load_test_data
from tiralib.config.config import BaseConfig
from tiralib.tiramisu import tiramisu_actions
from tiralib.tiramisu.schedule import Schedule
from tiralib.tiramisu.tiramisu_actions.matrix import MatrixTransform
from tiralib.tiramisu.tiramisu_program import TiramisuProgram


def test_matrix_init():
    BaseConfig.init()
    matrixTransform = MatrixTransform([1, 0, 0, 0, 0, 1, 0, 1, 0], ["comp00"])
    assert matrixTransform.params == [1, 0, 0, 0, 0, 1, 0, 1, 0]
    assert matrixTransform.comps == ["comp00"]


def test_initialize_action_for_tree():
    BaseConfig.init()
    sample = interchange_example()
    matrix = MatrixTransform([1, 0, 0, 0, 0, 1, 0, 1, 0], ["comp00"])
    matrix.initialize_action_for_tree(sample.tree)
    assert matrix.params == [1, 0, 0, 0, 0, 1, 0, 1, 0]
    assert matrix.comps == ["comp00"]
    matrix.tree is not None


def test_set_string_representations():
    BaseConfig.init()
    sample = interchange_example()
    matrix = MatrixTransform([1, 0, 0, 0, 0, 1, 0, 1, 0], ["comp00"])
    schedule = Schedule(sample)
    schedule.add_optimizations([matrix])
    assert (
        matrix.tiramisu_optim_str
        == "comp00.matrix_transform({{1,0,0},{0,0,1},{0,1,0}});"
    )


def test_legality_check():
    BaseConfig.init()
    sample = interchange_example()
    schedule = Schedule(sample)
    assert schedule.tree
    schedule.add_optimizations(
        [MatrixTransform([1, 0, 0, 0, 0, 1, 0, 1, 0], ["comp00"])],
    )
    legality_string = schedule.optims_list[0].legality_check_string
    assert legality_string == "comp00.matrix_transform({{1,0,0},{0,0,1},{0,1,0}});"


def test_matrix_transform_application():
    BaseConfig.init()
    _, test_cpps = load_test_data()
    sample = TiramisuProgram.init_server(
        test_cpps["function837782"],
        load_annotations=True,
        load_tree=True,
        reuseServer=True,
    )

    matrix_schedule = Schedule(sample)
    assert matrix_schedule.tree
    matrix_schedule.add_optimizations(
        [MatrixTransform([1, 0, 0, 0, 0, 1, 0, 1, 0], ["comp00"])],
    )

    matrix_result = sample.server.run("execution", matrix_schedule, 1)

    interchange_schedule = Schedule(sample)
    interchange_schedule.add_optimizations(
        [tiramisu_actions.Interchange([("comp00", 1), ("comp00", 2)], comps=["comp00"])]
    )

    interchange_result = sample.server.run("execution", interchange_schedule, 1)

    assert interchange_result.halide_ir == matrix_result.halide_ir
