import pytest

import tests.utils as test_utils
from athena.tiramisu.schedule import Schedule
from athena.tiramisu.tiramisu_actions.matrix import MatrixTransform
from athena.tiramisu.tiramisu_actions.tiling_2d import Tiling2D
from athena.tiramisu.tiramisu_actions.tiramisu_action import CannotApplyException
from athena.utils.config import BaseConfig
from tests.utils import interchange_example


def test_matrix_init():
    BaseConfig.init()
    sample = interchange_example()
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
        [MatrixTransform([1, 0, 0, 0, 0, 1, 0, 1, 0], ["comp00"])]
    )
    legality_string = schedule.optims_list[0].legality_check_string
    assert legality_string == "comp00.matrix_transform({{1,0,0},{0,0,1},{0,1,0}});"
