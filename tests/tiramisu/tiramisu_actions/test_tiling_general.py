import tests.utils as test_utils
from tiralib.tiramisu.tiramisu_actions.tiling_general import TilingGeneral

from tiralib.config import BaseConfig


def test_tiling_general_init():
    tiling_general = TilingGeneral([("comp00", 0), ("comp00", 1), 32, 32])
    assert tiling_general.iterators == [("comp00", 0), ("comp00", 1)]
    assert tiling_general.tile_sizes == [32, 32]
    assert tiling_general.comps == []

    tiling_general = TilingGeneral([("comp00", 0), ("comp00", 1), 32, 32], ["comp00"])
    assert tiling_general.iterators == [("comp00", 0), ("comp00", 1)]
    assert tiling_general.tile_sizes == [32, 32]
    assert tiling_general.comps == ["comp00"]


def test_initialize_action_for_tree():
    BaseConfig.init()
    sample = test_utils.gramschmidt_sample()
    tiling_general = TilingGeneral(
        [("R_up_init", 1), ("R_up", 2), ("A_out", 2), 10, 10, 10]
    )
    tiling_general.initialize_action_for_tree(sample.tree)
    assert tiling_general.iterators == [
        ("R_up_init", 1),
        ("R_up", 2),
        ("A_out", 2),
    ]
    assert tiling_general.tile_sizes_dict == {
        ("R_up_init", 1): 10,
        ("R_up", 2): 10,
        ("A_out", 2): 10,
    }
    assert tiling_general.comps == ["R_up_init", "R_up", "A_out"]


def test_set_string_representations():
    BaseConfig.init()
    sample = test_utils.gramschmidt_sample()
    tiling_general = TilingGeneral([("R_up_init", 1), ("R_up", 2), 10, 5])
    tiling_general.initialize_action_for_tree(sample.tree)
    assert tiling_general.iterators == [("R_up_init", 1), ("R_up", 2)]

    assert (
        tiling_general.tiramisu_optim_str
        == "R_up_init.tile(1, 10);\nR_up.tile(1, 2, 10, 5);\nA_out.tile(1, 10);\nclear_implicit_function_sched_graph();\n    nrm_init.then(nrm_comp,0).then(R_diag,0).then(Q_out,0).then(R_up_init,0).then(R_up,1).then(A_out,1);\n"  # noqa: E501
    )

    assert (
        tiling_general.str_representation
        == "TG(L1,L2,10,5,comps=['R_up_init', 'R_up', 'A_out'])"
    )


def test_get_candidates():
    BaseConfig.init()
    sample = test_utils.gramschmidt_sample()
    candidates = TilingGeneral.get_candidates(sample.tree)
    assert candidates == {
        ("nrm_init", 0): [
            [
                ("nrm_init", 0),
                ("nrm_comp", 1),
                ("Q_out", 1),
                ("R_up_init", 1),
                ("R_up", 2),
                ("A_out", 2),
            ],
            [
                ("R_up_init", 1),
                ("R_up", 2),
                ("A_out", 2),
            ],
        ]
    }

    tree = test_utils.tree_test_sample()
    candidates = TilingGeneral.get_candidates(tree)
    assert candidates == {
        ("comp01", 0): [
            [
                ("comp01", 0),
                ("comp01", 1),
                ("comp03", 1),
                ("comp03", 2),
                ("comp03", 3),
                ("comp04", 3),
            ]
        ]
    }

    t_tree = test_utils.tree_test_sample_imperfect_loops()
    candidates = TilingGeneral.get_candidates(t_tree)
    assert candidates == {
        ("comp01", 0): [
            [
                ("comp01", 0),
                ("comp01", 1),
                ("comp04", 1),
                ("comp02", 2),
                ("comp05", 2),
                ("comp03", 3),
            ],
            [("comp01", 1), ("comp02", 2)],
            [("comp02", 2), ("comp03", 3)],
            [
                ("comp01", 1),
                ("comp02", 2),
                ("comp03", 3),
            ],
            [("comp04", 1), ("comp05", 2)],
        ]
    }
