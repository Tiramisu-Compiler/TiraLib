import tests.utils as test_utils
from tiralib.tiramisu.tiramisu_actions.tiling_general import TilingGeneral

from tiralib.utils.config import BaseConfig


def test_tiling_general_init():
    tiling_general = TilingGeneral([("comp00", 0), ("comp00", 1), 32, 32])
    assert tiling_general.iterators == [("comp00", 0), ("comp00", 1)]
    assert tiling_general.tile_sizes == [32, 32]
    assert tiling_general.comps is None

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
        "c3_2": 10,
        "c5": 10,
        "c5_1": 10,
    }
    assert tiling_general.comps == ["R_up_init", "R_up", "A_out"]
    assert tiling_general.tiled_iterator_names == ["c3_2", "c5", "c5_1"]


def test_set_string_representations():
    BaseConfig.init()
    sample = test_utils.gramschmidt_sample()
    tiling_general = TilingGeneral(
        [("R_up_init", 1), ("R_up", 2), ("A_out", 2), 10, 5, 2]
    )
    tiling_general.initialize_action_for_tree(sample.tree)
    assert tiling_general.iterators == [
        ("R_up_init", 1),
        ("R_up", 2),
        ("A_out", 2),
    ]

    assert (
        tiling_general.tiramisu_optim_str
        == "R_up_init.tile(1, 10);\nR_up.tile(1, 2, 10, 5);\nA_out.tile(1, 2, 10, 2);\n"  # noqa: E501
    )

    assert (
        tiling_general.str_representation
        == "TG(L1,L2,L2,10,5,2,comps=['R_up_init', 'R_up', 'A_out'])"
    )


def test_get_candidates():
    BaseConfig.init()
    sample = test_utils.gramschmidt_sample()
    candidates = TilingGeneral.get_candidates(sample.tree)
    assert candidates == {
        sample.tree.iterators["c1"].id: [
            [
                sample.tree.iterators["c1"].id,
                sample.tree.iterators["c3"].id,
                sample.tree.iterators["c3_1"].id,
                sample.tree.iterators["c3_2"].id,
                sample.tree.iterators["c5"].id,
                sample.tree.iterators["c5_1"].id,
            ],
            [
                sample.tree.iterators["c3_2"].id,
                sample.tree.iterators["c5"].id,
                sample.tree.iterators["c5_1"].id,
            ],
        ]
    }

    tree = test_utils.tree_test_sample()
    candidates = TilingGeneral.get_candidates(tree)
    assert candidates == {
        tree.iterators["root"].id: [
            [
                tree.iterators["root"].id,
                tree.iterators["i"].id,
                tree.iterators["j"].id,
                tree.iterators["k"].id,
                tree.iterators["l"].id,
                tree.iterators["m"].id,
            ]
        ]
    }

    t_tree = test_utils.tree_test_sample_imperfect_loops()
    candidates = TilingGeneral.get_candidates(t_tree)
    assert candidates == {
        t_tree.iterators["root"].id: [
            [
                t_tree.iterators["root"].id,
                t_tree.iterators["i"].id,
                t_tree.iterators["i_1"].id,
                t_tree.iterators["j"].id,
                t_tree.iterators["j_1"].id,
                t_tree.iterators["k"].id,
            ],
            [t_tree.iterators["i"].id, t_tree.iterators["j"].id],
            [t_tree.iterators["j"].id, t_tree.iterators["k"].id],
            [
                t_tree.iterators["i"].id,
                t_tree.iterators["j"].id,
                t_tree.iterators["k"].id,
            ],
            [t_tree.iterators["i_1"].id, t_tree.iterators["j_1"].id],
        ]
    }
