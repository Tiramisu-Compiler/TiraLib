import tests.utils as test_utils
from tiralib.tiramisu.tiramisu_tree import TiramisuTree
from tiralib.config import BaseConfig


def test_from_annotations():
    data, _ = test_utils.load_test_data()
    # get program of first key from data
    program = data[list(data.keys())[0]]
    tiramisu_tree = TiramisuTree.from_annotations(program["program_annotation"])
    assert len(tiramisu_tree.roots) == 1

    BaseConfig.init()
    multi_roots = test_utils.multiple_roots_sample().tree

    assert len(multi_roots.roots) == 4
    assert len(multi_roots.iterators) == 7
    assert len(multi_roots.computations) == 4

    assert multi_roots.computations_absolute_order == {
        "A_hat": 1,
        "x_temp": 2,
        "x": 3,
        "w": 4,
    }


def test_get_candidate_sections():
    t_tree = test_utils.tree_test_sample()

    candidate_sections = t_tree.get_candidate_sections()

    assert len(candidate_sections) == 1
    root_id = ("comp01", 0)
    assert len(candidate_sections[root_id]) == 5
    assert candidate_sections[root_id][0] == [root_id]
    assert candidate_sections[root_id][1] == [("comp01", 1)]
    assert candidate_sections[root_id][2] == [
        ("comp03", 1),
        ("comp03", 2),
    ]
    assert candidate_sections[root_id][3] == [("comp03", 3)]
    assert candidate_sections[root_id][4] == [("comp04", 3)]


def test_get_candidate_computations():
    t_tree = test_utils.tree_test_sample()

    assert t_tree.get_iterator_subtree_computations(("comp01", 0)) == [
        "comp01",
        "comp03",
        "comp04",
    ]
    assert t_tree.get_iterator_subtree_computations(("comp01", 1)) == ["comp01"]
    assert t_tree.get_iterator_subtree_computations(("comp03", 1)) == [
        "comp03",
        "comp04",
    ]


def test_get_root_of_node():
    t_tree = test_utils.tree_test_sample()

    assert t_tree.get_root_of_node(("comp01", 1)) == ("comp01", 0)
    assert t_tree.get_root_of_node(("comp03", 1)) == ("comp01", 0)
    assert t_tree.get_root_of_node(("comp04", 3)) == ("comp01", 0)


def test_get_iterator_levels():
    t_tree = test_utils.tree_test_sample()

    assert t_tree.get_iterator_levels(
        [
            ("comp01", 0),
            ("comp01", 1),
            ("comp03", 1),
            ("comp03", 2),
            ("comp03", 3),
            ("comp04", 3),
        ]
    ) == [
        0,
        1,
        1,
        2,
        3,
        3,
    ]


def test_get_iterator_of_computation():
    t_tree = test_utils.tree_test_sample()

    assert t_tree.get_iterator_of_computation("comp01").name == ("comp01", 3)
    assert t_tree.get_iterator_of_computation("comp03").name == ("comp03", 3)
    assert t_tree.get_iterator_of_computation("comp04").name == ("comp04", 3)

    assert t_tree.get_iterator_of_computation("comp01", level=0).name == ("comp01", 0)
    assert t_tree.get_iterator_of_computation("comp03", level=1).name == ("comp03", 1)
