import pickle
from typing import Tuple

from tiralib.tiramisu.tiramisu_iterator_node import IteratorNode
from tiralib.tiramisu.tiramisu_program import TiramisuProgram
from tiralib.tiramisu.tiramisu_tree import TiramisuTree


def load_test_data() -> Tuple[dict, dict]:
    with open("examples/test_data.pkl", "rb") as f:
        dataset = pickle.load(f)
    with open("examples/test_data_cpps.pkl", "rb") as f:
        cpps = pickle.load(f)
    return dataset, cpps


def tree_test_sample():
    tiramisu_tree = TiramisuTree()
    tiramisu_tree.add_root(("comp01", 0))
    tiramisu_tree.iterators = {
        ("comp01", 0): IteratorNode(
            name="root",
            parent_iterator=None,
            lower_bound=0,
            upper_bound=256,
            child_iterators=[("comp01", 1), ("comp03", 1)],
            computations_list=[],
            level=0,
        ),
        ("comp01", 1): IteratorNode(
            name="i",
            parent_iterator=("comp01", 0),
            lower_bound=0,
            upper_bound=256,
            child_iterators=[],
            computations_list=["comp01"],
            level=1,
        ),
        ("comp03", 1): IteratorNode(
            name="j",
            parent_iterator=("comp01", 0),
            lower_bound=0,
            upper_bound=256,
            child_iterators=[("comp03", 2)],
            computations_list=[],
            level=1,
        ),
        ("comp03", 2): IteratorNode(
            name="k",
            parent_iterator=("comp03", 1),
            lower_bound=0,
            upper_bound=10,
            child_iterators=[("comp03", 3), ("comp04", 3)],
            computations_list=[],
            level=2,
        ),
        ("comp03", 3): IteratorNode(
            name="l",
            parent_iterator=("comp03", 2),
            lower_bound=0,
            upper_bound=256,
            child_iterators=[],
            computations_list=["comp03"],
            level=3,
        ),
        ("comp04", 3): IteratorNode(
            name="m",
            parent_iterator=("comp03", 2),
            lower_bound=0,
            upper_bound=256,
            child_iterators=[],
            computations_list=["comp04"],
            level=3,
        ),
    }
    tiramisu_tree.computations = [
        "comp01",
        "comp03",
        "comp04",
    ]

    tiramisu_tree.computations_absolute_order = {
        "comp01": 1,
        "comp03": 2,
        "comp04": 3,
    }
    tiramisu_tree.set_iterator_ids()
    return tiramisu_tree


def tree_test_sample_2():
    tiramisu_tree = TiramisuTree()
    tiramisu_tree.add_root(("comp01", 0))
    tiramisu_tree.iterators = {
        ("comp01", 0): IteratorNode(
            name="root",
            parent_iterator=None,
            lower_bound=0,
            upper_bound=256,
            child_iterators=[("comp01", 1), ("comp05", 1)],
            computations_list=[],
            level=0,
        ),
        ("comp01", 1): IteratorNode(
            name="i",
            parent_iterator=("comp01", 0),
            lower_bound=0,
            upper_bound=256,
            child_iterators=[],
            computations_list=["comp01"],
            level=1,
        ),
        ("comp05", 1): IteratorNode(
            name="j",
            parent_iterator=("comp01", 0),
            lower_bound=0,
            upper_bound=256,
            child_iterators=[("comp03", 2)],
            computations_list=["comp05", "comp06", "comp07"],
            level=1,
        ),
        ("comp03", 2): IteratorNode(
            name="k",
            parent_iterator=("comp05", 1),
            lower_bound=0,
            upper_bound=10,
            child_iterators=[("comp03", 3), ("comp04", 3)],
            computations_list=[],
            level=2,
        ),
        ("comp03", 3): IteratorNode(
            name="l",
            parent_iterator=("comp03", 2),
            lower_bound=0,
            upper_bound=256,
            child_iterators=[],
            computations_list=["comp03"],
            level=3,
        ),
        ("comp04", 3): IteratorNode(
            name="m",
            parent_iterator=("comp03", 2),
            lower_bound=0,
            upper_bound=256,
            child_iterators=[],
            computations_list=["comp04"],
            level=3,
        ),
    }
    tiramisu_tree.computations = [
        "comp01",
        "comp03",
        "comp04",
        "comp05",
        "comp06",
        "comp07",
    ]

    tiramisu_tree.computations_absolute_order = {
        "comp01": 1,
        "comp05": 2,
        "comp06": 3,
        "comp07": 4,
        "comp03": 5,
        "comp04": 6,
    }
    tiramisu_tree.set_iterator_ids()
    return tiramisu_tree


def tree_test_sample_3():
    tiramisu_tree = TiramisuTree()
    tiramisu_tree.add_root(("comp01", 0))
    tiramisu_tree.iterators = {
        ("comp01", 0): IteratorNode(
            name="root",
            parent_iterator=None,
            lower_bound=0,
            upper_bound=256,
            child_iterators=[("comp01", 1), ("comp05", 1)],
            computations_list=[],
            level=0,
        ),
        ("comp01", 1): IteratorNode(
            name="i",
            parent_iterator=("comp01", 0),
            lower_bound=0,
            upper_bound=256,
            child_iterators=[],
            computations_list=["comp01"],
            level=1,
        ),
        ("comp05", 1): IteratorNode(
            name="j",
            parent_iterator=("comp01", 0),
            lower_bound=0,
            upper_bound=256,
            child_iterators=[("comp03", 2)],
            computations_list=["comp05", "comp06", "comp07"],
            level=1,
        ),
        ("comp03", 2): IteratorNode(
            name="k",
            parent_iterator=("comp05", 1),
            lower_bound=0,
            upper_bound=10,
            child_iterators=[("comp03", 3)],
            computations_list=[],
            level=2,
        ),
        ("comp03", 3): IteratorNode(
            name="l",
            parent_iterator=("comp03", 2),
            lower_bound=0,
            upper_bound=256,
            child_iterators=[("comp03", 4)],
            computations_list=[],
            level=3,
        ),
        ("comp03", 4): IteratorNode(
            name="m",
            parent_iterator=("comp03", 3),
            lower_bound=0,
            upper_bound=256,
            child_iterators=[],
            computations_list=["comp03", "comp04"],
            level=4,
        ),
    }
    tiramisu_tree.computations = [
        "comp01",
        "comp03",
        "comp04",
        "comp05",
        "comp06",
        "comp07",
    ]

    tiramisu_tree.computations_absolute_order = {
        "comp01": 1,
        "comp05": 2,
        "comp03": 5,
        "comp04": 6,
        "comp06": 3,
        "comp07": 4,
    }
    tiramisu_tree.set_iterator_ids()
    return tiramisu_tree


def tree_test_sample_imperfect_loops():
    tiramisu_tree = TiramisuTree()
    tiramisu_tree.add_root(("comp01", 0))
    tiramisu_tree.iterators = {
        ("comp01", 0): IteratorNode(
            name="root",
            parent_iterator=None,
            lower_bound=0,
            upper_bound=256,
            child_iterators=[("comp01", 1), ("comp04", 1)],
            computations_list=[],
            level=0,
        ),
        ("comp01", 1): IteratorNode(
            name="i",
            parent_iterator=("comp01", 0),
            lower_bound=0,
            upper_bound=256,
            child_iterators=[("comp02", 2)],
            computations_list=["comp01"],
            level=1,
        ),
        ("comp02", 2): IteratorNode(
            name="j",
            parent_iterator=("comp01", 1),
            lower_bound=0,
            upper_bound=256,
            child_iterators=[("comp03", 3)],
            computations_list=["comp02"],
            level=2,
        ),
        ("comp03", 3): IteratorNode(
            name="k",
            parent_iterator=("comp02", 1),
            lower_bound=0,
            upper_bound=10,
            child_iterators=[],
            computations_list=["comp03"],
            level=3,
        ),
        ("comp04", 1): IteratorNode(
            name="i_1",
            parent_iterator=("comp01", 0),
            lower_bound=0,
            upper_bound=256,
            child_iterators=[("comp05", 2)],
            computations_list=["comp04"],
            level=1,
        ),
        ("comp05", 2): IteratorNode(
            name="j_1",
            parent_iterator=("comp04", 1),
            lower_bound=0,
            upper_bound=256,
            child_iterators=[],
            computations_list=["comp05"],
            level=2,
        ),
    }
    tiramisu_tree.computations = [
        "comp01",
        "comp02",
        "comp03",
        "comp04",
        "comp05",
    ]

    tiramisu_tree.computations_absolute_order = {
        "comp01": 1,
        "comp02": 2,
        "comp03": 3,
        "comp04": 4,
        "comp05": 5,
    }

    tiramisu_tree.set_iterator_ids()
    return tiramisu_tree


def benchmark_program_test_sample():
    tiramisu_func = TiramisuProgram.from_file(
        "examples/function_matmul_MEDIUM.cpp",
        # "examples/function_matmul_MEDIUM_wrapper.cpp",
        # "examples/function_matmul_MEDIUM_wrapper.h",
        # "examples/function_blur_MINI_generator.cpp",
        # "examples/function_blur_MINI_wrapper.cpp",
        # "examples/function_blur_MINI_wrapper.h",
        load_isl_ast=True,
        load_tree=True,
    )

    return tiramisu_func


def interchange_example() -> TiramisuProgram:
    test_data, test_cpps = load_test_data()

    tiramisu_func = TiramisuProgram.from_dict(
        name="function837782",
        data=test_data["function837782"],
        original_str=test_cpps["function837782"],
    )
    if tiramisu_func.annotations is None:
        raise ValueError("Annotations not found")

    tiramisu_func.tree = TiramisuTree.from_annotations(tiramisu_func.annotations)

    return tiramisu_func


def skewing_example() -> TiramisuProgram:
    test_data, test_cpps = load_test_data()

    tiramisu_func = TiramisuProgram.from_dict(
        name="function550013",
        data=test_data["function550013"],
        original_str=test_cpps["function550013"],
    )
    if tiramisu_func.annotations is None:
        raise ValueError("Annotations not found")

    tiramisu_func.tree = TiramisuTree.from_annotations(tiramisu_func.annotations)

    return tiramisu_func


def reversal_sample() -> TiramisuProgram:
    test_data, test_cpps = load_test_data()

    tiramisu_func = TiramisuProgram.from_dict(
        name="function824914",
        data=test_data["function824914"],
        original_str=test_cpps["function824914"],
    )
    if tiramisu_func.annotations is None:
        raise ValueError("Annotations not found")

    tiramisu_func.tree = TiramisuTree.from_annotations(tiramisu_func.annotations)

    return tiramisu_func


def unrolling_sample() -> TiramisuProgram:
    test_data, test_cpps = load_test_data()

    tiramisu_func = TiramisuProgram.from_dict(
        name="function552581",
        data=test_data["function552581"],
        original_str=test_cpps["function552581"],
    )
    if tiramisu_func.annotations is None:
        raise ValueError("Annotations not found")

    tiramisu_func.tree = TiramisuTree.from_annotations(tiramisu_func.annotations)

    return tiramisu_func


def tiling_2d_sample() -> TiramisuProgram:
    test_data, test_cpps = load_test_data()

    tiramisu_func = TiramisuProgram.from_dict(
        name="function554520",
        data=test_data["function554520"],
        original_str=test_cpps["function554520"],
    )
    if tiramisu_func.annotations is None:
        raise ValueError("Annotations not found")

    tiramisu_func.tree = TiramisuTree.from_annotations(tiramisu_func.annotations)

    return tiramisu_func


def tiling_3d_sample() -> TiramisuProgram:
    test_data, test_cpps = load_test_data()

    tiramisu_func = TiramisuProgram.from_dict(
        name="function608722",
        data=test_data["function608722"],
        original_str=test_cpps["function608722"],
    )
    if tiramisu_func.annotations is None:
        raise ValueError("Annotations not found")

    tiramisu_func.tree = TiramisuTree.from_annotations(tiramisu_func.annotations)

    return tiramisu_func


def tiling_3d_tree_sample() -> TiramisuTree:
    tiramisu_tree = TiramisuTree()
    tiramisu_tree.add_root(("comp03", 0))
    tiramisu_tree.iterators = {
        ("comp03", 0): IteratorNode(
            name="root",
            parent_iterator=None,
            lower_bound=0,
            upper_bound=10,
            child_iterators=[("comp03", 1)],
            computations_list=[],
            level=0,
        ),
        ("comp03", 1): IteratorNode(
            name="j",
            parent_iterator=("comp03", 0),
            lower_bound=0,
            upper_bound=10,
            child_iterators=[("comp03", 2)],
            computations_list=[],
            level=1,
        ),
        ("comp03", 2): IteratorNode(
            name="k",
            parent_iterator=("comp03", 1),
            lower_bound=0,
            upper_bound=10,
            child_iterators=[("comp03", 3)],
            computations_list=[],
            level=2,
        ),
        ("comp03", 3): IteratorNode(
            name="l",
            parent_iterator=("comp03", 2),
            lower_bound=0,
            upper_bound=10,
            child_iterators=[],
            computations_list=["comp03"],
            level=3,
        ),
    }
    tiramisu_tree.computations = [
        "comp03",
    ]
    tiramisu_tree.computations_absolute_order = {
        "comp03": 1,
    }

    tiramisu_tree.set_iterator_ids()
    return tiramisu_tree


def fusion_sample():
    tiramisu_prog = TiramisuProgram()

    tiramisu_tree = TiramisuTree()
    tiramisu_tree.add_root(("comp01", 0))
    tiramisu_tree.iterators = {
        ("comp01", 0): IteratorNode(
            name="root",
            parent_iterator=None,
            lower_bound=0,
            upper_bound=10,
            child_iterators=[("comp01", 1), ("comp03", 1)],
            computations_list=[],
            level=0,
        ),
        ("comp01", 1): IteratorNode(
            name="i",
            parent_iterator=("comp01", 0),
            lower_bound=0,
            upper_bound=10,
            child_iterators=[],
            computations_list=["comp01"],
            level=1,
        ),
        ("comp03", 1): IteratorNode(
            name="j",
            parent_iterator=("comp01", 0),
            lower_bound=0,
            upper_bound=10,
            child_iterators=[("comp03", 2)],
            computations_list=[],
            level=1,
        ),
        ("comp03", 2): IteratorNode(
            name="k",
            parent_iterator=("comp03", 1),
            lower_bound=0,
            upper_bound=10,
            child_iterators=[("comp03", 3), ("comp04", 3)],
            computations_list=[],
            level=2,
        ),
        ("comp03", 3): IteratorNode(
            name="l",
            parent_iterator=("comp03", 2),
            lower_bound=0,
            upper_bound=10,
            child_iterators=[],
            computations_list=["comp03"],
            level=3,
        ),
        ("comp04", 3): IteratorNode(
            name="m",
            parent_iterator=("comp03", 2),
            lower_bound=0,
            upper_bound=10,
            child_iterators=[],
            computations_list=["comp04"],
            level=3,
        ),
    }
    tiramisu_tree.computations = [
        "comp01",
        "comp03",
        "comp04",
    ]

    tiramisu_tree.computations_absolute_order = {
        "comp01": 1,
        "comp03": 2,
        "comp04": 3,
    }
    tiramisu_tree.set_iterator_ids()
    tiramisu_prog.tree = tiramisu_tree
    return tiramisu_prog


def multiple_roots_sample():
    tiramisu_func = TiramisuProgram.from_file(
        "examples/function_gemver_MINI_generator.cpp",
        load_isl_ast=True,
        load_tree=True,
    )
    return tiramisu_func


def gramschmidt_sample():
    tiramisu_func = TiramisuProgram.from_file(
        "examples/function_gramschmidt_MINI_generator.cpp",
        load_isl_ast=True,
        load_tree=True,
    )
    return tiramisu_func


def expansion_sample():
    tiramisu_func = TiramisuProgram.from_file(
        "./examples/function_test_expansion.cpp",
        load_isl_ast=True,
        load_tree=True,
    )

    return tiramisu_func
