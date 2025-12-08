import pytest

import tiralib.tiramisu.tiramisu_actions as tiramisu_actions
from tiralib.tiramisu.tiramisu_actions import TiramisuAction, TiramisuActionType
from tiralib.tiramisu.tiramisu_tree import TiramisuTree


def test_initialize_action_for_tree():
    tiramisu_action = TiramisuAction(
        type=TiramisuActionType.PARALLELIZATION, params=[1, 2, 3], comps=["a", "b", "c"]
    )
    with pytest.raises(NotImplementedError):
        tiramisu_action.initialize_action_for_tree(tiramisu_tree=TiramisuTree())


def test_set_string_representations():
    tiramisu_action = TiramisuAction(
        type=TiramisuActionType.PARALLELIZATION, params=[1, 2, 3], comps=["a", "b", "c"]
    )
    with pytest.raises(NotImplementedError):
        tiramisu_action.set_string_representations(tiramisu_tree=TiramisuTree())


def test_is_interchange():
    t_action = tiramisu_actions.Interchange(
        [("comp00", 0), ("comp00", 1)], comps=["comp00"]
    )

    assert t_action.is_interchange()


def test_is_tiling_2d():
    t_action = tiramisu_actions.Tiling2D([("comp00", 0), ("comp00", 1), 1, 1])

    assert t_action.is_tiling_2d()


def test_is_tiling_3d():
    t_action = tiramisu_actions.Tiling3D(
        [("comp00", 0), ("comp00", 1), ("comp00", 2), 1, 1, 1]
    )

    assert t_action.is_tiling_3d()


def test_is_parallelization():
    t_action = tiramisu_actions.Parallelization([("comp00", 0)])

    assert t_action.is_parallelization()


def test_is_skewing():
    t_action = tiramisu_actions.Skewing([("comp00", 0), ("comp00", 1), 1, 1])
    assert t_action.is_skewing()


def test_is_unrolling():
    t_action = tiramisu_actions.Unrolling([("comp00", 0), 2])
    assert t_action.is_unrolling()


def test_is_fusion():
    t_action = tiramisu_actions.Fusion([("", 1), ("", 1)])
    assert t_action.is_fusion()


def test_is_reversal():
    t_action = tiramisu_actions.Reversal([("", 0)])
    assert t_action.is_reversal()


def test_is_distribution():
    t_action = tiramisu_actions.Distribution([("", 0)])
    assert t_action.is_distribution()


def test_get_candidates():
    with pytest.raises(NotImplementedError):
        TiramisuAction.get_candidates(program_tree=TiramisuTree())


def test_get_types():
    assert len(TiramisuAction.get_types()) == 12


def test_str():
    t_action = tiramisu_actions.Interchange(
        [("comp00", 0), ("comp00", 1)], comps=["comp00"]
    )
    t_action.set_string_representations(TiramisuTree())
    assert str(t_action) == "I(L0,L1,comps=['comp00'])"


def test_repr():
    t_action = tiramisu_actions.Interchange(
        [("comp00", 0), ("comp00", 1)], comps=["comp00"]
    )
    assert (
        repr(t_action)
        == f"Action(type={TiramisuActionType.INTERCHANGE}, params={t_action.params}, comps={t_action.comps})"
    )


def test_eq():
    t_action = tiramisu_actions.Interchange(
        [("comp00", 0), ("comp00", 1)], comps=["comp00"]
    )
    t_action2 = tiramisu_actions.Interchange(
        [("comp00", 0), ("comp00", 1)], comps=["comp00"]
    )
    assert t_action == t_action2

    t_action2 = tiramisu_actions.Interchange(
        [("comp00", 0), ("comp00", 1)], comps=["comp01"]
    )

    assert t_action != t_action2
