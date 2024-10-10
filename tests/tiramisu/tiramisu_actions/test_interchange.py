from tiralib.tiramisu.schedule import Schedule
from tiralib.tiramisu.tiramisu_actions.interchange import Interchange

from tiralib.config import BaseConfig
from tests.utils import interchange_example


def test_interchange_init():
    BaseConfig.init()
    interchange = Interchange([("comp00", 0), ("comp00", 1)])
    assert interchange.params == [("comp00", 0), ("comp00", 1)]
    assert interchange.comps is None

    interchange = Interchange([("comp00", 0), ("comp00", 1)], ["comp00"])
    assert interchange.params == [("comp00", 0), ("comp00", 1)]
    assert interchange.comps == ["comp00"]


def test_initialize_action_for_tree():
    BaseConfig.init()
    sample = interchange_example()
    interchange = Interchange([("comp00", 0), ("comp00", 1)])
    interchange.initialize_action_for_tree(sample.tree)
    assert interchange.params == [("comp00", 0), ("comp00", 1)]
    assert interchange.comps == ["comp00"]


def test_set_string_representations():
    BaseConfig.init()
    sample = interchange_example()
    interchange = Interchange([("comp00", 0), ("comp00", 1)])
    schedule = Schedule(sample)
    schedule.add_optimizations([interchange])
    assert interchange.tiramisu_optim_str == "comp00.interchange(0,1);\n"


def test_get_candidates():
    BaseConfig.init()
    sample = interchange_example()
    candidates = Interchange.get_candidates(sample.tree)
    assert candidates == {
        sample.tree.iterators["i0"].id: [
            (sample.tree.iterators["i0"].id, sample.tree.iterators["i1"].id),
            (sample.tree.iterators["i0"].id, sample.tree.iterators["i2"].id),
            (sample.tree.iterators["i1"].id, sample.tree.iterators["i2"].id),
        ]
    }


def test_legality_check():
    BaseConfig.init()
    sample = interchange_example()
    schedule = Schedule(sample)
    assert schedule.tree
    schedule.add_optimizations([Interchange([("comp00", 0), ("comp00", 1)])])
    legality_string = schedule.optims_list[0].legality_check_string
    assert legality_string == "comp00.interchange(0,1);\n"
