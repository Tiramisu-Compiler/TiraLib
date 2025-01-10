from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, List

from tiralib.tiramisu.tiramisu_tree import TiramisuTree

if TYPE_CHECKING:
    from tiralib.tiramisu.tiramisu_tree import TiramisuTree

from tiralib.tiramisu.tiramisu_actions.tiramisu_action import (
    TiramisuAction,
    TiramisuActionType,
)


class MatrixTransform(TiramisuAction):
    """
    Matrix optimization command.
    """

    def __init__(self, params: List[int], comps: List[str]):
        # MatrixTransform takes the list of parameters of the polyhedral transformation matrix
        # assert that len(params) is a square number (square matrix)
        assert math.sqrt(len(params)) == int(math.sqrt(len(params))), (
            "Matrix is not square"
        )
        assert len(comps) == 1, "MatrixTransform can only be applied to one computation"
        rowLength = int(math.sqrt(len(params)))
        matrix = [params[i : i + rowLength] for i in range(0, len(params), rowLength)]

        self.params = params
        self.comps = comps
        self.matrix = matrix
        super().__init__(
            type=TiramisuActionType.MATRIX_TRANSFORM, params=params, comps=comps
        )

    def initialize_action_for_tree(self, tiramisu_tree: TiramisuTree):
        self.tree = copy.deepcopy(tiramisu_tree)

        # if comps are none get them from the tree
        self.set_string_representations(self.tree)

    def set_string_representations(self, tiramisu_tree: TiramisuTree):
        self.tiramisu_optim_str = ""

        row_vects = []
        for row in self.matrix:
            row_vects.append("{" + ",".join([str(p) for p in row]) + "}")
        mat_vect = "{" + ",".join(row_vects) + "}"
        self.tiramisu_optim_str = f"{self.comps[0]}.matrix_transform({mat_vect});"
        self.legality_check_string = self.tiramisu_optim_str

        self.str_representation = f"M({self.params},comps={self.comps})"
