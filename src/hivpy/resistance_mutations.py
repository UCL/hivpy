from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .population import Population

import hivpy.column_names as col


class ResistanceMutationsModule:

    def __init__(self):
        ...

    def init_resistance_variables(self, pop: Population):
        # FIXME: move drugs to ART module
        pop.init_variable(col.ART_ADHERENCE, 0)
        self.init_resistance_mutations(pop)

    def init_resistance_mutations(self, pop: Population):
        """
        Initialise drug resistance mutations at the start of the simulation to False.
        """
        pop.init_variable(col.TA_MUTATION, False)
        pop.init_variable(col.M184_MUTATION, False)
        pop.init_variable(col.K65_MUTATION, False)
        pop.init_variable(col.Q151_MUTATION, False)
        pop.init_variable(col.K103_MUTATION, False)
        pop.init_variable(col.Y181_MUTATION, False)
        pop.init_variable(col.G190_MUTATION, False)
        pop.init_variable(col.P32_MUTATION, False)
        pop.init_variable(col.P33_MUTATION, False)
        pop.init_variable(col.P46_MUTATION, False)
        pop.init_variable(col.P47_MUTATION, False)
        pop.init_variable(col.P50L_MUTATION, False)
        pop.init_variable(col.P50V_MUTATION, False)
        pop.init_variable(col.P54_MUTATION, False)
        pop.init_variable(col.P76_MUTATION, False)
        pop.init_variable(col.P82_MUTATION, False)
        pop.init_variable(col.P84_MUTATION, False)
        pop.init_variable(col.P88_MUTATION, False)
        pop.init_variable(col.P90_MUTATION, False)
        pop.init_variable(col.IN118_MUTATION, False)
        pop.init_variable(col.IN140_MUTATION, False)
        pop.init_variable(col.IN148_MUTATION, False)
        pop.init_variable(col.IN155_MUTATION, False)
        pop.init_variable(col.IN263_MUTATION, False)
