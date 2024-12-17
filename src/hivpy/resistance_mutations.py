from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .population import Population

import hivpy.column_names as col

from .common import timedelta


class ResistanceMutationsModule:

    def __init__(self):
        # factors affecting acquisition of new mutations
        self.active_drug_bins = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
        self.cont_on_art_bins = [timedelta(months=3).years(),
                                 timedelta(months=6).years()]
        self.adherence_bins = [0.5, 0.8]

        # new_mutation_matrix[active_drugs][cont_on_art][adherence]
        self.new_mutation_matrix = [[[0.05, 0.50, 0.50],  [0.05, 0.50, [0.50, 0.50]],  [0.05, 0.50, 0.50]],
                                    [[0.05, 0.50, 0.50],  [0.05, 0.50, [0.50, 0.50]],  [0.05, 0.50, 0.50]],
                                    [[0.05, 0.50, 0.50],  [0.05, 0.50, [0.50, 0.50]],  [0.05, 0.50, 0.50]],
                                    [[0.05, 0.45, 0.45],  [0.05, 0.45, [0.45, 0.45]],  [0.05, 0.45, 0.45]],
                                    [[0.05, 0.40, 0.40],  [0.05, 0.40, [0.40, 0.40]],  [0.05, 0.40, 0.40]],
                                    [[0.05, 0.35, 0.30],  [0.05, 0.35, [0.30, 0.30]],  [0.05, 0.35, 0.30]],
                                    [[0.05, 0.30, 0.20],  [0.05, 0.30, [0.20, 0.20]],  [0.05, 0.30, 0.20]],
                                    [[0.05, 0.30, 0.15],  [0.05, 0.30, [0.10, 0.10]],  [0.05, 0.30, 0.15]],
                                    [[0.05, 0.30, 0.10],  [0.05, 0.30, [0.05, 0.05]],  [0.05, 0.30, 0.10]],
                                    [[0.05, 0.25, 0.05],  [0.05, 0.20, [0.05, 0.05]],  [0.05, 0.25, 0.08]],
                                    [[0.05, 0.20, 0.03],  [0.05, 0.20, [0.03, 0.03]],  [0.05, 0.20, 0.03]],
                                    [[0.05, 0.15, 0.01],  [0.05, 0.15, [0.05, 0.01]],  [0.05, 0.18, 0.01]],
                                    [[0.05, 0.15, 0.002], [0.05, 0.10, [0.05, 0.002]], [0.05, 0.15, 0.002]]]

    def init_resistance_variables(self, pop: Population):
        # FIXME: move drugs to ART module
        pop.init_variable(col.ART_ADHERENCE, 0, n_prev_steps=1)
        pop.init_variable(col.CONT_ON_ART, timedelta(months=0))
        pop.init_variable(col.NUM_ACTIVE_DRUGS, 0)
        pop.init_variable(col.ON_NEV, False)
        pop.init_variable(col.ON_EFA, False)
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
