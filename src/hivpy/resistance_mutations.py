from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .population import Population

import numpy as np

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
        self.new_mutation_matrix = [[[0.05, 0.50, 0.50],
                                     [0.05, [0.50, 0.50, 0.50], [0.50, 0.50, 0.50]],
                                     [0.05, 0.50, 0.50]],   # active drugs == 0.00
                                    [[0.05, 0.50, 0.50],
                                     [0.05, [0.50, 0.50, 0.50], [0.50, 0.50, 0.50]],
                                     [0.05, 0.50, 0.50]],   # active drugs == 0.25
                                    [[0.05, 0.50, 0.50],
                                     [0.05, [0.50, 0.50, 0.50], [0.50, 0.50, 0.50]],
                                     [0.05, 0.50, 0.50]],   # active drugs == 0.50
                                    [[0.05, 0.45, 0.45],
                                     [0.05, [0.45, 0.45, 0.45], [0.45, 0.45, 0.45]],
                                     [0.05, 0.45, 0.45]],   # active drugs == 0.75
                                    [[0.05, 0.40, 0.40],
                                     [0.05, [0.40, 0.40, 0.40], [0.40, 0.40, 0.40]],
                                     [0.05, 0.40, 0.40]],   # active drugs == 1.00
                                    [[0.05, 0.35, 0.30],
                                     [0.05, [0.35, 0.35, 0.35], [0.30, 0.30, 0.30]],
                                     [0.05, 0.35, 0.30]],   # active drugs == 1.25
                                    [[0.05, 0.30, 0.20],
                                     [0.05, [0.30, 0.30, 0.30], [0.20, 0.20, 0.20]],
                                     [0.05, 0.30, 0.20]],   # active drugs == 1.50
                                    [[0.05, 0.30, 0.15],
                                     [0.05, [0.30, 0.30, 0.30], [0.10, 0.10, 0.10]],
                                     [0.05, 0.30, 0.15]],   # active drugs == 1.75
                                    [[0.05, 0.30, 0.10],
                                     [0.05, [0.30, 0.30, 0.30], [0.05, 0.05, 0.05]],
                                     [0.05, 0.30, 0.10]],   # active drugs == 2.00
                                    [[0.05, 0.25, 0.05],
                                     [0.05, [0.20, 0.20, 0.20], [0.05, 0.05, 0.05]],
                                     [0.05, 0.25, 0.08]],   # active drugs == 2.25
                                    [[0.05, 0.20, 0.03],
                                     [0.05, [0.20, 0.20, 0.20], [0.03, 0.03, 0.03]],
                                     [0.05, 0.20, 0.03]],   # active drugs == 2.50
                                    [[0.05, 0.15, 0.01],
                                     [0.05, [0.15, 0.15, 0.15], [0.05, 0.01, 0.01]],
                                     [0.05, 0.18, 0.01]],   # active drugs == 2.27
                                    [[0.05, 0.15, 0.002],
                                     [0.05, [0.10, 0.10, 0.10], [0.05, 0.002, 0.002]],
                                     [0.05, 0.15, 0.002]]]  # active drugs >= 3.00

        # cd4_delta_matrix[active_drugs][cont_on_art][adherence]
        self.cd4_delta_matrix = [[[-18, -17, -15],
                                  [-18, [-17, -17, -17], [-15, -15, -15]],
                                  [-18, -17, -15]],     # active drugs == 0.00
                                 [[-13, -7, -2],
                                  [-18, [-17.5, -17.5, -15], [-16, -14.5, -14]],
                                  [-17, -15, -12]],     # active drugs == 0.25
                                 [[-12, 0, 5],
                                  [-18, [-16, -16, -14], [-13, -13, -12]],
                                  [-17, -15, -12]],     # active drugs == 0.50
                                 [[-11, 3, 10],
                                  [-18, [-16, -14.5, -12.5], [-11, -11, -10.5]],
                                  [-17, -14, -10.5]],   # active drugs == 0.75
                                 [[-10, 5, 13],
                                  [-18, [-13, -13, -11], [-9, -9, -9]],
                                  [-18, -13, -9]],      # active drugs == 1.00
                                 [[-6, 8, 17],
                                  [-17.5, [-11.5, -11.5, -9], [-7.5, -7, -5]],
                                  [-17, -12, -5]],      # active drugs == 1.25
                                 [[-3, 10, 20],
                                  [-17, [-10, -10, 0], [-4.5, -4.5, 3]],
                                  [-17, -10, 3]],       # active drugs == 1.50
                                 [[-1, 13, 25],
                                  [-16.5, [-6, -6, 4], [1.5, 1.5, 19]],
                                  [-16.5, -7.5, 19]],   # active drugs == 1.75
                                 [[1, 15, 30],
                                  [-16, [-4.5, -4.5, 7], [7.5, 7.5, 21]],
                                  [-16, -4.5, 21]],     # active drugs == 2.00
                                 [[2, 20, 35],
                                  [-15.5, [-2, 8, 8], [23, 23, 23]],
                                  [-15.5, 8, 23]],      # active drugs == 2.25
                                 [[3, 23, 40],
                                  [-15, [0, 10, 10], [25, 25, 25]],
                                  [-15, 10, 25]],       # active drugs == 2.50
                                 [[4, 30, 80],
                                  [-14, [4.5, 13, 13], [28, 28, 28]],
                                  [-14, 13, 28]],       # active drugs == 2.27
                                 [[5, 30, 180],
                                  [-13, [7.5, 15, 15], [30, 30, 30]],
                                  [-13, 15, 30]]]       # active drugs >= 3.00

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

    def calc_prob_new_mutation(self, active_drugs, cont_on_art, adherence, adherence_tm1,
                               viral_load, viral_load_tm1, on_nev, on_efa):
        """
        Returns the probability of acquiring a new HIV mutation this time step.
        Affected by number of active ART drugs, how long an individual has been on ART,
        as well as their ART adherence and viral load.
        """
        # find matrix indices
        active_drug_index = np.digitize(active_drugs, self.active_drug_bins)
        cont_on_art_index = np.digitize(cont_on_art, self.cont_on_art_bins)
        adherence_index = np.digitize(adherence, self.adherence_bins)
        # adjust adherence index if taking specific ART drugs
        if adherence_index == 0 and (on_nev or on_efa):
            adherence_index += 1

        # lookup new mutation probability multiplier
        x = self.new_mutation_matrix[active_drug_index][cont_on_art_index][adherence_index]
        # account for adherence last time step when on ART for 3-6 months
        # FIXME: the way this is done should change once CD4 and viral load calculations are added
        if cont_on_art_index == 1 and adherence_index == 2:
            adherence_tm1_index = np.digitize(adherence_tm1, self.adherence_bins)
            x = x[adherence_tm1_index]

        prob_new_mutation = x * (viral_load + viral_load_tm1)/2

        return prob_new_mutation
