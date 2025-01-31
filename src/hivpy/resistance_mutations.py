from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .population import Population

import operator as op

import numpy as np

import hivpy.column_names as col

from .common import COND, SexType, rng, timedelta


class ResistanceMutationsModule:

    def __init__(self):
        # matrix indexing boundaries
        self.active_drug_bins = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
        self.cont_on_art_bins = [timedelta(months=3).years(), timedelta(months=6).years()]
        self.adherence_bins = [0.5, 0.8]

        # factors affecting change in viral load
        self.min_vl_on_art = 1.0
        self.vl_stdev_on_art = 0.5

        # factors affecting change in cd4 count
        self.hindered_cd4_recovery = round(-6 + (3 * rng.normal()))  # FIXME: dependent on time step length
        self.failed_insti_hinders_cd4_recovery = rng.choice([True, False])
        self.cd4_recovery_pi_factor = 3
        self.cd4_recovery_female_factor = 2
        self.cd4_stdev_on_art = 1.2  # on a sqrt scale

        # factors affecting acquisition of new mutations
        self.mutation_risk_change = rng.choice([0.5, 1, 2], p=[0.1, 0.8, 0.1])

        # viral_load_matrix[active_drugs][cont_on_art][adherence]
        # (a, b, c) tuples used to calculate a * max_viral_load + b + c * min_vl_on_art
        self.viral_load_matrix = [[[(1, 0, 0), (1, -0.05, 0), (1, -0.2, 0)],
                                   [[(1, 0, 0), (1, -0.05, 0), (1, -0.2, 0)],
                                    [(1, 0, 0), (1, -0.05, 0), (1, -0.2, 0)],
                                    [(1, 0, 0), (1, -0.05, 0), (1, -0.2, 0)]],
                                   [(1, 0, 0), (1, -0.05, 0), (1, -0.2, 0)]],      # active drugs == 0.00
                                  [[(1, 0.1, 0), (1, -0.05, 0), (1, -0.3, 0)],
                                   [[(1, 0, 0), (1, 0.1, 0), (1, 0.1, 0)],
                                    [(1, -0.05, 0), (1, -0.05, 0), (1, -0.3, 0)],
                                    [(1, -0.2, 0), (1, -0.35, 0), (1, -0.4, 0)]],
                                   [(1, 0, 0), (1, -0.1, 0), (1, -0.3, 0)]],       # active drugs == 0.25
                                  [[(1, 0.1, 0), (1, -0.1, 0), (1, -0.4, 0)],
                                   [[(1, 0, 0), (1, 0.1, 0), (1, 0.1, 0)],
                                    [(1, -0.2, 0), (1, -0.2, 0), (1, -0.4, 0)],
                                    [(1, -0.5, 0), (1, -0.5, 0), (1, -0.6, 0)]],
                                   [(1, -0.1, 0), (1, -0.3, 0), (1, -0.6, 0)]],    # active drugs == 0.50
                                  [[(1, 0.1, 0), (1, -0.25, 0), (1, -0.55, 0)],
                                   [[(1, 0, 0), (1, 0.1, 0), (1, 0.1, 0)],
                                    [(1, -0.2, 0), (1, -0.35, 0), (1, -0.55, 0)],
                                    [(1, -0.7, 0), (1, -0.7, 0), (1, -0.75, 0)]],
                                   [(1, -0.1, 0), (1, -0.4, 0), (1, -0.75, 0)]],   # active drugs == 0.75
                                  [[(1, 0.1, 0), (1, -0.4, 0), (1, -0.7, 0)],
                                   [[(1, 0, 0), (1, 0.1, 0), (1, 0.1, 0)],
                                    [(1, -0.5, 0), (1, -0.5, 0), (1, -0.7, 0)],
                                    [(1, -0.9, 0), (1, -0.9, 0), (1, -0.9, 0)]],
                                   [(1, -0.1, 0), (1, -0.5, 0), (1, -0.9, 0)]],    # active drugs == 1.00
                                  [[(1, -0.05, 0), (1, -0.5, 0), (1, -0.8, 0)],
                                   [[(1, -0.05, 0), (1, -0.05, 0), (1, -0.05, 0)],
                                    [(1, -0.65, 0), (1, -0.65, 0), (1, -0.9, 0)],
                                    [(1, -1, 0), (1, -1.05, 0), (1, -1.15, 0)]],
                                   [(1, -0.1, 0), (1, -0.6, 0), (1, -1.15, 0)]],   # active drugs == 1.25
                                  [[(1, 0, 0), (1, -0.6, 0), (1, -0.9, 0)],
                                   [[(1, -0.1, 0), (1, -0.1, 0), (1, -0.1, 0)],
                                    [(1, -0.8, 0), (1, -0.8, 0), (1, -1.5, 0)],
                                    [(1, -1.2, 0), (1, -1.2, 0), (1, -1.7, 0)]],
                                   [(1, -0.1, 0), (1, -0.7, 0), (1, -1.4, 0)]],    # active drugs == 1.50
                                  [[(1, -0.15, 0), (1, -0.8, 0), (1, -1.25, 0)],
                                   [[(1, -0.15, 0), (1, -0.15, 0), (1, -0.15, 0)],
                                    [(1, -1.1, 0), (1, -1.1, 0), (1, -2.4, 0)],
                                    [(1, -1.6, 0), (1, -1.6, 0), (0, 2.7, 0)]],
                                   [(1, -0.2, 0), (1, -1, 0), (1, -2, 0)]],        # active drugs == 1.75
                                  [[(1, -0.2, 0), (1, -0.9, 0), (1, -1.5, 0)],
                                   [[(1, -0.2, 0), (1, -0.2, 0), (1, -0.2, 0)],
                                    [(1, -1.2, 0), (1, -1.2, 0), (0, 2.4, 0)],
                                    [(1, -2, 0), (1, -2, 0), (0, 2, 0)]],
                                   [(1, -0.2, 0), (1, -1.2, 0), (1, -2.5, 0)]],    # active drugs == 2.00
                                  [[(1, -0.25, 0), (1, -1.1, 0), (1, -1.8, 0)],
                                   [[(1, -0.25, 0), (1, -0.25, 0), (1, -0.25, 0)],
                                    [(1, -1.35, 0), (0, 2.5, 0), (0, 2.2, 0)],
                                    [(0, 1.4, 0), (0, 1.4, 0), (0, 1.4, 0)]],
                                   [(1, -0.25, 0), (0, 1.6, 0), (0, 1.4, 0)]],     # active drugs == 2.25
                                  [[(1, -0.3, 0), (1, -1.2, 0), (1, -2.2, 0)],
                                   [[(1, -0.3, 0), (1, -0.3, 0), (1, -0.3, 0)],
                                    [(1, -1.5, 0), (0, 2.5, 0), (0, 1.8, 0)],
                                    [(0, 1.2, 0), (0, 1.2, 0), (0, 1.2, 0)]],
                                   [(1, -0.3, 0), (0, 1.2, 0), (0, 0, 1)]],        # active drugs == 2.50
                                  [[(1, -0.4, 0), (1, -1.6, 0), (1, -2.6, 0)],
                                   [[(1, -0.4, 0), (1, -0.4, 0), (1, -0.4, 0)],
                                    [(1, -1.8, 0), (0, 2.5, 0), (0, 1.6, 0)],
                                    [(0, 1.2, 0), (0, 1.2, 0), (0, 0, 1)]],
                                   [(1, -0.4, 0), (0, 1.2, 0), (0, 0, 1)]],        # active drugs == 2.75
                                  [[(1, -0.5, 0), (1, -2, 0), (1, -3, 0)],
                                   [[(1, -0.5, 0), (1, -0.5, 0), (1, -0.5, 0)],
                                    [(1, -2, 0), (0, 2.5, 0), (0, 1.2, 0)],
                                    [(0, 1.2, 0), (0, 1.2, 0), (0, 0, 1)]],
                                   [(1, -0.5, 0), (0, 1.2, 0), (0, 0, 1)]]]        # active drugs >= 3.00

        # cd4_delta_matrix[active_drugs][cont_on_art][adherence]
        self.cd4_delta_matrix = [[[-18, -17, -15],
                                  [[-18, -18, -18], [-17, -17, -17], [-15, -15, -15]],
                                  [-18, -17, -15]],     # active drugs == 0.00
                                 [[-13, -7, -2],
                                  [[-18, -18, -18], [-17.5, -17.5, -15], [-16, -14.5, -14]],
                                  [-17, -15, -12]],     # active drugs == 0.25
                                 [[-12, 0, 5],
                                  [[-18, -18, -18], [-16, -16, -14], [-13, -13, -12]],
                                  [-17, -15, -12]],     # active drugs == 0.50
                                 [[-11, 3, 10],
                                  [[-18, -18, -18], [-16, -14.5, -12.5], [-11, -11, -10.5]],
                                  [-17, -14, -10.5]],   # active drugs == 0.75
                                 [[-10, 5, 13],
                                  [[-18, -18, -18], [-13, -13, -11], [-9, -9, -9]],
                                  [-18, -13, -9]],      # active drugs == 1.00
                                 [[-6, 8, 17],
                                  [[-17.5, -17.5, -17.5], [-11.5, -11.5, -9], [-7.5, -7, -5]],
                                  [-17, -12, -5]],      # active drugs == 1.25
                                 [[-3, 10, 20],
                                  [[-17, -17, -17], [-10, -10, 0], [-4.5, -4.5, 3]],
                                  [-17, -10, 3]],       # active drugs == 1.50
                                 [[-1, 13, 25],
                                  [[-16.5, -16.5, -16.5], [-6, -6, 4], [1.5, 1.5, 19]],
                                  [-16.5, -7.5, 19]],   # active drugs == 1.75
                                 [[1, 15, 30],
                                  [[-16, -16, -16], [-4.5, -4.5, 7], [7.5, 7.5, 21]],
                                  [-16, -4.5, 21]],     # active drugs == 2.00
                                 [[2, 20, 35],
                                  [[-15.5, -15.5, -15.5], [-2, 8, 8], [23, 23, 23]],
                                  [-15.5, 8, 23]],      # active drugs == 2.25
                                 [[3, 23, 40],
                                  [[-15, -15, -15], [0, 10, 10], [25, 25, 25]],
                                  [-15, 10, 25]],       # active drugs == 2.50
                                 [[4, 30, 80],
                                  [[-14, -14, -14], [4.5, 13, 13], [28, 28, 28]],
                                  [-14, 13, 28]],       # active drugs == 2.75
                                 [[5, 30, 180],
                                  [[-13, -13, -13], [7.5, 15, 15], [30, 30, 30]],
                                  [-13, 15, 30]]]       # active drugs >= 3.00

        # new_mutation_matrix[active_drugs][cont_on_art][adherence]
        self.new_mutation_matrix = [[[0.05, 0.50, 0.50],
                                     [[0.05, 0.05, 0.05], [0.50, 0.50, 0.50], [0.50, 0.50, 0.50]],
                                     [0.05, 0.50, 0.50]],   # active drugs == 0.00
                                    [[0.05, 0.50, 0.50],
                                     [[0.05, 0.05, 0.05], [0.50, 0.50, 0.50], [0.50, 0.50, 0.50]],
                                     [0.05, 0.50, 0.50]],   # active drugs == 0.25
                                    [[0.05, 0.50, 0.50],
                                     [[0.05, 0.05, 0.05], [0.50, 0.50, 0.50], [0.50, 0.50, 0.50]],
                                     [0.05, 0.50, 0.50]],   # active drugs == 0.50
                                    [[0.05, 0.45, 0.45],
                                     [[0.05, 0.05, 0.05], [0.45, 0.45, 0.45], [0.45, 0.45, 0.45]],
                                     [0.05, 0.45, 0.45]],   # active drugs == 0.75
                                    [[0.05, 0.40, 0.40],
                                     [[0.05, 0.05, 0.05], [0.40, 0.40, 0.40], [0.40, 0.40, 0.40]],
                                     [0.05, 0.40, 0.40]],   # active drugs == 1.00
                                    [[0.05, 0.35, 0.30],
                                     [[0.05, 0.05, 0.05], [0.35, 0.35, 0.35], [0.30, 0.30, 0.30]],
                                     [0.05, 0.35, 0.30]],   # active drugs == 1.25
                                    [[0.05, 0.30, 0.20],
                                     [[0.05, 0.05, 0.05], [0.30, 0.30, 0.30], [0.20, 0.20, 0.20]],
                                     [0.05, 0.30, 0.20]],   # active drugs == 1.50
                                    [[0.05, 0.30, 0.15],
                                     [[0.05, 0.05, 0.05], [0.30, 0.30, 0.30], [0.10, 0.10, 0.10]],
                                     [0.05, 0.30, 0.15]],   # active drugs == 1.75
                                    [[0.05, 0.30, 0.10],
                                     [[0.05, 0.05, 0.05], [0.30, 0.30, 0.30], [0.05, 0.05, 0.05]],
                                     [0.05, 0.30, 0.10]],   # active drugs == 2.00
                                    [[0.05, 0.25, 0.05],
                                     [[0.05, 0.05, 0.05], [0.20, 0.20, 0.20], [0.05, 0.05, 0.05]],
                                     [0.05, 0.25, 0.08]],   # active drugs == 2.25
                                    [[0.05, 0.20, 0.03],
                                     [[0.05, 0.05, 0.05], [0.20, 0.20, 0.20], [0.03, 0.03, 0.03]],
                                     [0.05, 0.20, 0.03]],   # active drugs == 2.50
                                    [[0.05, 0.15, 0.01],
                                     [[0.05, 0.05, 0.05], [0.15, 0.15, 0.15], [0.05, 0.01, 0.01]],
                                     [0.05, 0.18, 0.01]],   # active drugs == 2.75
                                    [[0.05, 0.15, 0.002],
                                     [[0.05, 0.05, 0.05], [0.10, 0.10, 0.10], [0.05, 0.002, 0.002]],
                                     [0.05, 0.15, 0.002]]]  # active drugs >= 3.00

    def init_resistance_variables(self, pop: Population):
        # FIXME: move drugs and other ART-related columns to ART module
        pop.init_variable(col.ART_NAIVE, True)
        pop.init_variable(col.ON_ART, False)
        pop.init_variable(col.CONT_ON_ART, timedelta(months=0))
        pop.init_variable(col.CONT_ON_ARV, timedelta(months=0))
        pop.init_variable(col.ART_ADHERENCE, 0, n_prev_steps=1)
        pop.init_variable(col.NUM_ACTIVE_DRUGS, 0)
        self.init_arv_drugs(pop)
        pop.init_variable(col.RESISTANCE_MUTATIONS, 0)
        self.init_resistance_mutations(pop)

    def init_arv_drugs(self, pop: Population):
        """
        Initialise antiretroviral drugs at the start of the simulation to False.
        """
        pop.init_variable(col.ON_ZDV, False)
        pop.init_variable(col.ON_3TC, False)
        pop.init_variable(col.ON_TEN, False)
        pop.init_variable(col.ON_NEV, False)
        pop.init_variable(col.ON_DAR, False)
        pop.init_variable(col.ON_EFA, False)
        pop.init_variable(col.ON_LPR, False)
        pop.init_variable(col.ON_TAZ, False)
        pop.init_variable(col.ON_DOL, False)
        pop.init_variable(col.ON_CAB, False)
        pop.init_variable(col.ON_LEN, False)
        pop.init_variable(col.ON_OLE, False)
        pop.init_variable(col.ON_ISL, False)

    def init_resistance_mutations(self, pop: Population):
        """
        Initialise drug resistance mutations at the start of the simulation to False.
        """
        pop.init_variable(col.RTTA_MUTATIONS, 0)
        pop.init_variable(col.RT184_MUTATION, False)
        pop.init_variable(col.RT65_MUTATION, False)
        pop.init_variable(col.RT151_MUTATION, False)
        pop.init_variable(col.RT103_MUTATION, False)
        pop.init_variable(col.RT181_MUTATION, False)
        pop.init_variable(col.RT190_MUTATION, False)
        pop.init_variable(col.PR32_MUTATION, False)
        pop.init_variable(col.PR33_MUTATION, False)
        pop.init_variable(col.PR46_MUTATION, False)
        pop.init_variable(col.PR47_MUTATION, False)
        pop.init_variable(col.PR50L_MUTATION, False)
        pop.init_variable(col.PR50V_MUTATION, False)
        pop.init_variable(col.PR54_MUTATION, False)
        pop.init_variable(col.PR76_MUTATION, False)
        pop.init_variable(col.PR82_MUTATION, False)
        pop.init_variable(col.PR84_MUTATION, False)
        pop.init_variable(col.PR88_MUTATION, False)
        pop.init_variable(col.PR90_MUTATION, False)
        pop.init_variable(col.IN118_MUTATION, False)
        pop.init_variable(col.IN140_MUTATION, False)
        pop.init_variable(col.IN148_MUTATION, False)
        pop.init_variable(col.IN155_MUTATION, False)
        pop.init_variable(col.IN263_MUTATION, False)

    def get_all_matrix_indices(self, pop, sub_pop):
        """
        Returns all active drug, continuous ART usage, and adherence indices for the HIV+ sub-population.
        These indices are used to look up values in the viral load, CD4 delta, and new mutation matrices.
        """
        # find matrix indices
        active_drug_indices = np.digitize(pop.get_variable(col.NUM_ACTIVE_DRUGS, sub_pop), self.active_drug_bins)
        cont_on_art_indices = np.digitize([x.years() for x in pop.get_variable(col.CONT_ON_ART, sub_pop)], self.cont_on_art_bins)
        adherence_indices = np.digitize(pop.get_variable(col.ART_ADHERENCE, sub_pop), self.adherence_bins)
        adherence_tm1_indices = np.digitize(pop.get_variable(col.ART_ADHERENCE, sub_pop, dt=1), self.adherence_bins)
        # discount adherence last time step when not on ART for 3-6 months
        adherence_tm1_indices = np.where(cont_on_art_indices != 1, -1, adherence_tm1_indices)

        return active_drug_indices, cont_on_art_indices, adherence_indices, adherence_tm1_indices

    def get_individual_matrix_indices(self, i, on_nev=None, on_efa=None):
        """
        Returns the active drug, continuous ART usage, and adherence indices for a specific HIV+ individual.
        """
        # use row index to find matrix indices
        active_drug_index = self.active_drug_indices[i]
        cont_on_art_index = self.cont_on_art_indices[i]
        adherence_index = self.adherence_indices[i]
        adherence_tm1_index = self.adherence_tm1_indices[i]
        # adjust adherence index if taking specific ART drugs (only relevant to new mutation probability)
        if adherence_index == 0 and (on_nev or on_efa):
            adherence_index += 1

        return active_drug_index, cont_on_art_index, adherence_index, adherence_tm1_index

    def get_matrix_value(self, matrix, i, on_nev=None, on_efa=None):
        """
        Returns a value from a given matrix (expecting one of the viral load, CD4 delta, or new mutation matrices)
        for a specific individual.
        """
        active_drug_index, cont_on_art_index, \
            adherence_index, adherence_tm1_index = self.get_individual_matrix_indices(i, on_nev, on_efa)

        return (matrix[active_drug_index][cont_on_art_index][adherence_index][adherence_tm1_index]
                if adherence_tm1_index > -1 else matrix[active_drug_index][cont_on_art_index][adherence_index])

    def get_matrix_val(self, matrix, active_drugs, cont_on_art, adherence, adherence_tm1,
                       on_nev=None, on_efa=None):
        """
        Returns a value from either the new mutation matrix or the CD4 delta matrix given the input parameters.
        """
        # find matrix indices
        active_drug_index = np.digitize(active_drugs, self.active_drug_bins)
        cont_on_art_index = np.digitize(cont_on_art.years(), self.cont_on_art_bins)
        adherence_index = np.digitize(adherence, self.adherence_bins)

        # adjust adherence index if taking specific ART drugs (only relevant to new mutation probability)
        if adherence_index == 0 and (on_nev or on_efa):
            adherence_index += 1

        # lookup matrix value
        x = matrix[active_drug_index][cont_on_art_index][adherence_index]
        # account for adherence last time step when on ART for 3-6 months
        if cont_on_art_index == 1:
            adherence_tm1_index = np.digitize(adherence_tm1, self.adherence_bins)
            x = x[adherence_tm1_index]

        return x

    def viral_load(self, pop: Population, sub_pop):
        """
        Update viral load for HIV+ individuals.
        """
        # get viral load outcomes
        viral_load = pop.apply_function(self.calc_viral_load, 1, sub_pop)
        pop.set_present_variable(col.VIRAL_LOAD, viral_load, sub_pop)

    def get_viral_load_matrix(self, max_viral_load):
        # viral_load_matrix[active_drugs][cont_on_art][adherence]
        # FIXME: is there a better way to do this?
        return [[[max_viral_load, max_viral_load - 0.05, max_viral_load - 0.2],
                 [[max_viral_load, max_viral_load - 0.05, max_viral_load - 0.2],
                  [max_viral_load, max_viral_load - 0.05, max_viral_load - 0.2],
                  [max_viral_load, max_viral_load - 0.05, max_viral_load - 0.2]],
                 [max_viral_load, max_viral_load - 0.05, max_viral_load - 0.2]],        # active drugs == 0.00
                [[max_viral_load + 0.1, max_viral_load - 0.05, max_viral_load - 0.3],
                 [[max_viral_load, max_viral_load + 0.1, max_viral_load + 0.1],
                  [max_viral_load - 0.05, max_viral_load - 0.05, max_viral_load - 0.3],
                  [max_viral_load - 0.2, max_viral_load - 0.35, max_viral_load - 0.4]],
                 [max_viral_load, max_viral_load - 0.1, max_viral_load - 0.3]],         # active drugs == 0.25
                [[max_viral_load + 0.1, max_viral_load - 0.1, max_viral_load - 0.4],
                 [[max_viral_load, max_viral_load + 0.1, max_viral_load + 0.1],
                  [max_viral_load - 0.2, max_viral_load - 0.2, max_viral_load - 0.4],
                  [max_viral_load - 0.5, max_viral_load - 0.5, max_viral_load - 0.6]],
                 [max_viral_load - 0.1, max_viral_load - 0.3, max_viral_load - 0.6]],   # active drugs == 0.50
                [[max_viral_load + 0.1, max_viral_load - 0.25, max_viral_load - 0.55],
                 [[max_viral_load, max_viral_load + 0.1, max_viral_load + 0.1],
                  [max_viral_load - 0.2, max_viral_load - 0.35, max_viral_load - 0.55],
                  [max_viral_load - 0.7, max_viral_load - 0.7, max_viral_load - 0.75]],
                 [max_viral_load - 0.1, max_viral_load - 0.4, max_viral_load - 0.75]],  # active drugs == 0.75
                [[max_viral_load + 0.1, max_viral_load - 0.4, max_viral_load - 0.7],
                 [[max_viral_load, max_viral_load + 0.1, max_viral_load + 0.1],
                  [max_viral_load - 0.5, max_viral_load - 0.5, max_viral_load - 0.7],
                  [max_viral_load - 0.9, max_viral_load - 0.9, max_viral_load - 0.9]],
                 [max_viral_load - 0.1, max_viral_load - 0.5, max_viral_load - 0.9]],   # active drugs == 1.00
                [[max_viral_load - 0.05, max_viral_load - 0.5, max_viral_load - 0.8],
                 [[max_viral_load - 0.05, max_viral_load - 0.05, max_viral_load - 0.05],
                  [max_viral_load - 0.65, max_viral_load - 0.65, max_viral_load - 0.9],
                  [max_viral_load - 1.0, max_viral_load - 1.05, max_viral_load - 1.15]],
                 [max_viral_load - 0.1, max_viral_load - 0.6, max_viral_load - 1.15]],  # active drugs == 1.25
                [[max_viral_load, max_viral_load - 0.6, max_viral_load - 0.9],
                 [[max_viral_load - 0.1, max_viral_load - 0.1, max_viral_load - 0.1],
                  [max_viral_load - 0.8, max_viral_load - 0.8, max_viral_load - 1.5],
                  [max_viral_load - 1.2, max_viral_load - 1.2, max_viral_load - 1.7]],
                 [max_viral_load - 0.1, max_viral_load - 0.7, max_viral_load - 1.4]],   # active drugs == 1.50
                [[max_viral_load - 0.15, max_viral_load - 0.8, max_viral_load - 1.25],
                 [[max_viral_load - 0.15, max_viral_load - 0.15, max_viral_load - 0.15],
                  [max_viral_load - 1.1, max_viral_load - 1.1, max_viral_load - 2.4],
                  [max_viral_load - 1.6, max_viral_load - 1.6, 2.7]],
                 [max_viral_load - 0.2, max_viral_load - 1.0, max_viral_load - 2.0]],   # active drugs == 1.75
                [[max_viral_load - 0.2, max_viral_load - 0.9, max_viral_load - 1.5],
                 [[max_viral_load - 0.2, max_viral_load - 0.2, max_viral_load - 0.2],
                  [max_viral_load - 1.2, max_viral_load - 1.2, 2.4],
                  [max_viral_load - 2.0, max_viral_load - 2.0, 2.0]],
                 [max_viral_load - 0.2, max_viral_load - 1.2, max_viral_load - 2.5]],   # active drugs == 2.00
                [[max_viral_load - 0.25, max_viral_load - 1.1, max_viral_load - 1.8],
                 [[max_viral_load - 0.25, max_viral_load - 0.25, max_viral_load - 0.25],
                  [max_viral_load - 1.35, 2.5, 2.2],
                  [1.4, 1.4, 1.4]],
                 [max_viral_load - 0.25, 1.6, 1.4]],                                    # active drugs == 2.25
                [[max_viral_load - 0.3, max_viral_load - 1.2, max_viral_load - 2.2],
                 [[max_viral_load - 0.3, max_viral_load - 0.3, max_viral_load - 0.3],
                  [max_viral_load - 1.5, 2.5, 1.8],
                  [1.2, 1.2, 1.2]],
                 [max_viral_load - 0.3, 1.2, self.min_vl_on_art]],                      # active drugs == 2.50
                [[max_viral_load - 0.4, max_viral_load - 1.6, max_viral_load - 2.6],
                 [[max_viral_load - 0.4, max_viral_load - 0.4, max_viral_load - 0.4],
                  [max_viral_load - 1.8, 2.5, 1.6],
                  [1.2, 1.2, self.min_vl_on_art]],
                 [max_viral_load - 0.4, 1.2, self.min_vl_on_art]],                      # active drugs == 2.75
                [[max_viral_load - 0.5, max_viral_load - 2.0, max_viral_load - 3.0],
                 [[max_viral_load - 0.5, max_viral_load - 0.5, max_viral_load - 0.5],
                  [max_viral_load - 2.0, 2.5, 1.2],
                  [1.2, 1.2, self.min_vl_on_art]],
                 [max_viral_load - 0.5, 1.2, self.min_vl_on_art]]]                      # active drugs >= 3.00

    def calc_viral_load(self, person):
        """
        Returns an individual's viral load this time step.
        Affected by number of active ART drugs, how long an individual has been on ART,
        their ART adherence, as well as their viral load last time step.
        """
        # use person (row) index to find the right (a, b, c) tuple
        a, b, c = self.get_matrix_value(self.viral_load_matrix, person.name)
        # calculate base viral load value
        # a * max_viral_load + b + c * min_vl_on_art
        x = a * person[col.MAX_VIRAL_LOAD] + b + c * self.min_vl_on_art
        # calculate viral load changes
        viral_load = max(0, min(x + self.vl_stdev_on_art * rng.normal(), 6.5))

        return viral_load

    def cd4_change(self, pop: Population, sub_pop):
        """
        Update CD4 count for HIV+ individuals.
        """
        # FIXME: is there a better way to pass the the cd4_tm1 column string to calc_cd4_delta?
        self.cd4_tm1_col = pop.get_correct_column(col.CD4, dt=1)
        # get cd4 outcomes
        cd4_outcomes = pop.apply_function(self.calc_cd4_delta, 1, sub_pop)
        pop.set_present_variable(col.CD4, [i[0] for i in cd4_outcomes], sub_pop)
        pop.set_present_variable(col.CD4_DELTA, [i[1] for i in cd4_outcomes], sub_pop)

    def calc_cd4_delta(self, person):
        """
        Returns an individual's change in CD4 levels this time step.
        Affected by age, sex, number of active ART drugs, how long an individual has been on ART,
        their ART adherence, use of specific ART drugs, as well as CD4 levels last time step,
        maximum CD4 levels, and individual rate of CD4 recovery on ART.
        """
        # use person (row) index to lookup cd4 delta multiplier
        x = self.get_matrix_value(self.cd4_delta_matrix, person.name)

        # find base cd4 recovery
        base_cd4_recovery_on_art = 0
        # recovery is hindered by a failing nnrti (or possibly insti) regimen
        if (((person[col.ON_NEV] or person[col.ON_EFA]) or (self.failed_insti_hinders_cd4_recovery and person[col.ON_DOL]))
                and not (person[col.ON_LPR] or person[col.ON_TAZ] or person[col.ON_DAR])
                and person[col.NUM_ACTIVE_DRUGS] <= 2):
            base_cd4_recovery_on_art = self.hindered_cd4_recovery
        # recovery increases on pi
        if person[col.ON_LPR] or person[col.ON_TAZ] or person[col.ON_DAR]:
            base_cd4_recovery_on_art += self.cd4_recovery_pi_factor
        # recovery decreases with age
        base_cd4_recovery_on_art += (person[col.AGE] - 40) * - 0.3
        # faster recovery in women
        if person[col.SEX] == SexType.Female:
            base_cd4_recovery_on_art += self.cd4_recovery_female_factor

        # calculate change in cd4
        cd4_delta = base_cd4_recovery_on_art + person[col.CD4_RECOVERY_ON_ART] * x
        # changes for people on antiretroviral drugs
        if person[col.ON_PREP] or person[col.ON_ART]:
            # adjust cd4 delta for higher previous cd4 levels
            if 100 < person[self.cd4_tm1_col] <= 200:
                cd4_delta *= 0.85
            elif person[self.cd4_tm1_col] > 200:
                cd4_delta *= 0.7

        # calculate current cd4 levels
        cd4 = max(0, person[self.cd4_tm1_col] + cd4_delta)
        # changes for people on antiretroviral drugs
        if person[col.ON_PREP] or person[col.ON_ART]:
            # add cd4 variability
            cd4 = np.sqrt(cd4) + self.cd4_stdev_on_art * rng.normal() ** 2
            # adjust cd4 according to max value
            if cd4 > person[col.MAX_CD4]:
                cd4 = person[col.MAX_CD4] + rng.normal() * 50

        return cd4, cd4_delta

    def new_mutations(self, pop: Population, sub_pop):
        """
        Update resistance mutations for HIV+ individuals.
        """
        # FIXME: is there a better way to pass the the viral_load column strings to calc_prob_new_mutation?
        self.viral_load_col = pop.get_correct_column(col.VIRAL_LOAD, dt=0)
        self.viral_load_tm1_col = pop.get_correct_column(col.VIRAL_LOAD, dt=1)
        # get new mutation probabilities
        new_mutation_probs = pop.apply_function(self.calc_prob_new_mutation, 1, sub_pop)
        # outcomes
        r = rng.uniform(size=len(sub_pop))
        possible_mutations = r < new_mutation_probs

        # people who may develop a new mutation
        possible_mutation_pop = pop.apply_bool_mask(possible_mutations, sub_pop)
        if len(possible_mutation_pop) > 0:
            # FIXME: add individual mutations here
            ...

    def calc_prob_new_mutation(self, person):
        """
        Returns the probability of acquiring a new HIV mutation this time step.
        Affected by number of active ART drugs, how long an individual has been on ART,
        their ART adherence, as well as use of specific ART drugs and viral load.
        """
        # use person (row) index to lookup new mutation probability multiplier
        x = self.get_matrix_value(self.new_mutation_matrix, person.name, on_nev=person[col.ON_NEV], on_efa=person[col.ON_EFA])
        # calculate new mutation probability
        prob_new_mutation = min(x * (person[self.viral_load_col] + person[self.viral_load_tm1_col])/2 * self.mutation_risk_change, 1)

        return prob_new_mutation

    def update_resistance(self, pop: Population):
        """
        Update the viral load, CD4 count, and resistance mutations of HIV+ individuals.
        """
        infected_pop = pop.get_sub_pop(COND(col.HIV_STATUS, op.eq, True))
        if len(infected_pop) > 0:
            # find matrix indices
            self.active_drug_indices, self.cont_on_art_indices, \
                self.adherence_indices, self.adherence_tm1_indices = self.get_all_matrix_indices(pop, infected_pop)
            # update values
            self.viral_load(pop, infected_pop)
            self.cd4_change(pop, infected_pop)
            self.new_mutations(pop, infected_pop)
