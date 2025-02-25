from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .population import Population

import importlib.resources
import operator as op
from enum import IntEnum

import numpy as np

import hivpy.column_names as col

from .common import COND, SexType, rng, timedelta
from .resistance_mutations_data import ResistanceMutationsData


class MutationStatus(IntEnum):
    Majority = 0
    Minority = 1    # once a mutation is present,
    Absent = 2      # it can never be absent again


class ResistanceMutationsModule:

    def __init__(self):

        # init resistance data
        with importlib.resources.path("hivpy.data", "resistance_mutations.yaml") as data_path:
            self.rm_data = ResistanceMutationsData(data_path)

        # matrix indexing boundaries
        self.active_drug_bins = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
        self.cont_on_art_bins = [timedelta(months=3).years(), timedelta(months=6).years()]
        self.adherence_bins = [0.5, 0.8]

        # factors affecting change in viral load
        self.min_vl_on_art = self.rm_data.min_vl_on_art
        self.vl_stdev_on_art = self.rm_data.vl_stdev_on_art

        # factors affecting change in cd4 count
        self.hindered_cd4_recovery = round(-6 + (3 * rng.normal()))  # FIXME: dependent on time step length
        self.failed_insti_hinders_cd4_recovery = rng.choice([True, False])
        self.cd4_recovery_pi_factor = self.rm_data.cd4_recovery_pi_factor
        self.cd4_recovery_female_factor = self.rm_data.cd4_recovery_female_factor
        self.cd4_stdev_on_art = self.rm_data.cd4_stdev_on_art

        # factors affecting acquisition of new mutations
        self.mutation_risk_change = self.rm_data.mutation_risk_change.sample()
        self.risk_change_tams_resist = self.rm_data.risk_change_tams_resist
        self.risk_change_151_resist = self.rm_data.risk_change_151_resist
        # FIXME: resistance rates dependent on time step length
        self.ten_resist_rate = self.rm_data.ten_resist_rate.sample()
        self.dol_resist_rate = self.rm_data.dol_resist_rate.sample()
        self.len_resist_rate = self.rm_data.len_resist_rate.sample()
        self.incr_len_resist = self.rm_data.incr_len_resist
        self.cab_resist_factor = self.rm_data.cab_resist_factor.sample()
        self.risk_change_cab_resist = self.rm_data.risk_change_cab_resist.sample()

        # viral_load_matrix[active_drugs][cont_on_art_tm1][adherence][adherence_tm1]
        # (a, b, c) tuples used to calculate base viral load (a * max_viral_load + b + c * min_vl_on_art)
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

        # cd4_delta_matrix[active_drugs][cont_on_art_tm1][adherence][adherence_tm1]
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

        # new_mutation_matrix[active_drugs][cont_on_art_tm1][adherence][adherence_tm1]
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
        pop.init_variable(col.CONT_ON_ART, timedelta(months=0), n_prev_steps=1)
        pop.init_variable(col.CONT_ON_ARV, timedelta(months=0))
        pop.init_variable(col.NUM_ACTIVE_DRUGS, 0)
        pop.init_variable(col.ART_ADHERENCE, 0, n_prev_steps=1)
        pop.init_variable(col.RESISTANCE_INDEX, -1)
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
        pop.init_variable(col.RTTA_MUTATIONS, 0)  # only tams are tracked with integers
        pop.init_variable(col.RT184_MUTATION, MutationStatus.Absent)
        pop.init_variable(col.RT151_MUTATION, MutationStatus.Absent)
        pop.init_variable(col.RT65_MUTATION, MutationStatus.Absent)
        pop.init_variable(col.RT103_MUTATION, MutationStatus.Absent)
        pop.init_variable(col.RT181_MUTATION, MutationStatus.Absent)
        pop.init_variable(col.RT190_MUTATION, MutationStatus.Absent)
        pop.init_variable(col.PR32_MUTATION, MutationStatus.Absent)
        pop.init_variable(col.PR46_MUTATION, MutationStatus.Absent)
        pop.init_variable(col.PR47_MUTATION, MutationStatus.Absent)
        pop.init_variable(col.PR50L_MUTATION, MutationStatus.Absent)
        pop.init_variable(col.PR50V_MUTATION, MutationStatus.Absent)
        pop.init_variable(col.PR54_MUTATION, MutationStatus.Absent)
        pop.init_variable(col.PR76_MUTATION, MutationStatus.Absent)
        pop.init_variable(col.PR82_MUTATION, MutationStatus.Absent)
        pop.init_variable(col.PR84_MUTATION, MutationStatus.Absent)
        pop.init_variable(col.PR88_MUTATION, MutationStatus.Absent)
        pop.init_variable(col.PR90_MUTATION, MutationStatus.Absent)
        pop.init_variable(col.IN118_MUTATION, MutationStatus.Absent)
        pop.init_variable(col.IN140_MUTATION, MutationStatus.Absent)
        pop.init_variable(col.IN148_MUTATION, MutationStatus.Absent)
        pop.init_variable(col.IN155_MUTATION, MutationStatus.Absent)
        pop.init_variable(col.IN263_MUTATION, MutationStatus.Absent)
        pop.init_variable(col.CA66_MUTATION, MutationStatus.Absent)

    def get_mutation_presence(self, mutation: MutationStatus):
        """
        Helper function for simplifying MutationStatus presence and absence into a boolean.
        """
        return True if mutation != MutationStatus.Absent else False

    def get_all_matrix_indices(self, pop, sub_pop):
        """
        Returns all active drug, continuous ART usage, and adherence indices for the HIV+ sub-population.
        These indices are used to look up values in the viral load, CD4 delta, and new mutation matrices.
        """
        # find matrix indices
        active_drug_indices = np.digitize(pop.get_variable(col.NUM_ACTIVE_DRUGS, sub_pop), self.active_drug_bins)
        cont_on_art_tm1_indices = np.digitize([x.years() for x in pop.get_variable(col.CONT_ON_ART, sub_pop, dt=1)],
                                              self.cont_on_art_bins)
        adherence_indices = np.digitize(pop.get_variable(col.ART_ADHERENCE, sub_pop), self.adherence_bins)
        adherence_tm1_indices = np.digitize(pop.get_variable(col.ART_ADHERENCE, sub_pop, dt=1), self.adherence_bins)
        # discount adherence last time step when not on ART for 3-6 months
        adherence_tm1_indices = np.where(cont_on_art_tm1_indices != 1, -1, adherence_tm1_indices)

        return active_drug_indices, cont_on_art_tm1_indices, adherence_indices, adherence_tm1_indices

    def get_matrix_value(self, matrix, i, on_nev=None, on_efa=None):
        """
        Returns a value from a given matrix (expecting one of the viral load, CD4 delta, or new mutation matrices)
        for a specific individual.
        """
        active_drug_index, cont_on_art_tm1_index, \
            adherence_index, adherence_tm1_index = self.get_individual_matrix_indices(i, on_nev, on_efa)

        return (matrix[active_drug_index][cont_on_art_tm1_index][adherence_index][adherence_tm1_index]
                if adherence_tm1_index > -1 else matrix[active_drug_index][cont_on_art_tm1_index][adherence_index])

    def get_individual_matrix_indices(self, i, on_nev=None, on_efa=None):
        """
        Returns the active drug, continuous ART usage, and adherence indices for a specific HIV+ individual.
        """
        # use resistance index to find matrix indices
        active_drug_index = self.active_drug_indices[i]
        cont_on_art_tm1_index = self.cont_on_art_tm1_indices[i]
        adherence_index = self.adherence_indices[i]
        adherence_tm1_index = self.adherence_tm1_indices[i]
        # adjust adherence index if taking specific ART drugs (only relevant to new mutation probability)
        if adherence_index == 0 and (on_nev or on_efa):
            adherence_index += 1

        return active_drug_index, cont_on_art_tm1_index, adherence_index, adherence_tm1_index

    def update_viral_load_art(self, pop: Population, sub_pop):
        """
        Update viral load in HIV+ individuals.
        """
        # get viral load outcomes
        viral_load = pop.apply_function(self.calc_viral_load, 1, sub_pop)
        pop.set_present_variable(col.VIRAL_LOAD, viral_load, sub_pop)

    def calc_viral_load(self, person):
        """
        Returns an individual's viral load this time step.
        Affected by number of active ART drugs, how long an individual has been on ART,
        their ART adherence, as well as their viral load last time step.
        """
        # use resistance index to find the right (a, b, c) tuple
        a, b, c = self.get_matrix_value(self.viral_load_matrix, person[col.RESISTANCE_INDEX])
        # calculate base viral load value
        # a * max_viral_load + b + c * min_vl_on_art
        x = a * person[col.MAX_VIRAL_LOAD] + b + c * self.min_vl_on_art
        # calculate viral load changes
        viral_load = max(0, min(x + self.vl_stdev_on_art * rng.normal(), 6.5))

        return viral_load

    def update_cd4_art(self, pop: Population, sub_pop):
        """
        Update CD4 count in HIV+ individuals.
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
        # use resistance index to lookup cd4 delta multiplier
        x = self.get_matrix_value(self.cd4_delta_matrix, person[col.RESISTANCE_INDEX])

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
        base_cd4_recovery_on_art -= (person[col.AGE] - 40) * 0.3
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
            cd4 = (np.sqrt(cd4) + self.cd4_stdev_on_art * rng.normal()) ** 2
            # adjust cd4 according to max value
            if cd4 > person[col.MAX_CD4]:
                cd4 = person[col.MAX_CD4] + rng.normal() * 50

        return cd4, cd4_delta

    def update_new_mutations_arising_art(self, pop: Population, sub_pop):
        """
        Update new resistance mutations arising in HIV+ individuals.
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

            # tams
            tams = pop.transform_group([col.ON_ZDV, col.ON_3TC, col.RTTA_MUTATIONS],
                                       self.calc_rttams_outcomes, sub_pop=possible_mutation_pop)
            pop.set_present_variable(col.RTTA_MUTATIONS, tams, possible_mutation_pop)

            # calculate and set a new majority mutation
            def set_new_majority_mutation(mutation_col: str, drug_cols: list[str], calc_func):
                # find people undergoing a given mutation this time step
                mutation_mask = pop.transform_group(drug_cols, calc_func, sub_pop=possible_mutation_pop)
                mutation_pop = pop.apply_bool_mask(mutation_mask, possible_mutation_pop)
                # set outcomes
                if len(mutation_pop) > 0:
                    pop.set_present_variable(mutation_col, MutationStatus.Majority, mutation_pop)

            # m184
            set_new_majority_mutation(col.RT184_MUTATION, [col.ON_3TC, col.ON_ISL, col.RT184_MUTATION],
                                      self.calc_rt184m_outcomes)
            # q151
            set_new_majority_mutation(col.RT151_MUTATION, [col.ON_ZDV, col.RT151_MUTATION],
                                      self.calc_rt151m_outcomes)
            # k65
            set_new_majority_mutation(col.RT65_MUTATION, [col.ON_TEN, col.ON_ZDV, col.RT65_MUTATION],
                                      self.calc_rt65m_outcomes)

            # k103, y181, and g190 (nnrti mutations)
            k103 = pop.transform_group([col.ON_NEV, col.ON_EFA, col.RT181_MUTATION, col.RT190_MUTATION],
                                       self.calc_rt103m_outcomes, sub_pop=possible_mutation_pop)
            y181 = pop.transform_group([col.ON_NEV, col.ON_EFA, col.RT103_MUTATION, col.RT190_MUTATION],
                                       self.calc_rt181m_outcomes, sub_pop=possible_mutation_pop)
            g190 = pop.transform_group([col.ON_NEV, col.ON_EFA, col.RT103_MUTATION, col.RT181_MUTATION],
                                       self.calc_rt190m_outcomes, sub_pop=possible_mutation_pop)
            # make all calculations before setting outcomes to prevent changes
            # this time step from affecting each other
            if len(pop.apply_bool_mask(k103, possible_mutation_pop)) > 0:
                pop.set_present_variable(col.RT103_MUTATION, MutationStatus.Majority,
                                         pop.apply_bool_mask(k103, possible_mutation_pop))
            if len(pop.apply_bool_mask(y181, possible_mutation_pop)) > 0:
                pop.set_present_variable(col.RT181_MUTATION, MutationStatus.Majority,
                                         pop.apply_bool_mask(y181, possible_mutation_pop))
            if len(pop.apply_bool_mask(g190, possible_mutation_pop)) > 0:
                pop.set_present_variable(col.RT190_MUTATION, MutationStatus.Majority,
                                         pop.apply_bool_mask(g190, possible_mutation_pop))

            # p32
            set_new_majority_mutation(col.PR32_MUTATION, [col.ON_LPR], self.calc_pr32m_outcomes)
            # p46
            set_new_majority_mutation(col.PR46_MUTATION, [col.ON_LPR], self.calc_pr46m_outcomes)
            # p47
            set_new_majority_mutation(col.PR47_MUTATION, [col.ON_LPR], self.calc_pr47m_outcomes)
            # p50l
            set_new_majority_mutation(col.PR50L_MUTATION, [col.ON_TAZ], self.calc_pr50lm_outcomes)
            # p50v
            set_new_majority_mutation(col.PR50V_MUTATION, [col.ON_DAR], self.calc_pr50vm_outcomes)
            # p54
            set_new_majority_mutation(col.PR54_MUTATION, [col.ON_LPR, col.ON_DAR], self.calc_pr54m_outcomes)
            # p76
            set_new_majority_mutation(col.PR76_MUTATION, [col.ON_LPR, col.ON_DAR], self.calc_pr76m_outcomes)
            # p82
            set_new_majority_mutation(col.PR82_MUTATION, [col.ON_LPR], self.calc_pr82m_outcomes)
            # p84
            set_new_majority_mutation(col.PR84_MUTATION, [col.ON_DAR, col.ON_TAZ], self.calc_pr84m_outcomes)
            # p88
            set_new_majority_mutation(col.PR88_MUTATION, [col.ON_TAZ], self.calc_pr88m_outcomes)

            # tally up all mutations
            resistance_mutations = pop.apply_function(self.calc_total_mutations, 1, possible_mutation_pop)
            pop.set_present_variable(col.RESISTANCE_MUTATIONS, resistance_mutations, possible_mutation_pop)

    def calc_prob_new_mutation(self, person):
        """
        Returns the probability of acquiring a new HIV mutation this time step.
        Affected by number of active ART drugs, how long an individual has been on ART,
        their ART adherence, as well as use of specific ART drugs and viral load.
        """
        # use resistance index to lookup new mutation probability multiplier
        x = self.get_matrix_value(self.new_mutation_matrix, person[col.RESISTANCE_INDEX],
                                  on_nev=person[col.ON_NEV], on_efa=person[col.ON_EFA])
        # calculate new mutation probability
        prob_new_mutation = min(x * (person[self.viral_load_col] + person[self.viral_load_tm1_col])/2 * self.mutation_risk_change, 1)

        return prob_new_mutation

    def calc_rttams_outcomes(self, on_zdv, on_3tc, rttams, size):
        """
        Returns RT gene TAMs outcomes.
        """
        # outcomes
        r = rng.uniform(size=size) / self.risk_change_tams_resist
        prob_mutation = 0
        if on_zdv:
            if on_3tc:
                prob_mutation = 0.12
            else:
                prob_mutation = 0.20
        ta_mutations = r < prob_mutation
        extra_ta_mutations = (prob_mutation <= r) & (r < prob_mutation + 0.01) if prob_mutation > 0 else r < prob_mutation

        # increment tams
        tams = np.array([rttams] * size)
        tams[ta_mutations] += 1
        tams[extra_ta_mutations] += 2
        # cap number of mutations at 6
        tams[tams > 6] = 6

        return tams

    def calc_rt184m_outcomes(self, on_3tc, on_isl, rt184m, size):
        """
        Returns RT gene M184 majority mutation outcomes.
        """
        prob_mutation = 0.8 if on_3tc and rt184m != MutationStatus.Majority else 0
        m184_mutations = rng.uniform(size=size) < prob_mutation

        prob_mutation = 0.1 if on_isl and rt184m != MutationStatus.Majority else 0
        m184_mutations |= rng.uniform(size=size) < prob_mutation

        return m184_mutations

    def calc_rt151m_outcomes(self, on_zdv, rt151m, size):
        """
        Returns RT gene Q151 majority mutation outcomes.
        """
        prob_mutation = 0.02 if on_zdv and rt151m != MutationStatus.Majority else 0
        q151_mutations = rng.uniform(size=size) / self.risk_change_151_resist < prob_mutation

        return q151_mutations

    def calc_rt65m_outcomes(self, on_ten, on_zdv, rt65m, size):
        """
        Returns RT gene K65 majority mutation outcomes.
        """
        r = rng.uniform(size=size)
        prob_mutation = 0.02 if on_ten and rt65m != MutationStatus.Majority else 0
        k65_mutations = r < prob_mutation if on_zdv else r < self.ten_resist_rate

        return k65_mutations

    def calc_rt103m_outcomes(self, on_nev, on_efa, rt181m, rt190m, size):
        """
        Returns RT gene K103 majority mutation outcomes.
        """
        # outcomes on nev
        prob_mutation = 0.2 if on_nev and rt181m != MutationStatus.Majority and rt190m != MutationStatus.Majority else 0
        k103_mutations = rng.uniform(size=size) < prob_mutation

        # outcomes on efa
        prob_mutation = 0.6 if on_efa and rt181m != MutationStatus.Majority and rt190m != MutationStatus.Majority else 0
        k103_mutations |= rng.uniform(size=size) < prob_mutation

        return k103_mutations

    def calc_rt181m_outcomes(self, on_nev, on_efa, rt103m, rt190m, size):
        """
        Returns RT gene Y181 majority mutation outcomes.
        """
        # outcomes on nev
        prob_mutation = 0.4 if on_nev and rt103m != MutationStatus.Majority and rt190m != MutationStatus.Majority else 0
        y181_mutations = rng.uniform(size=size) < prob_mutation

        # outcomes on efa
        prob_mutation = 0.1 if on_efa and rt103m != MutationStatus.Majority and rt190m != MutationStatus.Majority else 0
        y181_mutations |= rng.uniform(size=size) < prob_mutation

        return y181_mutations

    def calc_rt190m_outcomes(self, on_nev, on_efa, rt103m, rt181m, size):
        """
        Returns RT gene G190 majority mutation outcomes.
        """
        # outcomes on nev
        prob_mutation = 0.2 if on_nev and rt103m != MutationStatus.Majority and rt181m != MutationStatus.Majority else 0
        g190_mutations = rng.uniform(size=size) < prob_mutation

        # outcomes on efa
        prob_mutation = 0.1 if on_efa and rt103m != MutationStatus.Majority and rt181m != MutationStatus.Majority else 0
        g190_mutations |= rng.uniform(size=size) < prob_mutation

        return g190_mutations

    def calc_pr32m_outcomes(self, on_lpr, size):
        """
        Returns PR gene P32 majority mutation outcomes.
        """
        prob_mutation = 0.01 if on_lpr else 0
        p32_mutations = rng.uniform(size=size) < prob_mutation

        return p32_mutations

    def calc_pr46m_outcomes(self, on_lpr, size):
        """
        Returns PR gene P46 majority mutation outcomes.
        """
        prob_mutation = 0.02 if on_lpr else 0
        p46_mutations = rng.uniform(size=size) < prob_mutation

        return p46_mutations

    def calc_pr47m_outcomes(self, on_lpr, size):
        """
        Returns PR gene P47 majority mutation outcomes.
        """
        prob_mutation = 0.01 if on_lpr else 0
        p47_mutations = rng.uniform(size=size) < prob_mutation

        return p47_mutations

    def calc_pr50lm_outcomes(self, on_taz, size):
        """
        Returns PR gene P50L majority mutation outcomes.
        """
        prob_mutation = 0.03 if on_taz else 0
        p50l_mutations = rng.uniform(size=size) < prob_mutation

        return p50l_mutations

    def calc_pr50vm_outcomes(self, on_dar, size):
        """
        Returns PR gene P50V majority mutation outcomes.
        """
        prob_mutation = 0.01 if on_dar else 0
        p50v_mutations = rng.uniform(size=size) < prob_mutation

        return p50v_mutations

    def calc_pr54m_outcomes(self, on_lpr, on_dar, size):
        """
        Returns PR gene P54 majority mutation outcomes.
        """
        # outcomes on lpr
        prob_mutation = 0.02 if on_lpr else 0
        p54_mutations = rng.uniform(size=size) < prob_mutation

        # outcomes on dar
        prob_mutation = 0.01 if on_dar else 0
        p54_mutations |= rng.uniform(size=size) < prob_mutation

        return p54_mutations

    def calc_pr76m_outcomes(self, on_lpr, on_dar, size):
        """
        Returns PR gene P76 majority mutation outcomes.
        """
        # outcomes on lpr
        prob_mutation = 0.02 if on_lpr else 0
        p76_mutations = rng.uniform(size=size) < prob_mutation

        # outcomes on dar
        prob_mutation = 0.01 if on_dar else 0
        p76_mutations |= rng.uniform(size=size) < prob_mutation

        return p76_mutations

    def calc_pr82m_outcomes(self, on_lpr, size):
        """
        Returns PR gene P82 majority mutation outcomes.
        """
        prob_mutation = 0.02 if on_lpr else 0
        p82_mutations = rng.uniform(size=size) < prob_mutation

        return p82_mutations

    def calc_pr84m_outcomes(self, on_dar, on_taz, size):
        """
        Returns PR gene P84 majority mutation outcomes.
        """
        # outcomes on dar
        prob_mutation = 0.01 if on_dar else 0
        p84_mutations = rng.uniform(size=size) < prob_mutation

        # outcomes on taz
        prob_mutation = 0.03 if on_taz else 0
        p84_mutations |= rng.uniform(size=size) < prob_mutation

        return p84_mutations

    def calc_pr88m_outcomes(self, on_taz, size):
        """
        Returns PR gene P88 majority mutation outcomes.
        """
        prob_mutation = 0.03 if on_taz else 0
        p88_mutations = rng.uniform(size=size) < prob_mutation

        return p88_mutations

    def calc_total_mutations(self, person):
        """
        Returns the total number of resistance mutations present in a given individual.
        """
        return (person[col.RTTA_MUTATIONS] +
                self.get_mutation_presence(MutationStatus(person[col.RT184_MUTATION])) +
                self.get_mutation_presence(MutationStatus(person[col.RT151_MUTATION])) +
                self.get_mutation_presence(MutationStatus(person[col.RT65_MUTATION])) +
                self.get_mutation_presence(MutationStatus(person[col.RT103_MUTATION])) +
                self.get_mutation_presence(MutationStatus(person[col.RT181_MUTATION])) +
                self.get_mutation_presence(MutationStatus(person[col.RT190_MUTATION])) +
                self.get_mutation_presence(MutationStatus(person[col.PR32_MUTATION])) +
                self.get_mutation_presence(MutationStatus(person[col.PR46_MUTATION])) +
                self.get_mutation_presence(MutationStatus(person[col.PR47_MUTATION])) +
                self.get_mutation_presence(MutationStatus(person[col.PR50L_MUTATION])) +
                self.get_mutation_presence(MutationStatus(person[col.PR50V_MUTATION])) +
                self.get_mutation_presence(MutationStatus(person[col.PR54_MUTATION])) +
                self.get_mutation_presence(MutationStatus(person[col.PR76_MUTATION])) +
                self.get_mutation_presence(MutationStatus(person[col.PR82_MUTATION])) +
                self.get_mutation_presence(MutationStatus(person[col.PR84_MUTATION])) +
                self.get_mutation_presence(MutationStatus(person[col.PR88_MUTATION])) +
                self.get_mutation_presence(MutationStatus(person[col.PR90_MUTATION])) +
                self.get_mutation_presence(MutationStatus(person[col.IN118_MUTATION])) +
                self.get_mutation_presence(MutationStatus(person[col.IN140_MUTATION])) +
                self.get_mutation_presence(MutationStatus(person[col.IN148_MUTATION])) +
                self.get_mutation_presence(MutationStatus(person[col.IN155_MUTATION])) +
                self.get_mutation_presence(MutationStatus(person[col.IN263_MUTATION])) +
                self.get_mutation_presence(MutationStatus(person[col.CA66_MUTATION])))

    def update_resistance(self, pop: Population):
        """
        Update the viral load, CD4 count, and resistance mutations of HIV+ individuals.
        """
        infected_pop = pop.get_sub_pop(COND(col.HIV_STATUS, op.eq, True))
        if len(infected_pop) > 0:

            # find matrix indices
            self.active_drug_indices, self.cont_on_art_tm1_indices, \
                self.adherence_indices, self.adherence_tm1_indices = self.get_all_matrix_indices(pop, infected_pop)
            pop.set_present_variable(col.RESISTANCE_INDEX, range(len(infected_pop)), infected_pop)

            # update values
            self.update_viral_load_art(pop, infected_pop)
            self.update_cd4_art(pop, infected_pop)
            self.update_new_mutations_arising_art(pop, infected_pop)
