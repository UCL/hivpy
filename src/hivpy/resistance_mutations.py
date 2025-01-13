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
        # factors affecting change in viral load
        self.min_vl_on_art = 1.0
        self.vl_stdev_on_art = 0.5

        # factors affecting change in cd4 count
        self.hindered_cd4_recovery = round(-6 + (3 * rng.normal()))
        self.failed_insti_hinders_cd4_recovery = rng.choice([True, False])  # FIXME: dependent on time step length?
        self.cd4_recovery_pi_factor = 3
        self.cd4_recovery_female_factor = 2
        self.cd4_stdev = 1.2  # on a sqrt scale

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

        # factors affecting acquisition of new mutations
        self.active_drug_bins = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
        self.cont_on_art_bins = [timedelta(months=3).years(),
                                 timedelta(months=6).years()]
        self.adherence_bins = [0.5, 0.8]

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
        # FIXME: move drugs to ART module
        pop.init_variable(col.ART_ADHERENCE, 0, n_prev_steps=1)
        pop.init_variable(col.CONT_ON_ART, timedelta(months=0))
        pop.init_variable(col.NUM_ACTIVE_DRUGS, 0)
        self.init_art_drugs(pop)
        self.init_resistance_mutations(pop)

    def init_art_drugs(self, pop: Population):
        """
        Initialise ART drugs at the start of the simulation to False.
        """
        pop.init_variable(col.ON_NEV, False)
        pop.init_variable(col.ON_EFA, False)
        pop.init_variable(col.ON_DOL, False)
        pop.init_variable(col.ON_LPR, False)
        pop.init_variable(col.ON_TAZ, False)
        pop.init_variable(col.ON_DAR, False)

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

    def get_matrix_val(self, matrix, active_drugs, cont_on_art, adherence, adherence_tm1,
                       on_nev=None, on_efa=None):
        """
        Returns a value from either the new mutation matrix or the CD4 delta matrix given the input parameters.
        """
        # find matrix indices
        active_drug_index = np.digitize(active_drugs, self.active_drug_bins)
        cont_on_art_index = np.digitize(cont_on_art, self.cont_on_art_bins)
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

    def viral_load_change(self, pop: Population, sub_pop):
        """
        Update viral load for HIV+ individuals.
        """
        # get viral load outcomes
        viral_load, viral_load_delta = pop.col_apply([col.NUM_ACTIVE_DRUGS, col.CONT_ON_ART, col.ART_ADHERENCE,
                                                      pop.get_correct_column(col.ART_ADHERENCE, dt=1),
                                                      col.MAX_VIRAL_LOAD, pop.get_correct_column(col.VIRAL_LOAD, dt=1)],
                                                     self.calc_viral_load_delta, sub_pop=sub_pop)

        pop.set_present_variable(col.VIRAL_LOAD, viral_load, sub_pop)
        pop.set_present_variable(col.VIRAL_LOAD_DELTA, viral_load_delta, sub_pop)

    def get_viral_load_matrix(self, max_viral_load):
        # viral_load_matrix[active_drugs][cont_on_art][adherence]
        # FIXME: is there a better way to do this?
        return [[[max_viral_load, max_viral_load - 0.05, max_viral_load - 0.2],
                 [[max_viral_load, max_viral_load - 0.05, max_viral_load - 0.2],
                  [max_viral_load, max_viral_load - 0.05, max_viral_load - 0.2],
                  [max_viral_load, max_viral_load - 0.05, max_viral_load - 0.2]],
                 [max_viral_load, max_viral_load - 0.05, max_viral_load - 0.2]],        # active drugs == 0.00
                [[max_viral_load + 0.1, max_viral_load - 0.05, max_viral_load - 0.3],
                 [[max_viral_load + 0.1, max_viral_load + 0.1, max_viral_load + 0.1],   # FIXME: +0.0 in SAS instead of +0.1 >> typo?
                  [max_viral_load - 0.05, max_viral_load - 0.05, max_viral_load - 0.3],
                  [max_viral_load - 0.2, max_viral_load - 0.35, max_viral_load - 0.4]],
                 [max_viral_load, max_viral_load - 0.1, max_viral_load - 0.3]],         # active drugs == 0.25
                [[max_viral_load + 0.1, max_viral_load - 0.1, max_viral_load - 0.4],
                 [[max_viral_load + 0.1, max_viral_load + 0.1, max_viral_load + 0.1],   # FIXME: +0.0 in SAS instead of +0.1 >> typo?
                  [max_viral_load - 0.2, max_viral_load - 0.2, max_viral_load - 0.4],
                  [max_viral_load - 0.5, max_viral_load - 0.5, max_viral_load - 0.6]],
                 [max_viral_load - 0.1, max_viral_load - 0.3, max_viral_load - 0.6]],    # active drugs == 0.50
                [[max_viral_load + 0.1, max_viral_load - 0.25, max_viral_load - 0.55],
                 [[max_viral_load + 0.1, max_viral_load + 0.1, max_viral_load + 0.1],   # FIXME: +0.0 in SAS instead of +0.1 >> typo?
                  [max_viral_load - 0.2, max_viral_load - 0.35, max_viral_load - 0.55],
                  [max_viral_load - 0.7, max_viral_load - 0.7, max_viral_load - 0.75]],
                 [max_viral_load - 0.1, max_viral_load - 0.4, max_viral_load - 0.75]],  # active drugs == 0.75
                [[max_viral_load + 0.1, max_viral_load - 0.4, max_viral_load - 0.7],
                 [[max_viral_load + 0.1, max_viral_load + 0.1, max_viral_load + 0.1],   # FIXME: +0.0 in SAS instead of +0.1 >> typo?
                  [max_viral_load - 0.5, max_viral_load - 0.5, max_viral_load - 0.7],
                  [max_viral_load - 0.9, max_viral_load - 0.9, max_viral_load - 0.9]],
                 [max_viral_load - 0.1, max_viral_load - 0.5, max_viral_load - 0.9]],   # active drugs == 1.00
                [[max_viral_load - 0.05, max_viral_load - 0.5, max_viral_load - 0.8],
                 [[max_viral_load - 0.05, max_viral_load - 0.05, max_viral_load - 0.05],
                  [max_viral_load - 0.65, max_viral_load - 0.65, max_viral_load - 0.9],
                  [max_viral_load - 1.0, max_viral_load - 1.05, max_viral_load - 1.15]],
                 [max_viral_load - 0.1, max_viral_load - 0.6, max_viral_load - 1.15]],  # active drugs == 1.25
                [[max_viral_load - 0.1, max_viral_load - 0.6, max_viral_load - 0.9],    # FIXME: -0.0 in SAS instead of -0.1 >> typo?
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
                 [max_viral_load - 0.4, 1.2, self.min_vl_on_art]],                     # active drugs == 2.75
                [[max_viral_load - 0.5, max_viral_load - 2.0, max_viral_load - 3.0],
                 [[max_viral_load - 0.5, max_viral_load - 0.5, max_viral_load - 0.5],
                  [max_viral_load - 2.0, 2.5, 1.2],
                  [1.2, 1.2, self.min_vl_on_art]],
                 [max_viral_load - 0.5, 1.2, self.min_vl_on_art]]]                      # active drugs >= 3.00

    def calc_viral_load_delta(self, active_drugs, cont_on_art, adherence, adherence_tm1, max_viral_load, viral_load_tm1):
        """
        Returns an individual's viral load and change in viral load this time step.
        Affected by number of active ART drugs, how long an individual has been on ART,
        their ART adherence, as well as their viral load last time step.
        """
        # lookup base viral load value
        x = self.get_matrix_val(self.get_viral_load_matrix(max_viral_load), active_drugs, cont_on_art, adherence, adherence_tm1)
        # calculate viral load changes
        # FIXME: in SAS the 0-6.5 vl clamp happens after the delta is calculated; should this be the case here as well?
        viral_load = max(0, min(x + (self.vl_stdev_on_art * rng.normal()), 6.5))
        viral_load_delta = viral_load - viral_load_tm1

        return viral_load, viral_load_delta

    def cd4_change(self, pop: Population, sub_pop):
        """
        Update CD4 count for HIV+ individuals.
        """
        # get cd4 outcomes
        cd4, cd4_delta = pop.col_apply([col.AGE, col.SEX, col.NUM_ACTIVE_DRUGS, col.CONT_ON_ART, col.ART_ADHERENCE,
                                        pop.get_correct_column(col.ART_ADHERENCE, dt=1), col.ON_NEV, col.ON_EFA,
                                        col.ON_DOL, col.ON_LPR, col.ON_TAZ, col.ON_DAR, pop.get_correct_column(col.CD4, dt=1),
                                        col.MAX_CD4, col.CD4_RECOVERY_ON_ART],
                                       self.calc_cd4_delta, sub_pop=sub_pop)

        pop.set_present_variable(col.CD4, cd4, sub_pop)
        pop.set_present_variable(col.CD4_DELTA, cd4_delta, sub_pop)

    def calc_cd4_delta(self, age, sex, active_drugs, cont_on_art, adherence, adherence_tm1,
                       on_nev, on_efa, on_dol, on_lpr, on_taz, on_dar, cd4_tm1, max_cd4, cd4_recovery_on_art):
        """
        Returns an individual's change in CD4 levels this time step.
        Affected by age, sex, number of active ART drugs, how long an individual has been on ART,
        their ART adherence, use of specific ART drugs, as well as CD4 levels last time step,
        maximum CD4 levels, and individual rate of CD4 recovery on ART.
        """
        # lookup cd4 delta multiplier
        x = self.get_matrix_val(self.new_mutation_matrix, active_drugs, cont_on_art, adherence, adherence_tm1)

        # find base cd4 recovery
        base_cd4_recovery_on_art = 0  # FIXME: dependent on time step length?
        # recovery is hindered by a failing nnrti (or possibly insti) regimen
        if (((on_nev or on_efa) or (self.failed_insti_hinders_cd4_recovery and on_dol))
                and not (on_lpr or on_taz or on_dar) and active_drugs <= 2):
            base_cd4_recovery_on_art = self.hindered_cd4_recovery
        # recovery increases on pi
        if on_lpr or on_dar or on_taz:
            base_cd4_recovery_on_art += self.cd4_recovery_pi_factor
        # recovery decreases with age
        base_cd4_recovery_on_art += (age - 40) * - 0.3
        # faster recovery in women
        if sex is SexType.Female:
            base_cd4_recovery_on_art += self.cd4_recovery_female_factor

        # calculate change in cd4
        cd4_delta = base_cd4_recovery_on_art + (cd4_recovery_on_art * x)
        cd4 = max(0, cd4_tm1 + cd4_delta)

        # FIXME: isn't this always true? should everyone be on ART?
        if active_drugs >= 0:
            # adjust cd4 delta for higher cd4 levels when on ART
            if 100 < cd4_tm1 <= 200:
                cd4_delta *= 0.85
            elif cd4_tm1 > 200:
                cd4_delta *= 0.7
            # add cd4 variability when on ART
            cd4 = np.sqrt(cd4) + self.cd4_stdev * rng.normal() ** 2

        if cont_on_art >= timedelta(months=0).years() and cd4 > max_cd4:
            # adjust cd4 according to max value when on ART
            cd4 = max_cd4 + rng.normal() * 50

        return cd4, cd4_delta

    def calc_prob_new_mutation(self, active_drugs, cont_on_art, adherence, adherence_tm1,
                               on_nev, on_efa, viral_load, viral_load_tm1):
        """
        Returns the probability of acquiring a new HIV mutation this time step.
        Affected by number of active ART drugs, how long an individual has been on ART,
        their ART adherence, as well as use of specific ART drugs and viral load.
        """
        # lookup new mutation probability multiplier
        x = self.get_matrix_val(self.new_mutation_matrix, active_drugs, cont_on_art, adherence,
                                adherence_tm1, on_nev, on_efa)
        # calculate new mutation probability
        prob_new_mutation = min(x * (viral_load + viral_load_tm1)/2, 1)

        return prob_new_mutation

    def update_resistance(self, pop: Population):
        """
        Update the viral load, CD4 count, and resistance mutations of HIV+ individuals.
        """
        infected_pop = pop.get_sub_pop(COND(col.HIV_STATUS, op.eq, True))
        if len(infected_pop) > 0:
            self.viral_load_change(pop, infected_pop)
            self.cd4_change(pop, infected_pop)
