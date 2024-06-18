from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .population import Population

import operator as op

import numpy as np
import pandas as pd

import hivpy.column_names as col

from . import output
from .common import COND, SexType, opposite_sex, rng, timedelta


class HIVStatusModule:

    initial_hiv_newp_threshold = 7  # lower limit for HIV infection at start of epidemic
    initial_hiv_prob = 0.8  # for those with enough partners at start of epidemic

    def __init__(self):
        self.output = output.simulation_output
        # FIXME: move these to data file
        # a more descriptive name would be nice
        self.tr_rate_primary = 0.16
        self.tr_rate_undetectable_vl = rng.choice([0.0000, 0.0001, 0.0010], p=[0.7, 0.2, 0.1])
        self.transmission_factor = rng.choice([1/1.5, 1, 1.5])
        self.stp_transmission_factor = rng.choice(
            [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1/0.8, 1/0.6, 1/0.4])
        # proportion of infected stps in population by sex and age group
        # age groups: 15-24, 25-34, 35-44, 45-54, 55-64
        self.ratio_infected_stp = {SexType.Male: np.zeros(5),
                                   SexType.Female: np.zeros(5)}
        # proportion of stps with different viral load groups in general population for each sex and age group
        self.ratio_vl_stp = {SexType.Male: [np.zeros(6)]*5,
                             SexType.Female: [np.zeros(6)]*5}
        # proportion of monogamous partners in general population for each sex and age group
        self.prop_monogamous = {SexType.Male: np.zeros(5),
                                SexType.Female: np.zeros(5)}
        self.incidence_factor = {SexType.Male: 1,
                                 SexType.Female: 1}
        self.incidence = {SexType.Male: 0,
                          SexType.Female: 0}
        self.women_transmission_factor = rng.choice([1., 1.5, 2.], p=[0.05, 0.25, 0.7])
        self.young_women_transmission_factor = rng.choice([1., 2., 3.]) * self.women_transmission_factor
        self.sti_transmission_factor = rng.choice([2., 3.])
        self.stp_transmission_means = self.transmission_factor * self.stp_transmission_factor * \
            np.array([0, self.tr_rate_undetectable_vl / self.transmission_factor, 0.01,
                      0.03, 0.06, 0.1, self.tr_rate_primary])
        self.stp_transmission_sigmas = np.array(
            [0, 0.000025, 0.0025, 0.0075, 0.015, 0.025, 0.075])
        self.circumcision_risk_reduction = 0.4  # reduce infection risk by 60%
        self.vl_base_change = rng.choice([1.0, 1.5, 2.0])  # TODO: move to data file
        self.cd4_base_change = rng.choice([0.7, 0.85, 1.0, 1/0.85, 1/0.7])  # TODO: move to data file

        self.initial_mean_sqrt_cd4 = 27.5
        self.sigma_cd4 = 1.2

        # HIV related risk of disease and death
        self.disease_cd4_boundaries = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175,
                                                200, 225, 250, 275, 300, 325, 350, 375, 400, 450, 500, 650])
        self.base_rate_disease = np.array([2.5, 1.8, 1.1, 0.8, 0.5, 0.4, 0.32, 0.28,
                                           0.23, 0.20, 0.17, 0.13, 0.10, 0.08, 0.065,
                                           0.055, 0.045, 0.037, 0.03, 0.025, 0.022, 0.02,
                                           0.016, 0.013, 0.01, 0.002])
        assert (len(self.disease_cd4_boundaries)+1 == len(self.base_rate_disease))
        self.disease_vl_boundaries = np.array([3, 4, 4.5, 5, 5.5])
        self.vl_disease_factor = np.array([0.2, 0.3, 0.6, 0.9, 1.2, 1.6])
        self.who3_risk_factor = 5

        self.who3_proportion_tb = 0.2
        self.tb_base_diagnosis_prob = rng.choice([0.25, 0.5, 0.75])

        self.prop_ADC_cryp_meningitis = 0.15
        self.CM_base_diagnosis_prob = rng.choice([0.25, 0.5, 0.75])

        self.prop_ADC_SBI = 0.15
        self.SBI_base_diagnosis_prob = rng.choice([0.25, 0.5, 0.75])

        self.prop_ADC_other = 1 - (self.prop_ADC_cryp_meningitis + self.prop_ADC_SBI)
        self.WHO4_base_diagnosis_prob = rng.choice([0.25, 0.5, 0.75])

        self.hiv_mortality_factor = 0.25
        self.tb_mortality_factor = rng.choice([1.5, 2, 3])
        self.sbi_mortality_factor = rng.choice([1.5, 2, 3])
        self.cm_mortality_factor = rng.choice([3, 5, 10])
        self.other_adc_mortality_factor = rng.choice([1.5, 2, 3])

    def init_HIV_variables(self, population: Population):
        population.init_variable(col.HIV_STATUS, False)
        population.init_variable(col.LTP_STATUS, False)
        population.init_variable(col.LTP_MONOGAMOUS, False)
        population.init_variable(col.DATE_HIV_INFECTION, None)
        population.init_variable(col.IN_PRIMARY_INFECTION, False)
        population.init_variable(col.CD4, 0.0)
        population.init_variable(col.MAX_CD4, 6.6 + rng.normal(0, 0.25, size=population.size))
        population.init_variable(col.HIV_DIAGNOSED, False)
        population.init_variable(col.HIV_DIAGNOSIS_DATE, None)
        population.init_variable(col.VIRAL_LOAD_GROUP, None)
        population.init_variable(col.VIRAL_LOAD, 0.0)
        population.init_variable(col.X4_VIRUS, False)

        population.init_variable(col.WHO3_EVENT, False)
        population.init_variable(col.NON_TB_WHO3, False)
        population.init_variable(col.TB, False)
        population.init_variable(col.TB_DIAGNOSED, False)
        population.init_variable(col.ADC, False)
        population.init_variable(col.C_MENINGITIS, False)
        population.init_variable(col.C_MENINGITIS_DIAGNOSED, False)
        population.init_variable(col.SBI, False)
        population.init_variable(col.SBI_DIAGNOSED, False)
        population.init_variable(col.WHO4_OTHER, False)
        population.init_variable(col.WHO4_OTHER_DIAGNOSED, False)

        self.init_resistance_mutations(population)

    def initial_HIV_status(self, population: pd.DataFrame):
        """
        Initialise HIV status at the start of the simulation to no infections.
        """
        # This may be useful as a separate method if we end up representing status
        # as something more complex than a boolean, e.g. an enum.
        return pd.Series(False, population.index)

    def introduce_HIV(self, population: Population):
        """
        Initialise HIV status at the start of the pandemic.
        """
        # At the start of the epidemic, we consider only people with short-term partners over
        # the threshold as potentially infected.
        initial_candidates = population.get_sub_pop(
            [(col.NUM_PARTNERS, op.ge, self.initial_hiv_newp_threshold)])
        # initial_candidates = population[col.NUM_PARTNERS] >= self.initial_hiv_newp_threshold
        # Each of them has the same probability of being infected.
        num_init_candidates = len(initial_candidates)
        rands = rng.uniform(size=num_init_candidates)
        initial_infection = rands < self.initial_hiv_prob
        population.set_present_variable(col.HIV_STATUS, initial_infection, sub_pop=initial_candidates)
        newly_infected = population.get_sub_pop([(col.HIV_STATUS, op.eq, True)])
        self.initialise_HIV_progression(population, newly_infected)

    def update_partner_risk_vectors(self, population: Population):
        """
        Calculate the risk factor associated with each sex and age group.
        """
        # Update viral load groups based on viral load / primary infection

        HIV_positive_pop = population.get_sub_pop([(col.HIV_STATUS, op.eq, True)])
        in_primary_infection = population.get_sub_pop([(col.IN_PRIMARY_INFECTION, op.eq, True)])

        population.set_present_variable(col.VIRAL_LOAD_GROUP, 5, in_primary_infection)

        # Should we be using for loops here or can we do better?
        for sex in SexType:
            for age_group in range(5):   # FIXME: need to get number of age groups from somewhere
                sub_pop = population.get_sub_pop([(col.SEX, op.eq, sex),
                                                  (col.SEX_MIX_AGE_GROUP, op.eq, age_group)])
                # total number of people partnered to people in this group
                n_stp_total = sum(population.get_variable(col.NUM_PARTNERS, sub_pop))
                # num people partnered to HIV+ people in this group
                HIV_positive_subpop = population.get_sub_pop_intersection(sub_pop, HIV_positive_pop)
                n_stp_of_infected = sum(population.get_variable(col.NUM_PARTNERS, HIV_positive_subpop))
                # probability of being HIV positive
                if n_stp_of_infected == 0:
                    self.ratio_infected_stp[sex][age_group] = 0
                else:
                    self.ratio_infected_stp[sex][age_group] = n_stp_of_infected / \
                        n_stp_total  # TODO: need to double check this definition
                # chances of being in a given viral group
                if n_stp_of_infected > 0:
                    self.ratio_vl_stp[sex][age_group] = [
                        sum(population.get_variable(col.NUM_PARTNERS,
                            population.get_sub_pop_intersection(
                                HIV_positive_subpop,
                                population.get_sub_pop([(col.VIRAL_LOAD_GROUP, op.eq, vg)])
                            )))/n_stp_of_infected for vg in range(6)]

    def set_primary_infection(self, population: Population):
        # Update primary infection status
        past_primary_infection = population.get_sub_pop(
            [(col.DATE_HIV_INFECTION, op.le, population.date - timedelta(days=90))])
        population.set_present_variable(col.IN_PRIMARY_INFECTION, False, past_primary_infection)

    def set_viral_load_groups(self, population: Population):
        HIV_positive_pop = population.get_sub_pop(COND(col.HIV_STATUS, op.eq, True))
        population.set_present_variable(col.VIRAL_LOAD_GROUP,
                                        np.digitize(population.get_variable(col.VIRAL_LOAD, HIV_positive_pop),
                                                    np.array([2.7, 3.7, 4.7, 5.7])),
                                        HIV_positive_pop)

    def init_resistance_mutations(self, population: Population):
        """
        Initialise drug resistance mutations at the start of the simulation to False.
        """
        population.init_variable(col.TA_MUTATION, False)
        population.init_variable(col.M184_MUTATION, False)
        population.init_variable(col.K65_MUTATION, False)
        population.init_variable(col.Q151_MUTATION, False)
        population.init_variable(col.K103_MUTATION, False)
        population.init_variable(col.Y181_MUTATION, False)
        population.init_variable(col.G190_MUTATION, False)
        population.init_variable(col.P32_MUTATION, False)
        population.init_variable(col.P33_MUTATION, False)
        population.init_variable(col.P46_MUTATION, False)
        population.init_variable(col.P47_MUTATION, False)
        population.init_variable(col.P50L_MUTATION, False)
        population.init_variable(col.P50V_MUTATION, False)
        population.init_variable(col.P54_MUTATION, False)
        population.init_variable(col.P76_MUTATION, False)
        population.init_variable(col.P82_MUTATION, False)
        population.init_variable(col.P84_MUTATION, False)
        population.init_variable(col.P88_MUTATION, False)
        population.init_variable(col.P90_MUTATION, False)
        population.init_variable(col.IN118_MUTATION, False)
        population.init_variable(col.IN140_MUTATION, False)
        population.init_variable(col.IN148_MUTATION, False)
        population.init_variable(col.IN155_MUTATION, False)
        population.init_variable(col.IN263_MUTATION, False)

    def stp_HIV_transmission(self, person):
        """
        Returns True if HIV transmission occurs, and False otherwise.
        """
        HIV_probabilities = np.array([self.ratio_infected_stp[opposite_sex(
            person[col.SEX])][age_group] for age_group in person[col.STP_AGE_GROUPS]])

        infection = False
        for partner in range(person[col.NUM_PARTNERS]):
            if (rng.random() < HIV_probabilities[partner]):
                stp_viral_group = rng.choice(6, p=self.ratio_vl_stp[opposite_sex(person[col.SEX])]
                                                                   [person[col.STP_AGE_GROUPS][partner]])
                viral_transmission_probability = max(0, rng.normal(self.stp_transmission_means[stp_viral_group],
                                                                   self.stp_transmission_sigmas[stp_viral_group]))
                if person[col.SEX] is SexType.Female:
                    if person[col.AGE] < 20:
                        viral_transmission_probability = (viral_transmission_probability
                                                          * self.young_women_transmission_factor)
                    else:
                        viral_transmission_probability = (viral_transmission_probability
                                                          * self.women_transmission_factor)
                elif person[col.CIRCUMCISED]:
                    viral_transmission_probability = (viral_transmission_probability
                                                      * self.circumcision_risk_reduction)

                if person[col.STI]:
                    viral_transmission_probability = viral_transmission_probability * self.sti_transmission_factor

                if (rng.random() < viral_transmission_probability):
                    # TODO: Superinfection, PREP, etc.
                    infection = True
                    self.output.infected_stp += 1
                    # in primary infection
                    if stp_viral_group == 5:
                        self.output.infected_primary_infection += 1
                    break

        return infection

    def ltp_transmission(self, population: Population):
        """
        Sets population set for which HIV transmission occurs from a long term partner.
        Considers monogamous and non-monogamous relationships.
        """
        num_neg_women_with_infected_parter = len(population.get_sub_pop([(col.SEX, op.eq, SexType.Female),
                                                                         (col.HIV_STATUS, op.eq, False),
                                                                         (col.LONG_TERM_PARTNER, op.eq, True),
                                                                         (col.LTP_STATUS, op.eq, True)]))
        num_pos_men_with_negative_partner = len(population.get_sub_pop([(col.SEX, op.eq, SexType.Male),
                                                                        (col.HIV_STATUS, op.eq, True),
                                                                        (col.LONG_TERM_PARTNER, op.eq, True),
                                                                        (col.LTP_STATUS, op.eq, False)]))

        num_neg_men_with_infected_parter = len(population.get_sub_pop([(col.SEX, op.eq, SexType.Male),
                                                                       (col.HIV_STATUS, op.eq, False),
                                                                       (col.LONG_TERM_PARTNER, op.eq, True),
                                                                       (col.LTP_STATUS, op.eq, True)]))
        num_pos_women_with_negative_partner = len(population.get_sub_pop([(col.SEX, op.eq, SexType.Female),
                                                                          (col.HIV_STATUS, op.eq, True),
                                                                          (col.LONG_TERM_PARTNER, op.eq, True),
                                                                          (col.LTP_STATUS, op.eq, False)]))

        ltp_hiv_status_difference_women = num_neg_women_with_infected_parter - num_pos_men_with_negative_partner
        ltp_hiv_status_difference_men = num_neg_men_with_infected_parter - num_pos_women_with_negative_partner

        def calculate_incidence_factor(delta_hiv_ltp):
            incidence_factor = 1
            boundaries = np.array([-5000, -2000, -500, -200, -75, -20])   # TODO / 100000 * population.size
            multiplier = [abs(delta_hiv_ltp)/3, abs(delta_hiv_ltp)/50,
                          abs(delta_hiv_ltp)/100, abs(delta_hiv_ltp)/100,  3.5, 2.5]
            for i in range(6):
                if delta_hiv_ltp < boundaries[i]:
                    incidence_factor = incidence_factor * multiplier[i]
                    break

            return incidence_factor

        self.incidence_factor = {SexType.Male: calculate_incidence_factor(ltp_hiv_status_difference_men),
                                 SexType.Female: calculate_incidence_factor(ltp_hiv_status_difference_women)}

        for sex in [SexType.Male, SexType.Female]:
            for age_group in range(5):
                this_sex_with_ltp = population.get_sub_pop([(col.SEX, op.eq, sex),
                                                            (col.AGE_GROUP, op.eq, age_group),
                                                            (col.LONG_TERM_PARTNER, op.eq, True)])
                op_sex_with_ltp = population.get_sub_pop([(col.SEX, op.eq, opposite_sex(sex)),
                                                          (col.AGE_GROUP, op.eq, age_group),
                                                          (col.LONG_TERM_PARTNER, op.eq, True)])
                num_op_sex_with_ltp = len(op_sex_with_ltp)
                num_op_sex_monogamous = len(population.get_sub_pop([(col.SEX, op.eq, opposite_sex(sex)),
                                                                    (col.AGE_GROUP, op.eq, age_group),
                                                                    (col.LONG_TERM_PARTNER, op.eq, True),
                                                                    (col.NUM_PARTNERS, op.eq, 0)]))
                if num_op_sex_with_ltp == 0:
                    self.prop_monogamous[opposite_sex(sex)][age_group] = 0
                else:
                    self.prop_monogamous[opposite_sex(sex)][age_group] = num_op_sex_monogamous / num_op_sex_with_ltp

                partner_is_monogamous = rng.uniform(0, 1, len(this_sex_with_ltp)
                                                    ) < self.prop_monogamous[opposite_sex(sex)][age_group]

                population.set_present_variable(col.LTP_MONOGAMOUS, partner_is_monogamous, this_sex_with_ltp)

        # Non Monogamous Partner Case: partner is infected by another person
        def calculate_infected_ltp(sex, age_group, size):
            non_monogamous_partners = population.get_sub_pop([(col.NUM_PARTNERS, op.gt, 0),
                                                              (col.LONG_TERM_PARTNER, op.eq, True),
                                                              (col.SEX, op.eq, opposite_sex(sex)),
                                                              (col.AGE_GROUP, op.eq, age_group)])
            if len(non_monogamous_partners) != 0:
                people_in_primary = population.get_sub_pop([(col.IN_PRIMARY_INFECTION, op.eq, True)])
                non_monogamous_primary = population.get_sub_pop_intersection(non_monogamous_partners, people_in_primary)
                self.incidence[sex] = len(non_monogamous_primary) / len(non_monogamous_partners)
                ltp_infected = (rng.uniform(0, 1, size) / self.incidence_factor[sex]) < self.incidence[sex]
            else:
                ltp_infected = 0
            return ltp_infected

        people_with_nonmonogamous_ltp = population.get_sub_pop([(col.LONG_TERM_PARTNER, op.eq, True),
                                                                (col.LTP_MONOGAMOUS, op.eq, False)])
        partner_infected = population.transform_group([col.SEX, col.AGE_GROUP],
                                                      calculate_infected_ltp,
                                                      use_size=True,
                                                      sub_pop=people_with_nonmonogamous_ltp)
        population.set_present_variable(col.LTP_STATUS, partner_infected, people_with_nonmonogamous_ltp)

        # Monogamous Partner Case: subjects infect partners
        def calculate_infected_ltp_monogamous(sex, age_group, size):
            monogamous_people_hiv_pos = population.get_sub_pop([(col.SEX, op.eq, sex),
                                                                (col.AGE_GROUP, op.eq, age_group),
                                                                (col.LONG_TERM_PARTNER, op.eq, True),
                                                                (col.LTP_MONOGAMOUS, op.eq, True),
                                                                (col.HIV_STATUS, op.eq, True)])
            people_in_primary = population.get_sub_pop(
                [(col.IN_PRIMARY_INFECTION, op.eq, True)])  # FIXME previous timestep?
            monogamous_primary = population.get_sub_pop_intersection(monogamous_people_hiv_pos, people_in_primary)
            viral_load = population.get_variable(col.VIRAL_LOAD, monogamous_people_hiv_pos)  # FIXME previous timestep?
            people_with_sti = population.get_sub_pop([(col.STI, op.eq, True)])
            monogamous_people_with_sti = population.get_sub_pop_intersection(monogamous_people_hiv_pos, people_with_sti)

            viral_load_boundaries = [2.7, 3.7, 4.7, 5.7]
            ub1 = [0.001, 0.01, 0.03, 0.06]
            ub2 = [0.000025, 0.0025, 0.0075, 0.015]

            risk_ltp = np.zeros(population.size)

            r = rng.uniform(0, 1, len(monogamous_people_hiv_pos))
            fold_tr = np.full(len(monogamous_people_hiv_pos), 1)
            fold_tr[r < 0.33] = 0.67
            fold_tr[r > 0.67] = 1.5
            rng_values = rng.normal(0, 1, len(monogamous_people_hiv_pos))
            risk_ltp[monogamous_people_hiv_pos] = np.maximum(0, (0.10 * fold_tr + 0.25 * rng_values))
            for vl in range(len(viral_load_boundaries)):
                vl_mask = viral_load[monogamous_people_hiv_pos] < viral_load_boundaries[vl]
                risk_ltp[monogamous_people_hiv_pos] = np.where(
                                                                vl_mask.any(),
                                                                np.maximum(0, ub1[vl] * fold_tr + ub2[vl] * rng_values),
                                                                risk_ltp[monogamous_people_hiv_pos])
                if vl_mask.any():
                    break

            if len(monogamous_primary) > 0:
                risk_ltp[monogamous_primary] = np.maximum(0, [0.16 + 0.075 * rng.normal(0, 1, len(monogamous_primary))])

            # higher transmission risk in women
            if sex == SexType.Male:
                r = rng.uniform(0, 1, len(monogamous_people_hiv_pos))
                fold_change_w = np.full(len(monogamous_people_hiv_pos), 1.5)
                fold_change_w[r < 0.05] = 1
                fold_change_w[r >= 0.30] = 2
                if age_group == 0:
                    fold_change_w[(0.33 <= r) & (r <= 0.67)] *= 3
                    fold_change_w[r > 0.67] *= 5

                risk_ltp[monogamous_people_hiv_pos] *= fold_change_w

            # higher transmission risk in people with STI
            if len(monogamous_people_with_sti) > 0:
                r = rng.uniform(0, 1, len(monogamous_people_with_sti))
                fold_change_sti = np.full(len(monogamous_people_with_sti), 3)
                fold_change_sti[r < 0.333] = 2
                fold_change_sti[r > 0.67] = 5

                risk_ltp[monogamous_people_with_sti] *= fold_change_sti

            ltp_infected = rng.uniform(0, 1, size) < risk_ltp[monogamous_people_hiv_pos]

            return ltp_infected

        people_with_monogamous_ltp = population.get_sub_pop([(col.LONG_TERM_PARTNER, op.eq, True),
                                                             (col.LTP_MONOGAMOUS, op.eq, True)])
        partner_infected = population.transform_group([col.SEX, col.AGE_GROUP],
                                                      calculate_infected_ltp_monogamous,
                                                      use_size=True,
                                                      sub_pop=people_with_monogamous_ltp)
        population.set_present_variable(col.LTP_STATUS, partner_infected, people_with_monogamous_ltp)

    def update_HIV_status(self, population: Population):
        """
        Update HIV status for new transmissions in the last time period.
        Super simple model where probability of being infected by a given person
        is prevalence times transmission risk (P x r).
        Probability of each new partner not infecting you then is (1-Pr),
        and then prob of n partners independently not infecting you is (1-Pr)**n,
        so probability of infection is 1-((1-Pr)**n).
        """
        self.update_partner_risk_vectors(population)
        # select uninfected people that have at least one short-term partner
        HIV_neg_active_pop = population.get_sub_pop([(col.HIV_STATUS, op.eq, False),
                                                     (col.NUM_PARTNERS, op.gt, 0)])

        # Get people who already have HIV prior to transmission (for updating their progression)
        initial_HIV_pos = population.get_sub_pop([(col.HIV_STATUS, op.eq, True)])

        # TODO: Add ltp HIV transmission
        # Determine HIV status after transmission
        new_HIV_status = population.apply_function(self.stp_HIV_transmission, 1, HIV_neg_active_pop)
        # Apply HIV status to sub-population
        population.set_present_variable(col.HIV_STATUS,
                                        new_HIV_status,
                                        HIV_neg_active_pop)
        newly_infected = population.get_sub_pop([(col.HIV_STATUS, op.eq, True),
                                                 (col.DATE_HIV_INFECTION, op.eq, None)])

        self.initialise_HIV_progression(population, newly_infected)
        self.update_HIV_progression(population, initial_HIV_pos)

    def initialise_HIV_progression(self, population: Population, newly_infected):
        """
        Sets the initial viral load and CD4 counts for people with newly acquired HIV.
        Depends on sex and age (not age group). CD4 counts depend on viral load, so is
        necessarily calculated second.
        Sets the date of HIV infection.
        """
        population.set_present_variable(col.DATE_HIV_INFECTION, population.date, newly_infected)
        population.set_present_variable(col.IN_PRIMARY_INFECTION, True, newly_infected)

        def set_initial_viral_load(person):
            initial_vl = rng.normal(4.075, 0.5) + (person[col.AGE] - 35)*0.005
            if person[col.SEX] == SexType.Female:
                initial_vl = initial_vl - 0.2
            initial_vl = min(6.5, initial_vl)
            return initial_vl

        population.set_present_variable(col.VIRAL_LOAD,
                                        population.apply_function(set_initial_viral_load, 1, sub_pop=newly_infected),
                                        sub_pop=newly_infected)

        def set_initial_CD4(person):
            sqrt_cd4 = self.initial_mean_sqrt_cd4 - (1.5 * person[col.VIRAL_LOAD]) + rng.normal(0, 2) \
                - (person[col.AGE] - 35)*0.05
            upper_sqrt_cd4 = np.sqrt(1500)
            lower_sqrt_cd4 = 18
            sqrt_cd4 = min(upper_sqrt_cd4, max(sqrt_cd4, lower_sqrt_cd4))  # clamp sqrt_cd4 to be in limits
            return sqrt_cd4**2

        population.set_present_variable(col.CD4,
                                        population.apply_function(set_initial_CD4, 1, sub_pop=newly_infected),
                                        sub_pop=newly_infected)

    def update_HIV_progression(self, population: Population, HIV_subpop):
        """
        Updates HIV related variables viral load & CD4 count
        """

        # For people who are not on treatment
        # Viral Load
        art_naive_pop = population.get_sub_pop_intersection(
            HIV_subpop, population.get_sub_pop([(col.ART_NAIVE, op.eq, True)]))
        ages = population.get_variable(col.AGE, art_naive_pop)
        delta_vl = self.vl_base_change*0.02275 + (0.05 * rng.normal(size=len(ages))) + (ages - 35)*0.00075
        prev_vl = population.get_variable(col.VIRAL_LOAD, art_naive_pop)

        population.set_present_variable(col.VIRAL_LOAD, prev_vl + delta_vl, art_naive_pop)
        high_vl = population.get_sub_pop_intersection(
            art_naive_pop, population.get_sub_pop([(col.VIRAL_LOAD, op.gt, 6.5)]))
        population.set_present_variable(col.VIRAL_LOAD, 6.5, high_vl)

        # CD4 count
        vl_lims = np.array([3, 3.5, 4., 4.5, 5., 5.5, 6])
        vl_groups = np.digitize(prev_vl, vl_lims)
        vl_group_factors = np.array([0.0, 0.022, 0.095, 0.4, 0.4, 0.85, 1.3, 1.75])
        x4 = population.get_variable(col.X4_VIRUS, art_naive_pop)
        delta_cd4_sqrt = vl_group_factors[vl_groups] * self.cd4_base_change \
            + rng.normal(size=len(vl_groups))*self.sigma_cd4 + x4*0.25
        prev_cd4 = population.get_variable(col.CD4, art_naive_pop)
        sqrt_cd4 = np.maximum(np.sqrt(prev_cd4) - delta_cd4_sqrt, 0)
        new_cd4 = sqrt_cd4**2
        population.set_present_variable(col.CD4, new_cd4, art_naive_pop)

        # TODO: people on treatment

    def HIV_related_disease_risk(self, pop: Population, time_step: timedelta):
        # TODO: does disease risk apply to everyone who is alive?
        # calculate disease base rate
        HIV_pos = pop.get_sub_pop(COND(col.HIV_STATUS, op.eq, True))
        cd4_risk_groups = np.digitize(pop.get_variable(col.CD4, HIV_pos), self.disease_cd4_boundaries)
        viral_load_risk_groups = np.digitize(pop.get_variable(col.VIRAL_LOAD, HIV_pos), self.disease_vl_boundaries)
        ages = pop.get_variable(col.AGE, HIV_pos)
        age_factor = (ages/38)**1.2
        base_rate = self.base_rate_disease[cd4_risk_groups] \
            * self.vl_disease_factor[viral_load_risk_groups] \
            * age_factor

        def disease_and_diagnosis(disease_col, diagnosis_col, disease_rate, diagnosis_prob):
            # Calculate occurence and diagnosis of given disease
            disease_risk = 1 - np.exp(-disease_rate * (time_step.month/12))
            r_disease = rng.uniform(size=len(HIV_pos))
            disease = r_disease < disease_risk
            diagnosis = r_disease < (disease_risk * diagnosis_prob)
            pop.set_present_variable(disease_col, disease, HIV_pos)
            pop.set_present_variable(diagnosis_col, diagnosis, HIV_pos)
            return (disease, diagnosis)

        # WHO stage 3 diseases
        non_tb_who3_rate = base_rate * self.who3_risk_factor * (1-self.who3_proportion_tb)
        tb_rate = base_rate * self.who3_risk_factor * self.who3_proportion_tb

        non_tb_who3_per_timestep = 1 - np.exp(-non_tb_who3_rate * (time_step.month/12))
        r_non_tb = rng.uniform(size=len(HIV_pos))
        who3_disease = r_non_tb < non_tb_who3_per_timestep
        pop.set_present_variable(col.NON_TB_WHO3, who3_disease, HIV_pos)

        # TB WHO3
        (tb, _) = disease_and_diagnosis(col.TB, col.TB_DIAGNOSED, tb_rate, self.tb_base_diagnosis_prob)
        pop.set_present_variable(col.WHO3_EVENT, (who3_disease | tb), HIV_pos)

        # cryptococcal meningitis
        (cm, _) = disease_and_diagnosis(col.C_MENINGITIS,
                                        col.C_MENINGITIS_DIAGNOSED,
                                        base_rate * self.prop_ADC_cryp_meningitis,
                                        self.CM_base_diagnosis_prob)

        # serious bacterial infection
        (sbi, _) = disease_and_diagnosis(col.SBI,
                                         col.SBI_DIAGNOSED,
                                         base_rate * self.prop_ADC_SBI,
                                         self.SBI_base_diagnosis_prob)

        # WHO stage 4
        (who4_other, _) = disease_and_diagnosis(col.WHO4_OTHER,
                                                col.WHO4_OTHER_DIAGNOSED,
                                                base_rate * self.prop_ADC_other,
                                                self.WHO4_base_diagnosis_prob)
        adc = (cm | sbi | who4_other)
        pop.set_present_variable(col.ADC, adc, HIV_pos)

        # DEATH
        # base death rate for HIV+ people
        HIV_death_rate = base_rate * self.hiv_mortality_factor  # hiv mortality factor
        # death rate for people with TB but no ADC
        tb_no_adc = (tb & (~adc))
        HIV_death_rate[tb_no_adc] = HIV_death_rate[tb_no_adc]*self.tb_mortality_factor
        # death rate for people with ADCs
        HIV_death_rate[cm] = HIV_death_rate[cm]*self.cm_mortality_factor
        HIV_death_rate[sbi] = HIV_death_rate[sbi]*self.sbi_mortality_factor
        HIV_death_rate[who4_other] = HIV_death_rate[who4_other]*self.other_adc_mortality_factor
        prob_death = 1 - np.exp(-HIV_death_rate * (time_step.month/12))
        r_death = rng.uniform(size=len(HIV_pos))
        HIV_deaths = r_death < prob_death
        self.output.record_HIV_deaths(pop, HIV_deaths)
        return HIV_deaths
