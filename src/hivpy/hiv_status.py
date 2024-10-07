from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .population import Population

import operator as op

import numpy as np
import pandas as pd

import hivpy.column_names as col

from . import output
from .common import (AND, COND, SexType, opposite_sex, rng, safe_ratio,
                     timedelta)


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
        # Ratio of non monogamous people in primary infection
        self.ratio_non_monogamous_primary = {SexType.Male: np.zeros(7),
                                             SexType.Female: np.zeros(7)}
        # proportion of stps with different viral load groups in general population for each sex and age group
        self.ratio_vl_stp = {SexType.Male: [np.zeros(6)]*5,
                             SexType.Female: [np.zeros(6)]*5}
        # proportion of monogamous partners in general population for each sex and age group
        self.prop_monogamous = {SexType.Male: np.zeros(5),
                                SexType.Female: np.zeros(5)}
        self.prevalence = {SexType.Male: np.zeros(5),
                           SexType.Female: np.zeros(5)}
        self.incidence_factor = {SexType.Male: 1,
                                 SexType.Female: 1}
        self.incidence = {SexType.Male: 0,
                          SexType.Female: 0}
        self.transmission_rate_means = [self.tr_rate_undetectable_vl, 0.01, 0.03, 0.06, 0.1, self.tr_rate_primary]
        self.transmission_rate_sigmas = [0.000025, 0.0025, 0.0075, 0.015, 0.025, 0.075]
        self.women_transmission_factor = rng.choice([1., 1.5, 2.], p=[0.05, 0.25, 0.7])
        self.young_women_transmission_factor = rng.choice([1., 3., 5.]) * self.women_transmission_factor
        self.sti_transmission_factor = rng.choice([2., 3.])
        self.stp_transmission_means = self.transmission_factor * self.stp_transmission_factor * \
            np.array([0, self.tr_rate_undetectable_vl / self.transmission_factor, 0.01,
                      0.03, 0.06, 0.1, self.tr_rate_primary])
        self.stp_transmission_sigmas = np.array(
            [0, 0.000025, 0.0025, 0.0075, 0.015, 0.025, 0.075])
        self.circumcision_risk_reduction = 0.4  # reduce infection risk by 60%
        self.vl_base_change = rng.choice([1.0, 1.5, 2.0])  # TODO: move to data file
        self.cd4_base_change = rng.choice([0.7, 0.85, 1.0, 1/0.85, 1/0.7])  # TODO: move to data file
        self.resistance_mutations_prop_vlg = np.zeros(6)  # TODO: resistance mutations per person/ viral load group

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

        self.diagnosis_rate = {SexType.Male: 0.0, SexType.Female: 0.0}
        self.ltp_diagnosis_rate = {SexType.Male: 0.0, SexType.Female: 0.0}
        self.prob_repeated_ltp = 0.5
        self.prob_ltp_remain_suppressed = 0.97
        self.prob_ltp_continue_ART = 0.98

    # Initialisation ----------------------------------------------------------------------------------

    def init_HIV_variables(self, population: Population):
        # Personal HIV status / progression
        population.init_variable(col.HIV_STATUS, False)
        population.init_variable(col.RISK_LTP_INFECTED, 0.0)
        population.init_variable(col.DATE_HIV_INFECTION, None)
        population.init_variable(col.IN_PRIMARY_INFECTION, False)
        population.init_variable(col.HIV_INFECTION_GE6M, False)  # FIXME: DUMMY variable
        population.init_variable(col.RESISTANCE_MUTATIONS, 0)
        population.init_variable(col.CD4, 0.0)
        population.init_variable(col.MAX_CD4, 6.6 + rng.normal(0, 0.25, size=population.size))
        population.init_variable(col.HIV_DIAGNOSED, False)
        population.init_variable(col.HIV_DIAGNOSIS_DATE, None)
        population.init_variable(col.UNDER_CARE, False)
        population.init_variable(col.VIRAL_LOAD_GROUP, None)
        population.init_variable(col.VIRAL_LOAD, 0.0)
        population.init_variable(col.VIRAL_SUPPRESSION, False)
        population.init_variable(col.X4_VIRUS, False)

        # TODO: move to ART module
        population.init_variable(col.ON_ART, False)

        # Long term partners
        population.init_variable(col.LTP_STATUS, False)
        population.init_variable(col.RECENT_LTP_STATUS, False)
        population.init_variable(col.LTP_DIAGNOSED, False)
        population.init_variable(col.RECENT_LTP_DIAGNOSED, False)
        population.init_variable(col.LTP_IN_PRIMARY, False)
        population.init_variable(col.LTP_MONOGAMOUS, False)
        population.init_variable(col.LTP_INFECTION_DATE, None)
        population.init_variable(col.LTP_ART, False)
        population.init_variable(col.RECENT_LTP_ART, False)
        population.init_variable(col.LTP_VIRAL_SUPPRESSED, False)

        # Disease
        population.init_variable(col.WHO3_EVENT, False)
        population.init_variable(col.NON_TB_WHO3, False)
        population.init_variable(col.TB, False)
        population.init_variable(col.TB_DIAGNOSED, False)
        population.init_variable(col.TB_INFECTION_DATE, None)
        population.init_variable(col.TB_INITIAL_INFECTION, False)
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

    # Updating Statistics -----------------------------------------------------------------------------
    # These functions all need to be called each time step in order to use the HIV module -------------

    def update_HIV_statistics(self, population: Population):
        """
        Updates all relevant population statistics for this module to run correctly
        """
        self.update_partner_risk_vectors(population)
        self.update_LTP_risk_vectors(population)
        self.update_diagnosis_stats(population)
        self.update_viral_suppression_stats(population)
        self.update_art_stats(population)

    def update_HIV_prevalence(self, population):
        for sex in [SexType.Male, SexType.Female]:
            for age_group in range(5):
                opposite_sex = population.get_sub_pop([(col.SEX, op.ne, sex),
                                                       (col.AGE_GROUP, op.eq, age_group)])
                opposite_sex_with_hiv = population.get_sub_pop([(col.SEX, op.ne, sex),
                                                                (col.AGE_GROUP, op.eq, age_group),
                                                                (col.HIV_STATUS, op.eq, True)])

                if len(opposite_sex) != 0:
                    self.prevalence[sex][age_group] = len(opposite_sex_with_hiv) / len(opposite_sex)

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

    def update_LTP_risk_vectors(self, population: Population):
        """
        Calculate risk factors such as non monogamous incidence for different sex and age groups
        """
        primary_population = population.get_sub_pop(COND(col.IN_PRIMARY_INFECTION, op.eq, True))
        for sex in SexType:
            for age_group in range(1, 6):  # 5 groups from 15-25 up to 55-65
                non_monogamous = population.get_sub_pop(AND(COND(col.LTP_AGE_GROUP, op.eq, age_group),
                                                            COND(col.SEX, op.eq, sex),
                                                            COND(col.LONG_TERM_PARTNER, op.eq, True),
                                                            COND(col.NUM_PARTNERS, op.gt, 0)))
                non_monogamous_pos = population.get_sub_pop_intersection(non_monogamous, primary_population)
                num_non_monogamous = len(non_monogamous)
                if (num_non_monogamous == 0):
                    self.ratio_non_monogamous_primary[sex][age_group] = 0
                else:
                    self.ratio_non_monogamous_primary[sex][age_group] = len(non_monogamous_pos) / num_non_monogamous

    def update_diagnosis_stats(self, population: Population):
        people_with_hiv = population.get_sub_pop(COND(col.HIV_STATUS, op.eq, True))
        people_diagnosed = population.get_sub_pop(COND(col.HIV_DIAGNOSED, op.eq, True))

        pop_by_sex = {SexType.Male: population.get_sub_pop(COND(col.SEX, op.eq, SexType.Male)),
                      SexType.Female: population.get_sub_pop(COND(col.SEX, op.eq, SexType.Female))}

        people_with_infected_ltp = population.get_sub_pop(AND(COND(col.LONG_TERM_PARTNER, op.eq, True),
                                                              COND(col.LTP_STATUS, op.eq, True)))
        ltp_diagnosed = population.get_sub_pop_intersection(
            people_with_infected_ltp,
            population.get_sub_pop(COND(col.LTP_DIAGNOSED, op.eq, True)))

        for sex in SexType:
            other_sex = opposite_sex(sex)
            self.diagnosis_rate[sex] = safe_ratio(len(population.get_sub_pop_intersection(people_diagnosed, pop_by_sex[sex])),
                                                  len(population.get_sub_pop_intersection(people_with_hiv, pop_by_sex[sex])))
            self.ltp_diagnosis_rate[sex] = safe_ratio(len(population.get_sub_pop_intersection(ltp_diagnosed, pop_by_sex[other_sex])),
                                                      len(population.get_sub_pop_intersection(people_with_infected_ltp, pop_by_sex[other_sex])))

    def update_viral_suppression_stats(self, population):
        num_viral_suppressed = len(population.get_sub_pop(AND(COND(col.AGE, op.ge, 15),
                                                              COND(col.AGE, op.lt, 65),
                                                              COND(col.ON_ART, op.eq, True),
                                                              COND(col.VIRAL_SUPPRESSION, op.eq, True))))
        num_on_art = len(population.get_sub_pop(AND(COND(col.AGE, op.ge, 15),
                                                    COND(col.AGE, op.lt, 65),
                                                    COND(col.ON_ART, op.eq, True))))
        self.proportion_viral_suppressed = safe_ratio(num_viral_suppressed, num_on_art)

        num_ltp_viral_suppressed = len(population.get_sub_pop(AND(COND(col.AGE, op.ge, 15),
                                                                  COND(col.AGE, op.lt, 65),
                                                                  COND(col.LTP_ART, op.eq, True),
                                                                  COND(col.LTP_VIRAL_SUPPRESSED, op.eq, True))))
        num_ltp_on_art = len(population.get_sub_pop(AND(COND(col.AGE, op.ge, 15),
                                                        COND(col.AGE, op.lt, 65),
                                                        COND(col.LTP_ART, op.eq, True))))
        self.proportion_ltp_viral_suppressed = safe_ratio(num_ltp_viral_suppressed, num_ltp_on_art)
        self.diff_proportion_viral_suppressed = self.proportion_viral_suppressed - self.proportion_ltp_viral_suppressed

    # TODO: Probably move to ART module when it is ready
    def update_art_stats(self, population):
        num_diagnosed_on_art = len(population.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, True),
                                                              COND(col.ON_ART, op.eq, True),
                                                              COND(col.AGE, op.ge, 15),
                                                              COND(col.AGE, op.lt, 65))))
        num_diagnosed = len(population.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, True),
                                                       COND(col.AGE, op.ge, 15),
                                                       COND(col.AGE, op.lt, 65))))
        self.proportion_diagnosed_on_art = safe_ratio(num_diagnosed_on_art, num_diagnosed)

        num_ltp_diagnosed_on_art = len(population.get_sub_pop(AND(COND(col.LTP_DIAGNOSED, op.eq, True),
                                                                  COND(col.LTP_ART, op.eq, True),
                                                                  COND(col.AGE, op.ge, 15),
                                                                  COND(col.AGE, op.lt, 65))))
        num_ltp_diagnosed = len(population.get_sub_pop(AND(COND(col.LTP_DIAGNOSED, op.eq, True),
                                                           COND(col.AGE, op.ge, 15),
                                                           COND(col.AGE, op.lt, 65))))
        self.proportion_LTP_diagnosed_on_art = safe_ratio(num_ltp_diagnosed_on_art, num_ltp_diagnosed)

        self.diff_proportion_on_art = self.proportion_diagnosed_on_art - self.proportion_LTP_diagnosed_on_art

    # Short Term Partner Transmission -----------------------------------------------------------------

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

    # Long Term Partner Transmission and Progression --------------------------------------------------

    def update_ltp_HIV(self, population: Population):
        self.update_HIV_statistics(population)
        self.ltp_acquiring_HIV(population)
        self.diagnose_and_treat_ltp(population)
        self.set_infection_from_infected_ltp(population)
        self.set_new_ltp_already_infected(population)

    def set_ltp_age_groups(self, population: Population):
        age_groups = np.digitize(population.get_variable(col.AGE), [15, 25, 35, 45, 55, 65])
        population.set_present_variable(col.LTP_AGE_GROUP, age_groups)

    def ltp_acquiring_HIV(self, population: Population):
        """
        Sets HIV status for existing long-term partners.
        LTP can be infected by the "subject" (person in the population) if LTP is monogamous.
        LTP can be infected by another person if the LTP is non-monogamous.
        Needs to be called after update_LTP_risk_vectors.
        """

        # Fractional differences in number of serodiscordant couples based on sex of HIV negative partner
        ltp_hiv_status_difference_neg_women = self.get_hiv_status_difference(SexType.Female, population) \
            / population.size
        ltp_hiv_status_difference_neg_men = self.get_hiv_status_difference(SexType.Male, population) \
            / population.size

        def calculate_incidence_factor(delta_hiv_ltp):
            incidence_factor = 1
            boundaries = np.array([-0.05, -0.02, -0.005, -0.002, -0.00075, -0.0002])
            multiplier = [abs(delta_hiv_ltp)/3, abs(delta_hiv_ltp)/50,
                          abs(delta_hiv_ltp)/100, abs(delta_hiv_ltp)/100,  3.5, 2.5]
            for i in range(6):
                if delta_hiv_ltp < boundaries[i]:
                    incidence_factor = multiplier[i]
                    break

            return incidence_factor

        self.incidence_factor = {SexType.Male: calculate_incidence_factor(ltp_hiv_status_difference_neg_men),
                                 SexType.Female: calculate_incidence_factor(ltp_hiv_status_difference_neg_women)}

        self.set_ltp_age_groups(population)

        self.set_monogamous_ltp(population)

        # Non Monogamous Partner Case: partner is infected by another person
        def calculate_infected_ltp(sex, age_group, size):
            ltp_infected = (rng.uniform(0, 1, size) / self.incidence_factor[opposite_sex(sex)]) \
                < self.ratio_non_monogamous_primary[sex][age_group]
            return ltp_infected

        people_with_nonmonogamous_ltp = population.get_sub_pop([(col.LONG_TERM_PARTNER, op.eq, True),
                                                                (col.LTP_MONOGAMOUS, op.eq, False),
                                                                (col.LTP_STATUS, op.eq, False)])
        partner_infected = population.transform_group([col.SEX, col.LTP_AGE_GROUP],
                                                      calculate_infected_ltp,
                                                      use_size=True,
                                                      sub_pop=people_with_nonmonogamous_ltp)
        population.set_present_variable(col.LTP_STATUS, partner_infected, people_with_nonmonogamous_ltp)

        # Monogamous Partner Case: subjects infect partners
        def calculate_infected_ltp_monogamous(sex, age_group, vl_group, sti, size):
            risk_to_ltp = rng.normal(
                self.transmission_rate_means[vl_group],
                self.transmission_rate_sigmas[vl_group],
                size=size
            )
            if (sex == SexType.Male):  # male subject means female partner
                if (age_group == 1):  # group 1 is 15 <= age < 25
                    risk_to_ltp *= self.young_women_transmission_factor
                else:
                    risk_to_ltp *= self.women_transmission_factor

            if sti:
                risk_to_ltp *= self.sti_transmission_factor

            return rng.uniform(0, 1, size=size) < risk_to_ltp

        people_with_monogamous_ltp_and_hiv = population.get_sub_pop([(col.LONG_TERM_PARTNER, op.eq, True),
                                                                    (col.LTP_MONOGAMOUS, op.eq, True),
                                                                    (col.HIV_STATUS, op.eq, True)])
        partner_infected = population.transform_group([col.SEX, col.LTP_AGE_GROUP, col.VIRAL_LOAD_GROUP, col.STI],
                                                      calculate_infected_ltp_monogamous,
                                                      use_size=True,
                                                      sub_pop=people_with_monogamous_ltp_and_hiv)
        population.set_present_variable(col.LTP_STATUS, partner_infected, people_with_monogamous_ltp_and_hiv)
        # TODO: record ltp infections for output

        # TODO: balance case where both are HIV+ (SAS 4563)
        # balancing of number of males and females in HIV-concordant couples
        # if there is an imbalance, a random set of people of that sex have their ltp status reset
        males_in_concordant = population.get_sub_pop([(col.HIV_STATUS, op.eq, True),
                                                      (col.LONG_TERM_PARTNER, op.eq, True),
                                                      (col.LTP_STATUS, op.eq, True),
                                                      (col.SEX, op.eq, SexType.Male)])
        females_in_concordant = population.get_sub_pop([(col.HIV_STATUS, op.eq, True),
                                                        (col.LONG_TERM_PARTNER, op.eq, True),
                                                        (col.LTP_STATUS, op.eq, True),
                                                        (col.SEX, op.eq, SexType.Female)])
        if (len(males_in_concordant) > 0 and len(females_in_concordant) > 0):
            ratio_concordance = len(females_in_concordant) / len(males_in_concordant)
            if (ratio_concordance > 1):
                random_concordant_females = \
                    population.get_sub_pop_from_array(rng.random(len(females_in_concordant))
                                                      > (1 / ratio_concordance), females_in_concordant)
                self.reset_ltp_status(population, random_concordant_females)
            elif (ratio_concordance < 1):
                random_concordant_males = \
                    population.get_sub_pop_from_array(rng.random(len(males_in_concordant))
                                                      > (ratio_concordance), males_in_concordant)
                self.reset_ltp_status(population, random_concordant_males)

    def set_monogamous_ltp(self, population):
        for sex in [SexType.Male, SexType.Female]:
            for age_group in range(5):
                this_sex_with_ltp = population.get_sub_pop([(col.SEX, op.eq, sex),
                                                            (col.LTP_AGE_GROUP, op.eq, age_group),
                                                            (col.LONG_TERM_PARTNER, op.eq, True)])
                op_sex_with_ltp = population.get_sub_pop([(col.SEX, op.eq, opposite_sex(sex)),
                                                          (col.LTP_AGE_GROUP, op.eq, age_group),
                                                          (col.LONG_TERM_PARTNER, op.eq, True)])
                num_op_sex_with_ltp = len(op_sex_with_ltp)
                num_op_sex_monogamous = len(population.get_sub_pop([(col.SEX, op.eq, opposite_sex(sex)),
                                                                    (col.LTP_AGE_GROUP, op.eq, age_group),
                                                                    (col.LONG_TERM_PARTNER, op.eq, True),
                                                                    (col.NUM_PARTNERS, op.eq, 0)]))
                if num_op_sex_with_ltp == 0:
                    self.prop_monogamous[opposite_sex(sex)][age_group] = 0
                else:
                    self.prop_monogamous[opposite_sex(sex)][age_group] = num_op_sex_monogamous / num_op_sex_with_ltp

                partner_is_monogamous = rng.uniform(0, 1, len(this_sex_with_ltp)
                                                    ) < self.prop_monogamous[opposite_sex(sex)][age_group]

                population.set_present_variable(col.LTP_MONOGAMOUS, partner_is_monogamous, this_sex_with_ltp)

    def diagnose_and_treat_ltp(self, population: Population):
        """
        Set diagnosis, viral supression, and ART status for long term partners
        """
        # Address inbalances in proportion of people of each sex who are diagnosed, and the proportion of
        # LTPs of each sex who are diagnosed (adjusts LTP_DIAGNOSED)
        ltp_undiagnosed = population.get_sub_pop(AND(COND(col.LONG_TERM_PARTNER, op.eq, True),
                                                     COND(col.LTP_STATUS, op.eq, True),
                                                     COND(col.LTP_DIAGNOSED, op.eq, False)))
        self.diagnose_ltp(population, ltp_undiagnosed)

        # ART for those diagnosed
        ltp_on_ART = population.get_sub_pop(COND(col.LTP_ART, op.eq, True))
        continuing_ART = self.get_ltps_continuing_art(ltp_on_ART)

        ltp_off_ART = population.get_sub_pop(AND(COND(col.LTP_DIAGNOSED, op.eq, True),
                                                 COND(col.LTP_ART, op.eq, False)))
        starting_ART = self.get_ltps_starting_art(ltp_off_ART)

        # Update ART statuses
        population.set_present_variable(col.LTP_ART, continuing_ART, ltp_on_ART)
        population.set_present_variable(col.LTP_ART, starting_ART, ltp_off_ART)

        # Viral load suppression in LTP
        viral_suppressed_ltp = population.get_sub_pop(COND(col.LTP_VIRAL_SUPPRESSED, op.eq, True))
        viral_unsuppressed_ltp = population.get_sub_pop(COND(col.LTP_VIRAL_SUPPRESSED, op.eq, False))

        # 3% chance that virally suppressed person becomes un-suppressed
        remaining_suppressed = rng.uniform(size=len(viral_suppressed_ltp)) < self.prob_ltp_remain_suppressed

        # chance of becoming virally suppressed if not previously suppressed
        becoming_suppressed = self.get_ltp_becoming_suppressed(viral_unsuppressed_ltp)

        # Update viral suppression
        population.set_present_variable(col.LTP_VIRAL_SUPPRESSED, remaining_suppressed, viral_suppressed_ltp)
        population.set_present_variable(col.LTP_VIRAL_SUPPRESSED, becoming_suppressed, viral_unsuppressed_ltp)

    def get_ltp_becoming_suppressed(self, viral_unsuppressed_ltp):
        prob_viral_suppressed = 0
        if (self.diff_proportion_viral_suppressed > 0 and self.diff_proportion_viral_suppressed < 0.05):
            prob_viral_suppressed = self.proportion_viral_suppressed * 0.2
        elif (self.diff_proportion_viral_suppressed < 0.1):
            prob_viral_suppressed = self.proportion_viral_suppressed * 0.5
        else:
            prob_viral_suppressed = self.proportion_viral_suppressed
        becoming_suppressed = rng.uniform(size=len(viral_unsuppressed_ltp)) < prob_viral_suppressed
        return becoming_suppressed

    def get_ltps_continuing_art(self, ltp_on_ART):
        # 2% chance that LTP on ART go off ART
        continuing_ART = rng.uniform(size=len(ltp_on_ART)) < self.prob_ltp_continue_ART
        return continuing_ART

    def get_ltps_starting_art(self, ltp_off_ART):
        prob_start_ART = 0
        if (self.diff_proportion_on_art > 0 and self.diff_proportion_on_art < 0.05):
            prob_start_ART = self.proportion_diagnosed_on_art * 0.2
        elif (self.diff_proportion_on_art < 0.1):
            prob_start_ART = self.proportion_diagnosed_on_art * 0.5
        elif (self.proportion_diagnosed_on_art < 0.95):
            prob_start_ART = self.proportion_diagnosed_on_art
        else:
            prob_start_ART = 1

        # TODO: Update with ART intro date when there is an ART module in place
        starting_ART = rng.uniform(size=len(ltp_off_ART)) < prob_start_ART
        return starting_ART

    def diagnose_ltp(self, population, ltp_undiagnosed_subpop):
        for sex in SexType:
            proportion_diagnosed = self.diagnosis_rate[sex]
            diagnosis_discrepency = proportion_diagnosed - self.ltp_diagnosis_rate[sex]
            prob_diagnosis_correction = 0
            if (0 <= diagnosis_discrepency < 0.5):
                prob_diagnosis_correction = proportion_diagnosed / 5
            elif (0.5 <= diagnosis_discrepency < 0.1):
                prob_diagnosis_correction = proportion_diagnosed / 2
            elif (0.1 <= diagnosis_discrepency):
                prob_diagnosis_correction = proportion_diagnosed
            ltp_undiagnosed_by_sex = population.get_sub_pop_intersection(ltp_undiagnosed_subpop,
                                                                         population.get_sub_pop(COND(col.SEX, op.eq, sex)))
            random_mask = rng.uniform(size=len(ltp_undiagnosed_by_sex)) < prob_diagnosis_correction
            random_subset = population.get_sub_pop_from_array(random_mask, ltp_undiagnosed_by_sex)
            population.set_present_variable(col.LTP_DIAGNOSED, True, random_subset)

    def reset_ltp_status(self, population, sub_pop):
        """
        Helper function; resets LONG_TERM_PARTNER, LTP_STATUS, and LTP_MONOGAMOUS to false
        """
        population.set_present_variable(col.LONG_TERM_PARTNER, False, sub_pop)
        population.set_present_variable(col.LTP_STATUS, False, sub_pop)
        population.set_present_variable(col.LTP_MONOGAMOUS, False, sub_pop)
        population.set_present_variable(col.LTP_DIAGNOSED, False, sub_pop)
        population.set_present_variable(col.LTP_ART, False, sub_pop)

    def update_recent_ltp_status(self, population: Population):
        # Updates recent LTP status for people with LTPs
        # People without LTPs are unchanged (i.e. fixed from previous partner)
        people_with_ltp = population.get_sub_pop(COND(col.LONG_TERM_PARTNER, op.eq, True))
        ltp_statuses = population.get_variable(col.LTP_STATUS, people_with_ltp)
        ltp_diagnoses = population.get_variable(col.LTP_DIAGNOSED, people_with_ltp)
        ltps_on_art = population.get_variable(col.LTP_ART, people_with_ltp)
        population.set_present_variable(col.RECENT_LTP_STATUS, ltp_statuses, people_with_ltp)
        population.set_present_variable(col.RECENT_LTP_DIAGNOSED, ltp_diagnoses, people_with_ltp)
        population.set_present_variable(col.RECENT_LTP_ART, ltps_on_art, people_with_ltp)

    def set_infection_from_infected_ltp(self, population: Population):
        """
        Sets the probability of infection for the population
        for which infection occurs from an infected long term partner.
        """
        def calculate_risk_of_infection(viral_suppression, ltp_primary, size):
            people_vs = population.get_sub_pop([(col.VIRAL_SUPPRESSION, op.eq, viral_suppression)])
            risk = rng.normal((0.05*self.transmission_factor), 0.0125, size)
            vlgroup = 3
            if viral_suppression:
                risk = rng.normal(self.tr_rate_undetectable_vl, 0.000025, size)
                vlgroup = 0
            if ltp_primary:
                risk = rng.normal(self.tr_rate_primary, 0.075, size)
                vlgroup = 5
            population.set_present_variable(col.RISK_LTP_INFECTED, risk, people_vs)
            population.set_present_variable(col.VIRAL_LOAD_GROUP, vlgroup, people_vs)
            population.set_present_variable(col.RESISTANCE_MUTATIONS,
                                            self.resistance_mutations_prop_vlg[vlgroup],
                                            people_vs)

        people_with_infected_ltp = population.get_sub_pop([(col.LONG_TERM_PARTNER, op.eq, True),
                                                           (col.LTP_STATUS, op.eq, True)])

        ltp_infection_date = population.get_variable(col.LTP_INFECTION_DATE, people_with_infected_ltp)
        ltp_primary_infection = ltp_infection_date > (population.date - timedelta(days=90))
        population.set_present_variable(col.LTP_IN_PRIMARY, ltp_primary_infection, people_with_infected_ltp)

        population.transform_group([col.VIRAL_SUPPRESSION, col.LTP_IN_PRIMARY],
                                   calculate_risk_of_infection,
                                   use_size=True,
                                   sub_pop=people_with_infected_ltp)

    def set_new_ltp_already_infected(self, population: Population):
        """
        Calculates the probability of new partners being infected
        """
        self.update_HIV_prevalence(population)
        
        # chance for new LTP to be the most recent LTP
        # Carries over properties like HIV status and diagnosis
        people_with_new_ltp = population.get_sub_pop(COND(col.LTP_NEW, op.eq, True))
        people_with_prev_ltp = population.get_sub_pop_from_array(rng.uniform(size=len(people_with_new_ltp)) < self.prob_repeated_ltp,
                                                                 people_with_new_ltp)
        population.set_present_variable(col.LTP_STATUS,
                                        population.get_variable(col.RECENT_LTP_STATUS, people_with_prev_ltp),
                                        people_with_prev_ltp)
        population.set_present_variable(col.LTP_DIAGNOSED,
                                        population.get_variable(col.RECENT_LTP_DIAGNOSED, people_with_prev_ltp),
                                        people_with_prev_ltp)
        population.set_present_variable(col.LTP_ART,
                                        population.get_variable(col.RECENT_LTP_ART, people_with_prev_ltp),
                                        people_with_prev_ltp)
        
        # Apply to all uninfected LTP based on prevalence
        uninfected_ltp = population.get_sub_pop(AND(COND(col.LTP_NEW, op.eq, True),
                                                    COND(col.LTP_STATUS, op.eq, False)))
        def calculate_new_ltp_infection(sex, age_group, size):
            infected = rng.uniform(size=size) < self.prevalence[sex][age_group]
            return infected
        
        new_ltp_infected = population.transform_group([col.SEX, col.AGE_GROUP],
                                                      calculate_new_ltp_infection,
                                                      use_size=True,
                                                      sub_pop=uninfected_ltp)
        population.set_present_variable(col.LTP_STATUS, new_ltp_infected, uninfected_ltp)

        new_infected_ltp = population.get_sub_pop(AND(COND(col.LTP_STATUS, op.eq, True),
                                                      COND(col.LTP_NEW, op.eq, True)))

        self.diagnose_ltp(population, new_infected_ltp)

        # ART of diagnosed partners (not currently carried on from recent LTP)
        # TODO: check if this continuation of recent partner ART is correct
        ltp_on_ART = population.get_sub_pop(AND(COND(col.LTP_ART, op.eq, True),
                                                COND(col.LTP_NEW, op.eq, True)))
        continuing_ART = self.get_ltps_continuing_art(ltp_on_ART)

        ltp_off_ART = population.get_sub_pop(AND(COND(col.LTP_DIAGNOSED, op.eq, True),
                                                 COND(col.LTP_ART, op.eq, False),
                                                 COND(col.LTP_NEW, op.eq, True)))
        starting_ART = self.get_ltps_starting_art(ltp_off_ART)
        # Update ART statuses
        population.set_present_variable(col.LTP_ART, continuing_ART, ltp_on_ART)
        population.set_present_variable(col.LTP_ART, starting_ART, ltp_off_ART)

        # Viral load suppression in LTP
        viral_suppressed_ltp = population.get_sub_pop(AND(COND(col.LTP_VIRAL_SUPPRESSED, op.eq, True),
                                                          COND(col.LTP_NEW, op.eq, True)))
        viral_unsuppressed_ltp = population.get_sub_pop(AND(COND(col.LTP_VIRAL_SUPPRESSED, op.eq, False),
                                                            COND(col.LTP_NEW, op.eq, True)))

        # 3% chance that virally suppressed person becomes un-suppressed
        remaining_suppressed = rng.uniform(size=len(viral_suppressed_ltp)) < self.prob_ltp_remain_suppressed

        # chance of becoming virally suppressed if not previously suppressed
        becoming_suppressed = self.get_ltp_becoming_suppressed(viral_unsuppressed_ltp)

        # Update viral suppression
        population.set_present_variable(col.LTP_VIRAL_SUPPRESSED, remaining_suppressed, viral_suppressed_ltp)
        population.set_present_variable(col.LTP_VIRAL_SUPPRESSED, becoming_suppressed, viral_unsuppressed_ltp)

    # HIV Progression ---------------------------------------------------------------------------------

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

    def get_hiv_status_difference(self, sex, population: Population):
        """
        Gets the different in the number of people of a given sex who are HIV negative and who have infected LTP,
        and the number of people of the opposite sex who are HIV positive and have a negative LTP.
        Only applies to those between 15 and 65.
        """
        num_neg_sex_with_infected_parter = len(population.get_sub_pop([(col.SEX, op.eq, sex),
                                                                       (col.HIV_STATUS, op.eq, False),
                                                                       (col.LONG_TERM_PARTNER, op.eq, True),
                                                                       (col.LTP_STATUS, op.eq, True),
                                                                       (col.AGE, op.ge, 15),
                                                                       (col.AGE, op.lt, 65)]))
        num_pos_other_sex_with_negative_partner = len(population.get_sub_pop([(col.SEX, op.eq, opposite_sex(sex)),
                                                                              (col.HIV_STATUS, op.eq, True),
                                                                              (col.LONG_TERM_PARTNER, op.eq, True),
                                                                              (col.LTP_STATUS, op.eq, False),
                                                                              (col.AGE, op.ge, 15),
                                                                              (col.AGE, op.lt, 65)]))

        ltp_hiv_status_difference = num_neg_sex_with_infected_parter - num_pos_other_sex_with_negative_partner
        return ltp_hiv_status_difference

    def update_HIV_status(self, population: Population):
        """
        Update HIV status for new transmissions in the last time period.
        Super simple model where probability of being infected by a given person
        is prevalence times transmission risk (P x r).
        Probability of each new partner not infecting you then is (1-Pr),
        and then prob of n partners independently not infecting you is (1-Pr)**n,
        so probability of infection is 1-((1-Pr)**n).
        """
        self.update_HIV_statistics(population)
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

    def reset_diagnoses(self, pop: Population):
        # Resets variables for diagnosis this timestep
        pop.set_present_variable(col.TB_DIAGNOSED, False)
        pop.set_present_variable(col.C_MENINGITIS_DIAGNOSED, False)
        pop.set_present_variable(col.SBI_DIAGNOSED, False)
        pop.set_present_variable(col.WHO4_OTHER_DIAGNOSED, False)

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
        new_tb = pop.get_sub_pop_from_array(tb, HIV_pos)
        pop.set_present_variable(col.TB_INFECTION_DATE, pop.date, new_tb)
        # FIXME: may need to update this reset for non-HIV related TB
        pop.set_present_variable(col.TB_INITIAL_INFECTION, False)
        pop.set_present_variable(col.TB_INITIAL_INFECTION, True, new_tb)

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
