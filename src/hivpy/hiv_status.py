from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .population import Population

import operator

import numpy as np
import pandas as pd

import hivpy.column_names as col

from .common import AND, COND, SexType, opposite_sex, rng


class HIVStatusModule:

    initial_hiv_newp_threshold = 7  # lower limit for HIV infection at start of epidemic
    initial_hiv_prob = 0.8  # for those with enough partners at start of epidemic

    def __init__(self):
        self.stp_HIV_rate = {SexType.Male: np.zeros(5),
                             SexType.Female: np.zeros(5)}  # FIXME
        self.stp_viral_group_rate = {SexType.Male: np.array([np.zeros(7)]*5),
                                     SexType.Female: np.array([np.zeros(7)]*5)}
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
        # viral load ratios in general population for each sex and age group
        self.ratio_vl_stp = {SexType.Male: [np.zeros(6), np.zeros(6), np.zeros(6),
                                                 np.zeros(6), np.zeros(6)],
                                  SexType.Female: [np.zeros(6), np.zeros(6), np.zeros(6),
                                                   np.zeros(6), np.zeros(6)]}
        self.fold_change_w = rng.choice([1., 1.5, 2.], p=[0.05, 0.25, 0.7])
        self.fold_change_yw = rng.choice([1., 2., 3.]) * self.fold_change_w
        self.fold_change_sti = rng.choice([2., 3.])
        self.transmission_means = self.stp_transmission_factor * \
            np.array([0, self.tr_rate_undetectable_vl, 0.01, 0.03, 0.06, 0.1, self.tr_rate_primary])
        self.transmission_sigmas = np.array(
            [0, 0.000025**2, 0.0025**2, 0.0075**2, 0.015**2, 0.025**2, 0.075**2])
        self.circumcision_risk_reduction = 0.4  # reduce infection risk by 60%

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
            [(col.NUM_PARTNERS, operator.ge, self.initial_hiv_newp_threshold)])
        # initial_candidates = population[col.NUM_PARTNERS] >= self.initial_hiv_newp_threshold
        # Each of them has the same probability of being infected.
        num_init_candidates = len(initial_candidates)
        rands = rng.uniform(size=num_init_candidates)
        initial_infection = rands < self.initial_hiv_prob
        hiv_status = pd.Series(False, index=population.data.index)
        hiv_status.loc[initial_candidates] = initial_infection
        return hiv_status

    def update_partner_risk_vectors(self, population: Population):
        """
        Calculate the risk factor associated with each sex and age group.
        """
        # Should we be using for loops here or can we do better?
        for sex in SexType:
            for age_group in range(5):   # FIXME: need to get number of age groups from somewhere
                sub_pop = population.get_sub_pop([(col.SEX, operator.eq, sex),
                                                  (col.SEX_MIX_AGE_GROUP, operator.eq, age_group)])
                # total number of people partnered to people in this group
                n_stp_total = sum(population.get_variable(col.NUM_PARTNERS, sub_pop))
                # num people partnered to HIV+ people in this group
                HIV_positive_pop = population.get_sub_pop([(col.HIV_STATUS, operator.eq, True)])
                n_stp_of_infected = sum(population.get_variable(col.NUM_PARTNERS,
                                                                population.get_sub_pop_intersection(
                                                                    sub_pop,
                                                                    HIV_positive_pop
                                                                )))
                # probability of being HIV positive
                if n_stp_of_infected == 0:
                    self.stp_HIV_rate[sex][age_group] = 0
                else:
                    self.stp_HIV_rate[sex][age_group] = n_stp_of_infected / \
                        n_stp_total  # TODO: need to double check this definition
                # chances of being in a given viral group
                if n_stp_total > 0:
                    self.stp_viral_group_rate[sex][age_group] = [
                        sum(population.get_variable(col.NUM_PARTNERS,
                            population.get_sub_pop_intersection(
                                sub_pop,
                                population.get_sub_pop([(col.VIRAL_LOAD_GROUP, operator.eq, vg)])
                            )))/n_stp_total for vg in range(7)]
                else:
                    self.stp_viral_group_rate[sex][age_group] = np.array([1, 0, 0, 0, 0, 0, 0])

    def set_dummy_viral_load(self, population: Population):
        """
        Dummy function to set viral load until this
        part of the code has been implemented properly.
        """
        population.init_variable(col.VIRAL_LOAD_GROUP, rng.choice(7, population.size))

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

    def get_infection_prob(self, sex, age, n_partners, stp_age_groups):
        # Slow example that avoids repeating the iterations over partners
        # three times by putting them as part of
        # one for loop, but for loops in python will be slow.
        target_sex = opposite_sex(sex)
        infection_prob = np.zeros(n_partners)
        for i in range(n_partners):
            stp_viral_group = rng.choice(
                7, p=self.stp_viral_group_rate[target_sex][stp_age_groups[i]])
            HIV_probability = self.stp_HIV_rate[opposite_sex(target_sex)][stp_age_groups[i]]
            infection_prob[i] = HIV_probability * max(0, rng.normal(
                self.transmission_means[stp_viral_group],
                self.transmission_sigmas[stp_viral_group]))
            if (sex == SexType.Female):
                if (age < 20):
                    infection_prob[i] *= self.fold_change_yw
                else:
                    infection_prob[i] *= self.fold_change_w
        return infection_prob

    def update_infected_stp_stats(self, population: Population):
        """
        Update a dictionary containing information about the proportion of infected short-term
        partners in the population, as well as a dictionary containing information about
        viral load group proportions of sexually active people in the population.
        Both dictionaries group the information by sex and age group.
        """

        # people with HIV
        infected_pop = population.get_sub_pop(COND(col.HIV_STATUS, operator.eq, True))

        # sexually active people by sex
        active_male_pop = population.get_sub_pop(AND(COND(col.SEX, operator.eq, SexType.Male),
                                                     COND(col.NUM_PARTNERS, operator.gt, 0)))
        active_female_pop = population.get_sub_pop(AND(COND(col.SEX, operator.eq, SexType.Female),
                                                       COND(col.NUM_PARTNERS, operator.gt, 0)))

        # people by age group
        age_group_1_pop = population.get_sub_pop(AND(COND(col.AGE, operator.ge, 15),
                                                     COND(col.AGE, operator.lt, 25)))
        age_group_2_pop = population.get_sub_pop(AND(COND(col.AGE, operator.ge, 25),
                                                     COND(col.AGE, operator.lt, 35)))
        age_group_3_pop = population.get_sub_pop(AND(COND(col.AGE, operator.ge, 35),
                                                     COND(col.AGE, operator.lt, 45)))
        age_group_4_pop = population.get_sub_pop(AND(COND(col.AGE, operator.ge, 45),
                                                     COND(col.AGE, operator.lt, 55)))
        age_group_5_pop = population.get_sub_pop(AND(COND(col.AGE, operator.ge, 55),
                                                     COND(col.AGE, operator.lt, 65)))

        # sexually active men of various age groups
        active_male_pop_1 = population.get_sub_pop_intersection(active_male_pop, age_group_1_pop)
        active_male_pop_2 = population.get_sub_pop_intersection(active_male_pop, age_group_2_pop)
        active_male_pop_3 = population.get_sub_pop_intersection(active_male_pop, age_group_3_pop)
        active_male_pop_4 = population.get_sub_pop_intersection(active_male_pop, age_group_4_pop)
        active_male_pop_5 = population.get_sub_pop_intersection(active_male_pop, age_group_5_pop)
        active_male_pop_by_age = [active_male_pop_1, active_male_pop_2, active_male_pop_3,
                                  active_male_pop_4, active_male_pop_5]

        # sexually active men with HIV of various age groups
        infected_male_pop_1 = population.get_sub_pop_intersection(active_male_pop_1, infected_pop)
        infected_male_pop_2 = population.get_sub_pop_intersection(active_male_pop_2, infected_pop)
        infected_male_pop_3 = population.get_sub_pop_intersection(active_male_pop_3, infected_pop)
        infected_male_pop_4 = population.get_sub_pop_intersection(active_male_pop_4, infected_pop)
        infected_male_pop_5 = population.get_sub_pop_intersection(active_male_pop_5, infected_pop)
        infected_male_pop_by_age = [infected_male_pop_1, infected_male_pop_2, infected_male_pop_3,
                                    infected_male_pop_4, infected_male_pop_5]

        # sexually active women of various age groups
        active_female_pop_1 = population.get_sub_pop_intersection(active_female_pop, age_group_1_pop)
        active_female_pop_2 = population.get_sub_pop_intersection(active_female_pop, age_group_2_pop)
        active_female_pop_3 = population.get_sub_pop_intersection(active_female_pop, age_group_3_pop)
        active_female_pop_4 = population.get_sub_pop_intersection(active_female_pop, age_group_4_pop)
        active_female_pop_5 = population.get_sub_pop_intersection(active_female_pop, age_group_5_pop)
        active_female_pop_by_age = [active_female_pop_1, active_female_pop_2, active_female_pop_3,
                                    active_female_pop_4, active_female_pop_5]

        # sexually active women with HIV of various age groups
        infected_female_pop_1 = population.get_sub_pop_intersection(active_female_pop_1, infected_pop)
        infected_female_pop_2 = population.get_sub_pop_intersection(active_female_pop_2, infected_pop)
        infected_female_pop_3 = population.get_sub_pop_intersection(active_female_pop_3, infected_pop)
        infected_female_pop_4 = population.get_sub_pop_intersection(active_female_pop_4, infected_pop)
        infected_female_pop_5 = population.get_sub_pop_intersection(active_female_pop_5, infected_pop)
        infected_female_pop_by_age = [infected_female_pop_1, infected_female_pop_2, infected_female_pop_3,
                                      infected_female_pop_4, infected_female_pop_5]

        for age_group in range(5):

            # update proportion of infected stps
            self.ratio_infected_stp[SexType.Male][age_group] = \
                len(infected_male_pop_by_age[age_group])/len(active_male_pop_by_age[age_group]) \
                if len(active_male_pop_by_age[age_group]) > 0 else 0
            self.ratio_infected_stp[SexType.Female][age_group] = \
                len(infected_female_pop_by_age[age_group])/len(active_female_pop_by_age[age_group]) \
                if len(active_female_pop_by_age[age_group]) > 0 else 0

            for vl_group in range(6):

                # populations with given viral load group
                vl_group_pop = population.get_sub_pop(COND(col.VIRAL_LOAD_GROUP, operator.eq, vl_group+1))
                male_vl_group_pop = population.get_sub_pop_intersection(infected_male_pop_by_age[age_group],
                                                                        vl_group_pop)
                female_vl_group_pop = population.get_sub_pop_intersection(infected_female_pop_by_age[age_group],
                                                                          vl_group_pop)

                # update proportion of stps with each viral load group
                self.ratio_vl_stp[SexType.Male][age_group][vl_group] = \
                    len(male_vl_group_pop)/len(active_male_pop_by_age[age_group]) \
                    if len(active_male_pop_by_age[age_group]) > 0 else 0
                self.ratio_vl_stp[SexType.Female][age_group][vl_group] = \
                    len(female_vl_group_pop)/len(active_female_pop_by_age[age_group]) \
                    if len(active_female_pop_by_age[age_group]) > 0 else 0

    def stp_HIV_transmission(self, person):
        # TODO: Add circumcision, STIs etc.
        """
        Returns True if HIV transmission occurs, and False otherwise.
        """

        stp_viral_groups = np.array([
            rng.choice(7, p=self.stp_viral_group_rate[opposite_sex(person[col.SEX])][age_group])
            for age_group in person[col.STP_AGE_GROUPS]])
        HIV_probabilities = np.array([self.stp_HIV_rate[opposite_sex(
            person[col.SEX])][age_group] for age_group in person[col.STP_AGE_GROUPS]])
        viral_transmission_probabilities = np.array([max(0, rng.normal(
            self.transmission_means[group], self.transmission_sigmas[group]))
            for group in stp_viral_groups])
        if person[col.SEX] is SexType.Female:
            if person[col.AGE] < 20:
                viral_transmission_probabilities = (viral_transmission_probabilities
                                                    * self.fold_change_yw)
            else:
                viral_transmission_probabilities = (viral_transmission_probabilities
                                                    * self.fold_change_w)
        elif person[col.CIRCUMCISED]:
            viral_transmission_probabilities = (viral_transmission_probabilities
                                                * self.circumcision_risk_reduction)
        prob_uninfected = np.prod(1-(HIV_probabilities * viral_transmission_probabilities))
        r = rng.random()
        return r > prob_uninfected

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
        HIV_neg_active_pop = population.get_sub_pop([(col.HIV_STATUS, operator.eq, False),
                                                     (col.NUM_PARTNERS, operator.gt, 0)])
        # determine HIV status after transmission
        new_HIV_status = population.apply_function(self.stp_HIV_transmission, 1, HIV_neg_active_pop)
        # apply HIV status to sub-population
        population.set_present_variable(col.HIV_STATUS,
                                        new_HIV_status,
                                        HIV_neg_active_pop)
