import numpy as np
import pandas as pd

import hivpy.column_names as col

from .common import SexType, opposite_sex, rng


class HIVStatusModule:
    hiv_a = -0.00016  # placeholder value for now
    hiv_b = 0.0128    # placeholder value for now
    hiv_c = -0.156    # placeholder value for now

    def __init__(self):
        self.stp_HIV_rate = {SexType.Male: np.zeros(5),
                             SexType.Female: np.zeros(5)}  # FIXME
        self.stp_viral_group_rate = {SexType.Male: np.array([np.zeros(7)]*5),
                                     SexType.Female: np.array([np.zeros(7)]*5)}
        # FIXME move these to data file
        # a more descriptive name would be nice
        self.fold_tr_newp = rng.choice(
            [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1/0.8, 1/0.6, 1/0.4])
        self.fold_change_w = rng.choice([1., 1.5, 2.], p=[0.05, 0.25, 0.7])
        self.fold_change_yw = rng.choice([1., 2., 3.]) * self.fold_change_w
        self.fold_change_sti = rng.choice([2., 3.])
        self.tr_rate_primary = 0.16
        self.tr_rate_undetectable_vl = rng.choice([0.0000, 0.0001, 0.0010], p=[0.7, 0.2, 0.1])
        self.transmission_means = self.fold_tr_newp * \
            np.array([0, self.tr_rate_undetectable_vl, 0.01, 0.03, 0.06, 0.1, self.tr_rate_primary])
        self.transmission_sigmas = np.array(
            [0, 0.000025**2, 0.0025**2, 0.0075**2, 0.015**2, 0.025**2, 0.075**2])

    def _prob_HIV_initial(self, age):
        """Completely arbitrary placeholder function for initial HIV presence in population"""
        return 2*np.clip(self.hiv_a*age**2 + self.hiv_b*age + self.hiv_c, 0.0, 1.0)

    def initial_HIV_status(self, population: pd.DataFrame):
        """Initialise HIV status based on age (& sex?)"""
        """Assume zero prevalence for age < 15"""
        hiv_status = self._prob_HIV_initial(population[col.AGE]) > rng.random(len(population))
        return hiv_status

    def update_partner_risk_vectors(self, population):
        """calculate the risk factor associated with each sex and age group"""
        # Should we be using for loops here or can we do better?
        for sex in SexType:
            for age_group in range(5):   # FIXME need to get number of age groups from somewhere
                sub_pop_indices = population.data.index[(population.data[col.SEX] == sex) & (
                    population.data[col.SEX_MIX_AGE_GROUP] == age_group)]
                sub_pop = population.data.loc[sub_pop_indices]
                # total number of people partnered to people in this group
                n_stp_total = sum(sub_pop[col.NUM_PARTNERS])
                # num people parters to HIV+ people in this group
                n_infected = sum(sub_pop.loc[sub_pop[col.HIV_STATUS]][col.NUM_PARTNERS])
                # Probability of being HIV prositive
                if n_infected == 0:
                    self.stp_HIV_rate[sex][age_group] = 0
                else:
                    self.stp_HIV_rate[sex][age_group] = n_infected / \
                        n_stp_total  # TODO: need to double check this definition
                # Chances of being in a given viral group
                if n_stp_total > 0:
                    self.stp_viral_group_rate[sex][age_group] = [
                        sum(sub_pop.loc[sub_pop[col.VIRAL_LOAD_GROUP] == vg,
                            col.NUM_PARTNERS])/n_stp_total for vg in range(7)]
                else:
                    self.stp_viral_group_rate[sex][age_group] = np.array([1, 0, 0, 0, 0, 0, 0])

    def set_dummy_viral_load(self, population):
        """Dummy function to set viral load until this
        part of the code has been implemented properly"""
        population.data[col.VIRAL_LOAD_GROUP] = rng.choice(7, population.size)

    def get_infection_prob(self, sex, age, n_partners, stp_age_groups):
        # Slow example that avoid repeating the iterations over partners
        # three time by putting them as part of
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

    def stp_HIV_transmission(self, person):
        # TODO: Add circumcision, STIs etc.
        """Returns True if HIV transmission occurs, and False otherwise"""
        stp_viral_groups = np.array([rng.choice(7, p=self.stp_viral_group_rate[opposite_sex(
            person[col.SEX])][age_group]) for age_group in person[col.STP_AGE_GROUPS]])
        HIV_probabilities = np.array([self.stp_HIV_rate[opposite_sex(
            person[col.SEX])][age_group] for age_group in person[col.STP_AGE_GROUPS]])
        viral_transmission_probabilities = np.array([max(0, rng.normal(
            self.transmission_means[group], self.transmission_sigmas[group]))
            for group in stp_viral_groups])
        if person[col.SEX] == SexType.Female:
            if person[col.AGE] < 20:
                viral_transmission_probabilities = (viral_transmission_probabilities
                                                    * self.fold_change_yw)
            else:
                viral_transmission_probabilities = (viral_transmission_probabilities
                                                    * self.fold_change_w)
        prob_uninfected = np.prod(1-(HIV_probabilities * viral_transmission_probabilities))
        r = rng.random()
        return r > prob_uninfected

    def update_HIV_status(self, population):
        """Update HIV status for new transmissions in the last time period.\\
            Super simple model where probability of being infected by a given person
            is prevalence times transmission risk (P x r).\\
            Probability of each new partner not infecting you then is (1-Pr)\\
            Then prob of n partners independently not infecting you is (1-Pr)**n\\
            So probability of infection is 1-((1-Pr)**n)"""
        self.update_partner_risk_vectors(population)
        HIV_neg_idx = population.data.index[(~population.data[col.HIV_STATUS]) & (
            population.data[col.NUM_PARTNERS] > 0)]
        sub_pop = population.data.loc[HIV_neg_idx]
        population.data.loc[HIV_neg_idx, col.HIV_STATUS] = sub_pop.apply(
            self.stp_HIV_transmission, axis=1)
