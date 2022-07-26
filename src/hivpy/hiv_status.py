import operator

import numpy as np
import pandas as pd

import hivpy.column_names as col

from .common import SexType, rng
from .sexual_behaviour import selector


class HIVStatusModule:
    hiv_a = -0.00016  # placeholder value for now
    hiv_b = 0.0128    # placeholder value for now
    hiv_c = -0.156    # placeholder value for now

    def __init__(self):
        self.stp_risk_vectors = {SexType.Male: np.zeros(10),
                                     SexType.Female: np.zeros(10)}  # FIXME

    def _prob_HIV_initial(self, age):
        """Completely arbitrary placeholder function for initial HIV presence in population"""
        return 2*np.clip(self.hiv_a*age**2 + self.hiv_b*age + self.hiv_c, 0.0, 1.0)

    def initial_HIV_status(self, population: pd.DataFrame):
        """Initialise HIV status based on age (& sex?)"""
        """Assume zero prevalence for age < 15"""
        hiv_status = self._prob_HIV_initial(population[col.AGE]) > rng.random(len(population))
        return hiv_status

    def calculate_partner_risk_vectors(self, population):
        """calculate the risk factor associated with each sex and age group"""
        # Should we be using for loops here or can we do better? 
        for sex in SexType:
            for age_group in range(10):   # FIXME 
                sub_pop_indices = population.data[col.AGE_GROUP == age_group]
                sub_pop = population.data.loc[sub_pop_indices]
                n_total = len(sub_pop)
                n_infected = sum(sub_pop[col.HIV_STATUS == True])
                self.stp_risk_vectors[sex][age_group] = n_total/n_infected  # need to double check this definition

    def num_infected_partners(self, num_partners, sex, partner_age_groups):
        """Calculates the number of infected short term partners"""
        infected_partners = sum([rng.random() < self.stp_risk_vectors[sex, age] for age in partner_age_groups])
        return infected_partners

    def update_HIV_status(self, population: pd.DataFrame):
        """Update HIV status for new transmissions in the last time period.\\
            Super simple model where probability of being infected by a given person
            is prevalence times transmission risk (P x r).\\
            Probability of each new partner not infecting you then is (1-Pr)\\
            Then prob of n partners independently not infecting you is (1-Pr)**n\\
            So probability of infection is 1-((1-Pr)**n)"""
        HIV_neg_idx = selector(population, HIV_status=(operator.eq, False))
        rands = rng.uniform(0.0, 1.0, sum(HIV_neg_idx))
        HIV_prevalence = sum(population[col.HIV_STATUS])/len(population)
        HIV_infection_risk = 0.1  # made up, based loosely on transmission probabilities
        n_partners = population.loc[HIV_neg_idx, col.NUM_PARTNERS]
        HIV_prob = 1-((1-HIV_prevalence*HIV_infection_risk)**n_partners)
        population.loc[HIV_neg_idx, col.HIV_STATUS] = (rands <= HIV_prob)
