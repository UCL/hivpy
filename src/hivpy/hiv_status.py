import operator

import numpy as np
import pandas as pd

from .sexual_behaviour import selector


class HIVStatusModule:
    hiv_a = -0.00016  # placeholder value for now
    hiv_b = 0.0128    # placeholder value for now
    hiv_c = -0.156    # placeholder value for now

    def _prob_HIV_initial(self, age):
        """Completely arbitrary placeholder function for initial HIV presence in population"""
        return np.clip(self.hiv_a*age**2 + self.hiv_b*age + self.hiv_c, 0.0, 1.0)

    def initial_HIV_status(self, population: pd.DataFrame):
        """Initialise HIV status based on age (& sex?)"""
        """Assume zero prevalence for age < 15"""
        hiv_status = self._prob_HIV_initial(population["age"]) > np.random.rand(len(population))
        return hiv_status

    def update_HIV_status(self, population: pd.DataFrame):
        """Update HIV status for new transmissions in the last time period.\\
            Super simple model where probability of being infected by a given person
            is prevalence times transmission risk (P x r).\\
            Probability of each new partner not infecting you then is (1-Pr)\\
            Then prob of n partners independently not infecting you is (1-Pr)**n\\
            So probability of infection is 1-((1-Pr)**n)"""
        HIV_neg_idx = selector(population, HIV_status=(operator.eq, False))
        rands = np.random.uniform(0.0, 1.0, sum(HIV_neg_idx))
        HIV_prevalence = sum(population['HIV_status'])/len(population)
        HIV_infection_risk = 0.005  # made up, based loosely on transmission probabilities
        n_partners = population.loc[HIV_neg_idx, "num_partners"]
        HIV_prob = 1-((1-HIV_prevalence*HIV_infection_risk)**n_partners)
        population.loc[HIV_neg_idx, "HIV_status"] = (HIV_prob <= rands)
