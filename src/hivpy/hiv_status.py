import operator

import pandas as pd

import hivpy.column_names as col

from .common import rng
from .sexual_behaviour import selector


class HIVStatusModule:
    initial_hiv_newp_threshold = 7  # lower limit for HIV infection at start of epidemic
    initial_hiv_prob = 0.8  # for those with enough partners at start of epidemic

    def initial_HIV_status(self, population: pd.DataFrame):
        """Initialise HIV status at the start of the simulation to no infections."""
        # This may be useful as a separate method if we end up representing status
        # as something more complex than a boolean, e.g. an enum.
        return pd.Series(False, population.index)

    def introduce_HIV(self, population: pd.DataFrame):
        """Initialise HIV status at the start of the pandemic."""
        # At the start of the epidemic, we consider only people with short-term partners over
        # the threshold as potentially infected.
        initial_candidates = population[col.NUM_PARTNERS] >= self.initial_hiv_newp_threshold
        # Each of them has the same probability of being infected.
        initial_infection = rng.uniform(size=initial_candidates.sum()) < self.initial_hiv_prob
        hiv_status = pd.Series(False, index=population.index)
        hiv_status.loc[initial_candidates] = initial_infection
        return hiv_status

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
        HIV_infection_risk = 0.2  # made up, based loosely on transmission probabilities
        n_partners = population.loc[HIV_neg_idx, col.NUM_PARTNERS]
        HIV_prob = 1-((1-HIV_prevalence*HIV_infection_risk)**n_partners)
        population.loc[HIV_neg_idx, col.HIV_STATUS] = (rands <= HIV_prob)
