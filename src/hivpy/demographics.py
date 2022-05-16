import logging
import operator
from math import exp, inf

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from hivpy.common import SexType, between, rng, selector
from hivpy.exceptions import SimulationException

SexDType = pd.CategoricalDtype(iter(SexType))


# Default values
# Can wrap those in something that ensures they have a description?
FEMALE_RATIO = 0.52
USE_STEPWISE_AGES = False
INC_CAT = 1

DEATH_RATE_MALE = {
    (15, 20): 0.002,
    (20, 25): 0.0032,
    (25, 30): 0.0058,
    (30, 35): 0.0075,
    (35, 40): 0.008,
    (40, 45): 0.01,
    (45, 50): 0.012,
    (50, 55): 0.019,
    (55, 60): 0.025,
    (60, 65): 0.035,
    (65, 70): 0.045,
    (70, 75): 0.055,
    (75, 80): 0.065,
    (80, 85): 0.1,
    (85, inf): 0.4
}
DEATH_RATE_FEMALE = {
    (15, 20): 0.0015,
    (20, 25): 0.0028,
    (25, 30): 0.004,
    (30, 35): 0.004,
    (35, 40): 0.0042,
    (40, 45): 0.0055,
    (45, 50): 0.0075,
    (50, 55): 0.011,
    (55, 60): 0.015,
    (60, 65): 0.021,
    (65, 70): 0.03,
    (70, 75): 0.038,
    (75, 80): 0.05,
    (80, 85): 0.07,
    (85, inf): 0.15
}
# DEATH_RATES = {  # annual death rates per sex
#     SexType.Male: DEATH_RATE_MALE,
#     SexType.Female: DEATH_RATE_FEMALE
# }
DEATH_RATES = {  # annual death rates per sex
    SexType.Male: [0] + list(DEATH_RATE_MALE.values()),
    SexType.Female: [0] + list(DEATH_RATE_FEMALE.values())
}


class StepwiseAgeDistribution:
    # stepwise distributions
    stepwise_model1 = np.array([0.18, 0.165, 0.144, 0.114, 0.09, 0.08, 0.068,
                                0.047, 0.036, 0.027, 0.021, 0.016, 0.012])
    stepwise_model2 = np.array([0.15, 0.13, 0.12, 0.11, 0.1, 0.09, 0.08, 0.065,
                                0.048, 0.04, 0.03, 0.021, 0.016])
    stepwise_model3 = np.array([0.128, 0.119, 0.113, 0.104, 0.097, 0.09, 0.081,
                                0.074, 0.06, 0.05, 0.038, 0.026, 0.02])
    stepwise_boundaries = np.array([-68, -55, -45, -35, -25, -15, -5, 5, 15,
                                    25, 35, 45, 55, 65])

    def __init__(self, ages, probabilities):
        assert len(ages) == len(probabilities) + 1
        self.probability_list = probabilities
        self.age_boundaries = ages

    @classmethod
    def select_model(cls, inc_cat):
        assert inc_cat in [1, 2, 3]
        if(inc_cat == 1):
            return cls(cls.stepwise_boundaries, cls.stepwise_model1)
        elif(inc_cat == 2):
            return cls(cls.stepwise_boundaries, cls.stepwise_model2)
        else:
            return cls(cls.stepwise_boundaries, cls.stepwise_model3)

    # Generate cumulative probabilites from stepwise distribution
    def _gen_cumulative_prob(self):
        N = self.probability_list.size
        if(not np.isclose(sum(self.probability_list), 1.0, atol=1e-10, equal_nan=False)):
            raise SimulationException("Age probability distribution does not sum to one.")
        if(any(self.probability_list <= 0)):
            raise SimulationException(f"Probability density cannot be negative anywhere: \
                                      {self.probability_list}")
        CP = np.cumsum(self.probability_list)
        CP[N-1] = 1.0
        return CP

    # Generate Age With Stepwise Distribution
    # the size of ages should be one larger than the cumulative probabilty
    # so that each bucket has an upper and lower bound.
    def gen_ages(self, N):
        """Generate Age With Stepwise Distribution

        the size of ages should be one larger than the cumulative probabilty
        so that each bucket has an upper and lower bound.
        """
        cpd = self._gen_cumulative_prob()
        p0 = 0.0
        rands = rng.random(N)
        ages_out = np.zeros(N)
        for i in range(0, cpd.size):
            # in order to vectorise this method we create a mask of values
            # that need to change for each age group
            prob_mask = (p0 < rands) & (rands <= cpd[i])
            ages_out += ((self.age_boundaries[i]
                          + (rands - p0)/(cpd[i]-p0)
                          * (self.age_boundaries[i+1]-self.age_boundaries[i]))
                         * prob_mask)
            p0 = cpd[i]

        return ages_out


class ContinuousAgeDistribution:
    """
    Class to handle age distributions using a continuous probability density.
    Linear-Exponential function is used for the example distribution.

    P(age) = (m*age + c)*exp(A(age-B))

    This fits all three example cases quite neatly.
    P(age): Probability density at a given age
    m: gradient of linear part, determines decline in population with age
    c: intercept of linear part, determines maximum age
    A: exponent of exponential part, determines the curvature
    B: offset of exponential part, scaling parameter
    """

    def _integrated_linexp(self, x, m, c, A, B):
        """Intregral of linear-exponential function

        Un-normalised cumulative probability distribution for age.
        """
        return np.exp(A*(x-B))*(m*x+c - m/A)/A

    n_inversion_points = 11

    # Example parameters
    modelParams1 = [-7.19e-4, 5.39e-2, -8.10e-3, 2.12e1]
    modelParams2 = [-1.03e-3, 7.45e-2, -1.12e-3, 8.47]
    modelParams3 = [-1.15e-3, 8.47e-2, 2.24e-3, 2.49e1]

    def __init__(self, min_age, max_age, modelParams):
        self.min_age = min_age
        model_age_limit = -modelParams[1]/modelParams[0]
        if(max_age > model_age_limit):
            logging.getLogger("Demographics").warning(f"Max age exceeds the maximum age limit for "
                                                      f"this model (negative probability). "
                                                      f"Adjusting max age to {model_age_limit}")
            self.max_age = model_age_limit
        else:
            self.max_age = max_age
        self.cpd = lambda x: self._integrated_linexp(x, *modelParams)

    @classmethod
    def select_model(cls, inc_cat):
        if(inc_cat == 1):
            return cls(-68, 65, cls.modelParams1)
        elif(inc_cat == 2):
            return cls(-68, 65, cls.modelParams2)
        else:
            return cls(-68, 65, *cls.modelParams3)

    def gen_ages(self, N):
        """Generate ages using a (non-normalised) continuous cumulative probability distribution

        Given an analytic PD, this should also be analytically defined
        Cumulative probability distribution is defined in _integratedLinexp
        """
        # Normalise distribution over given range
        C = self.cpd(self.min_age)
        M = 1/(self.cpd(self.max_age)-self.cpd(self.min_age))

        def norm_dist(x):
            return M*(self.cpd(x) - C)

        # sample and invert the normalised distribution (in case analytic inverse in impractical)
        NormX = np.linspace(self.min_age, self.max_age, self.n_inversion_points)
        NormY = norm_dist(NormX)

        # fix the start and end values in case of numerical errors
        NormY[0] = 0.0
        NormY[self.n_inversion_points-1] = 1.0
        NormInv = interp1d(NormY, NormX, kind='cubic')

        # generate N random numbers in (0,1) and convert to ages
        R = rng.random(N)
        Ages = NormInv(R)
        return Ages


class DemographicsModule:

    def __init__(self, **kwargs):
        params = {
            "female_ratio": FEMALE_RATIO,
            "use_stepwise_ages": USE_STEPWISE_AGES,
            "inc_cat": INC_CAT,
            "death_rates": DEATH_RATES
        }
        # allow setting some parameters explicitly
        # could be useful if we have another method for more complex initialization,
        # e.g. from a config file
        for param,  value in kwargs.items():
            assert param in params, f"{param} is not related to this module."
            params[param] = value
        self.params = params

    def initialize_sex(self, count):
        sex_distribution = (
            1 - self.params['female_ratio'], self.params['female_ratio'])
        return pd.Series(rng.choice(SexType, count, p=sex_distribution)).astype(SexDType)

    def initialise_age(self, count):
        if(self.params['use_stepwise_ages'] is True):
            age_distribution = StepwiseAgeDistribution.select_model(self.params['inc_cat'])
        else:
            age_distribution = ContinuousAgeDistribution.select_model(self.params['inc_cat'])

        return age_distribution.gen_ages(count)

    def determine_deaths_old(self, population_data: pd.DataFrame) -> pd.Series:
        """Get which individuals die in a time step, as a boolean Series."""
        all_died = pd.Series([False] * len(population_data))
        for sex in SexType:
            for (low_age, high_age), rate in self.params["death_rates"][sex].items():
                # Find everyone in the relevant age range
                index = selector(population_data, sex=(operator.eq, sex),
                                 age=(between, (low_age, high_age)))
                group = population_data[index]
                # Probability of dying, assuming time step of 3 months
                prob_of_death = 1 - exp(-rate / 4)
                died = rng.choice([True, False], size=len(group),
                                  p=[prob_of_death, 1 - prob_of_death])
                # Mark those who died from this group
                all_died |= pd.Series(died, index=group.index)
        return population_data.date_of_death.isnull() & all_died

    def determine_deaths(self, population_data: pd.DataFrame) -> pd.Series:
        """Get which individuals die in a time step, as a boolean Series."""
        # This binning should perhaps happen when the date advances
        # Age groups are the same regardless of sex
        age_limits = [lower for (lower, upper) in DEATH_RATE_MALE] + [inf]
        population_data["age_group"] = np.digitize(population_data.age, age_limits)

        death_probs = np.full(len(population_data), np.nan)
        age_groups = population_data.groupby(["sex", "age_group"])
        for (sex, age_group), entries in age_groups.groups.items():
            rate = self.params["death_rates"][sex][age_group]
            # Probability of dying, assuming time step of 3 months
            prob_of_death = 1 - exp(-rate / 4)
            death_probs[entries] = prob_of_death
            # Mark those who died from this group and sex
        rands = rng.random(len(population_data))
        return population_data.date_of_death.isnull() & (rands < death_probs)
