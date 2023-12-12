from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .population import Population

import importlib.resources
import logging
import operator as op
from math import exp

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import hivpy.column_names as col
from hivpy.common import SexType, rng, timedelta
from hivpy.demographics_data import DemographicsData
from hivpy.exceptions import SimulationException

from . import output

SexDType = pd.CategoricalDtype(iter(SexType))


# Default values
# Can wrap those in something that ensures they have a description?
USE_STEPWISE_AGES = False
INC_CAT = 1


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
        if (inc_cat == 1):
            return cls(cls.stepwise_boundaries, cls.stepwise_model1)
        elif (inc_cat == 2):
            return cls(cls.stepwise_boundaries, cls.stepwise_model2)
        else:
            return cls(cls.stepwise_boundaries, cls.stepwise_model3)

    # Generate cumulative probabilites from stepwise distribution
    def _gen_cumulative_prob(self):
        N = self.probability_list.size
        if (not np.isclose(sum(self.probability_list), 1.0, atol=1e-10, equal_nan=False)):
            raise SimulationException("Age probability distribution does not sum to one.")
        if (any(self.probability_list <= 0)):
            raise SimulationException(f"Probability density cannot be negative anywhere: \
                                      {self.probability_list}")
        CP = np.cumsum(self.probability_list)
        CP[N-1] = 1.0
        return CP

    def gen_ages(self, N):
        """
        Generate Age With Stepwise Distribution.

        The size of ages should be one larger than the cumulative probabilty
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
    Linear-Exponential function is used for the example distribution:

    P(age) = (m*age + c)*exp(A(age-B))

    This fits all three example cases quite neatly.

    P(age): Probability density at a given age. \n
    m: gradient of linear part, determines decline in population with age. \n
    c: intercept of linear part, determines maximum age. \n
    A: exponent of exponential part, determines the curvature. \n
    B: offset of exponential part, scaling parameter. \n
    """

    def _integrated_linexp(self, x, m, c, A, B):
        """
        Intregral of linear-exponential function.

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
        if (max_age > model_age_limit):
            logging.getLogger("Demographics").warning(f"Max age exceeds the maximum age limit for "
                                                      f"this model (negative probability). "
                                                      f"Adjusting max age to {model_age_limit}")
            self.max_age = model_age_limit
        else:
            self.max_age = max_age
        self.cpd = lambda x: self._integrated_linexp(x, *modelParams)

    @classmethod
    def select_model(cls, inc_cat):
        if (inc_cat == 1):
            return cls(-68, 65, cls.modelParams1)
        elif (inc_cat == 2):
            return cls(-68, 65, cls.modelParams2)
        else:
            return cls(-68, 65, *cls.modelParams3)

    def gen_ages(self, N):
        """
        Generate ages using a (non-normalised) continuous cumulative probability distribution.

        Given an analytic PD, this should also be analytically defined.
        Cumulative probability distribution is defined in _integratedLinexp.
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
        with importlib.resources.path("hivpy.data", "demographics.yaml") as data_path:
            self.data = DemographicsData(data_path)
        self.output = output.simulation_output
        params = {
            "female_ratio": self.data.female_ratio,
            "use_stepwise_ages": USE_STEPWISE_AGES,
            "inc_cat": INC_CAT,
            "death_rates": self.data.death_rates
        }
        # allow setting some parameters explicitly
        # could be useful if we have another method for more complex initialization,
        # e.g. from a config file
        for param,  value in kwargs.items():
            assert param in params, f"{param} is not related to this module."
            params[param] = value
        self.params = params

        # DEATH
        # TODO: move these params into the config file
        self.non_hiv_tb_death_risk = 0.3
        self.non_hiv_tb_risk = 5e-4
        self.off_art_hiv_mortality_factor = rng.choice([1.5, 2, 3])

    def initialise_sex(self, count):
        sex_distribution = (
            1 - self.params['female_ratio'], self.params['female_ratio'])
        return pd.Series(rng.choice(SexType, count, p=sex_distribution)).astype(SexDType)

    def initialise_age(self, count):
        if (self.params['use_stepwise_ages'] is True):
            age_distribution = StepwiseAgeDistribution.select_model(self.params['inc_cat'])
        else:
            age_distribution = ContinuousAgeDistribution.select_model(self.params['inc_cat'])

        return age_distribution.gen_ages(count)

    def initialise_hard_reach(self, population):
        # base probabilities for being hard to reach
        self.prob_hard_reach_f = round(rng.uniform(0.05, 0.2), 2)
        self.prob_hard_reach_m = self.prob_hard_reach_f + round(rng.uniform(0, 0.1), 2)
        # split population by sex
        female_pop = population.index[(population[col.SEX] == SexType.Female)]
        male_pop = population.index[(population[col.SEX] == SexType.Male)]
        r_f = rng.uniform(size=len(female_pop))
        r_m = rng.uniform(size=len(male_pop))
        # outcomes
        hard_reach_f = r_f < self.prob_hard_reach_f
        hard_reach_m = r_m < self.prob_hard_reach_m
        population.loc[female_pop, col.HARD_REACH] = hard_reach_f
        population.loc[male_pop, col.HARD_REACH] = hard_reach_m

    def determine_deaths(self, pop: Population, time_step: timedelta) -> pd.Series:
        """
        Get which individuals die in a time step, as a boolean Series.
        """
        # Age based death probability
        # This binning should perhaps happen when the date advances
        # Age groups are the same regardless of sex
        def _probability_of_death(sex: SexType, age_group: int, HIV_status: bool) -> float:
            rate = self.params["death_rates"][sex][age_group]
            # Probability of dying, assuming time step of 3 months
            prob_of_death = 1 - exp(-rate * (time_step.month/12))
            if (HIV_status):
                prob_of_death *= self.off_art_hiv_mortality_factor
            return prob_of_death

        age_limits = self.data.death_age_limits
        over_15 = pop.get_sub_pop([(col.AGE, op.ge, 15)])
        pop.set_present_variable(col.AGE_GROUP, np.digitize(pop.get_variable(col.AGE, over_15), age_limits), over_15)

        death_probs = pop.transform_group([col.SEX, col.AGE_GROUP, col.HIV_STATUS],
                                          _probability_of_death, use_size=False, sub_pop=over_15)
        rands = rng.random(len(over_15))
        deaths = rands < death_probs
        self.output.record_non_HIV_deaths(pop, deaths)
        return deaths
