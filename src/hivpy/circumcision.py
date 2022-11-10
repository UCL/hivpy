import importlib.resources
from datetime import timedelta

import hivpy.column_names as col

from .circumcision_data import CircumcisionData
from .common import SexType, rng


class CircumcisionModule:

    def __init__(self, **kwargs):
        # init cirumcision data
        with importlib.resources.path("hivpy.data", "circumcision.yaml") as data_path:
            self.c_data = CircumcisionData(data_path)
        self.mc_int = self.c_data.mc_int
        self.circ_inc_rate = self.c_data.circ_inc_rate.sample()
        self.prob_birth_circ = self.c_data.prob_birth_circ.sample()

    def init_birth_circumcision_all(self, population, date):
        """
        Initialise circumcision at birth for the entire male population,
        both born and unborn.
        """
        male_population = population.index[population[col.SEX] == SexType.Male]
        r = rng.uniform(size=len(male_population))
        circumcision = r < self.prob_birth_circ
        population.loc[male_population, col.CIRCUMCISED] = circumcision
        # split newly circumcised population into born and unborn
        circ_born_population = population.index[population[col.CIRCUMCISED]
                                                & (population[col.AGE] >= 0.25)]
        circ_unborn_population = population.index[population[col.CIRCUMCISED]
                                                  & (population[col.AGE] < 0.25)]
        # use current simulation start date as circumcision date for born individuals
        population.loc[circ_born_population, col.CIRCUMCISION_DATE] = date
        # find date where each unborn individual's age would be 0.25
        population.loc[circ_unborn_population,
                       col.CIRCUMCISION_DATE] = population[col.AGE].transform(
                                                lambda x: date - timedelta(days=(x+0.25)*365))

    def init_birth_circumcision_born(self, population, date):
        """
        Initialise circumcision at birth for all born males.
        """
        male_born_population = population.index[(population[col.SEX] == SexType.Male)
                                                & (population[col.AGE] >= 0.25)]
        r = rng.uniform(size=len(male_born_population))
        circumcision = r < self.prob_birth_circ
        population.loc[male_born_population, col.CIRCUMCISED] = circumcision
        # all circumcised men get a circumcision date of the start of the simulation
        population.loc[population[col.CIRCUMCISED], col.CIRCUMCISION_DATE] = date

    def update_birth_circumcision(self, population, time_step, date):
        """
        Update birth circumcision for newly born males.
        """
        # assumes ages have already been incremented
        newborn_males = population.index[(population[col.SEX] == SexType.Male)
                                         & (population[col.AGE] >= 0.25)
                                         & (population[col.AGE] - time_step.days / 365 < 0.25)]
        r = rng.uniform(size=len(newborn_males))
        circumcision = r < self.prob_birth_circ
        population.loc[newborn_males, col.CIRCUMCISED] = circumcision
        # newly circumcised men get the current date set as their circumcision date
        circ_newborn_males = population.index[population[col.CIRCUMCISED]
                                              & population[col.CIRCUMCISION_DATE].isnull()]
        population.loc[circ_newborn_males, col.CIRCUMCISION_DATE] = date
