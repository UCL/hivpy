import importlib.resources
from datetime import timedelta

import numpy as np

import hivpy.column_names as col

from .circumcision_data import CircumcisionData
from .common import SexType, rng


class CircumcisionModule:

    def __init__(self, **kwargs):
        # init cirumcision data
        with importlib.resources.path("hivpy.data", "circumcision.yaml") as data_path:
            self.c_data = CircumcisionData(data_path)
        self.mc_int = self.c_data.mc_int
        self.test_link_circ_prob = self.c_data.test_link_circ_prob
        self.circ_inc_rate_year_i = self.c_data.circ_inc_rate_year_i
        self.circ_inc_rate = self.c_data.circ_inc_rate.sample()
        self.rel_incr_circ_post_2013 = self.c_data.rel_incr_circ_post_2013.sample()
        self.circ_inc_15_19 = self.c_data.circ_inc_15_19.sample()
        self.circ_red_20_30 = self.c_data.circ_red_20_30.sample()
        self.circ_red_30_50 = self.c_data.circ_red_30_50.sample()
        self.prob_birth_circ = self.c_data.prob_birth_circ.sample()
        # TODO: these fields doesn't belong here and/or
        # need to be implemented properly another time
        self.year_interv = 2022
        self.covid_disrup_affected = 0
        self.vmmc_disrup_covid = 0

    def init_birth_circumcision_all(self, population, date):
        """
        Initialise circumcision at birth for the entire male population,
        both born and unborn. COVID disruption is not factored in.
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

        This is an alternative birth circumcision initialisation
        method to `init_birth_circumcision_all` and requires
        the use of `update_birth_circumcision` at every time step
        to work as expected.
        """
        male_born_population = population.index[(population[col.SEX] == SexType.Male)
                                                & (population[col.AGE] >= 0.25)]
        r = rng.uniform(size=len(male_born_population))
        circumcision = r < self.prob_birth_circ
        population.loc[male_born_population, col.CIRCUMCISED] = circumcision
        # all circumcised males get a circumcision date of the start of the simulation
        population.loc[population[col.CIRCUMCISED], col.CIRCUMCISION_DATE] = date

    def update_birth_circumcision(self, population, time_step, date):
        """
        Update birth circumcision for newly born males.
        COVID disruption is factored in.

        Requires birth circumcision to be initialised with
        `init_birth_circumcision_born` instead of
        `init_birth_circumcision_all` in order to work as expected.
        """
        # covid disruption causes circumcision probability to be 0
        if (self.covid_disrup_affected != 1) & (self.vmmc_disrup_covid != 1):
            # assumes ages have already been incremented
            newborn_males = population.index[(population[col.SEX] == SexType.Male)
                                             & (population[col.AGE] >= 0.25)
                                             & (population[col.AGE] - time_step.days / 365 < 0.25)]
            r = rng.uniform(size=len(newborn_males))
            circumcision = r < self.prob_birth_circ
            population.loc[newborn_males, col.CIRCUMCISED] = circumcision
            # newly circumcised males get the current date set as their circumcision date
            circ_newborn_males = population.index[population[col.CIRCUMCISED]
                                                  & population[col.CIRCUMCISION_DATE].isnull()]
            population.loc[circ_newborn_males, col.CIRCUMCISION_DATE] = date

    def update_vmmc(self, pop):
        """
        Update voluntary medical male circumcision intervention.
        COVID disruption is factored in.
        """
        # keep date in circumcision for now for the sake of
        # the function passed to transform_group
        self.date = pop.date  # should year be float instead of int?

        # no further circumcision
        if (self.vmmc_disrup_covid == 1) \
           | ((self.year_interv <= self.date.year)
              & (self.circ_inc_rate_year_i == 2)) \
           | ((self.year_interv + 5 <= self.date.year)
              & (self.circ_inc_rate_year_i == 4)):
            self.test_link_circ_prob = 0

        # only begin VMMC after a specific year
        elif self.mc_int < self.date.year:

            # circumcision stops in 10-14 year olds
            if ((self.circ_inc_rate_year_i == 1)
                | (self.circ_inc_rate_year_i == 3)
                | (self.circ_inc_rate_year_i == 4)) \
               & (self.year_interv <= self.date.year):
                uncirc_male_population = pop.data.index[(pop.data[col.SEX] == SexType.Male)
                                                        & ~pop.data[col.CIRCUMCISED]
                                                        & (pop.data[col.AGE] >= 15)
                                                        & (pop.data[col.AGE] < 50)]
            # get uncircumcised male population of specific ages
            else:
                uncirc_male_population = pop.data.index[(pop.data[col.SEX] == SexType.Male)
                                                        & ~pop.data[col.CIRCUMCISED]
                                                        & (pop.data[col.AGE] >= 10)
                                                        & (pop.data[col.AGE] < 50)]

            # continue if uncircumcised males are present this timestep
            if len(uncirc_male_population) > 0:

                # group males by age groups
                # TODO: change age group col name to be more descriptive
                pop.data.loc[uncirc_male_population,
                             col.AGE_GROUP] = np.digitize(pop.data.loc[uncirc_male_population,
                                                                       col.AGE], [10, 20, 30, 50])
                # calculate vmmc outcomes
                circumcision = pop.transform_group([col.AGE_GROUP], self.calc_prob_circ,
                                                   sub_pop=uncirc_male_population)
                pop.data.loc[uncirc_male_population, col.CIRCUMCISED] = circumcision
                pop.data.loc[uncirc_male_population, col.VMMC] = circumcision
                # newly circumcised males get the current date set as their circumcision date
                new_circ_males = pop.data.index[pop.data[col.CIRCUMCISED]
                                                & pop.data[col.CIRCUMCISION_DATE].isnull()]
                pop.data.loc[new_circ_males, col.CIRCUMCISION_DATE] = self.date

    def calc_prob_circ(self, age_group, size):
        """
        Calculates the circumcision probability for a given
        age group and returns VMMC outcomes.
        """
        age_mod = 1
        # age group 1 has no modifier in most cases
        if age_group == 2:
            age_mod = self.circ_red_20_30
        elif age_group == 3:
            age_mod = self.circ_red_30_50

        calc_date = self.date.year
        # cap date at 2019 for calculations
        if 2019 < self.date.year:
            calc_date = 2019

        # circumcision probability for a given age group
        if 2013 < self.date.year:
            # case where age group 1 has a modifier
            ag1_has_mod = (age_group == 1) & (self.circ_inc_rate_year_i == 1) \
                          & (self.year_interv <= self.date.year)
            if ag1_has_mod:
                prob_circ = ((2013 - self.mc_int) + (calc_date - 2013)
                             * self.rel_incr_circ_post_2013 * self.circ_inc_15_19) \
                            * self.circ_inc_rate
            else:
                prob_circ = ((2013 - self.mc_int) + (calc_date - 2013)
                             * self.rel_incr_circ_post_2013) * self.circ_inc_rate * age_mod
        else:
            prob_circ = (calc_date - self.mc_int) * self.circ_inc_rate * age_mod
        # outcomes
        r = rng.uniform(size=size)
        circumcision = r < prob_circ

        return circumcision
