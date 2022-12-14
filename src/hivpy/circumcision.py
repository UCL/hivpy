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

        self.vmmc_start_year = self.c_data.vmmc_start_year
        self.circ_rate_change_year = self.c_data.circ_rate_change_year
        self.prob_circ_calc_cutoff_year = self.c_data.prob_circ_calc_cutoff_year
        self.prob_circ_after_test = self.c_data.prob_circ_after_test
        self.policy_intervention_year = self.c_data.policy_intervention_year
        self.circ_policy_scenario = self.c_data.circ_policy_scenario
        # NOTE: the covid disrup field may not belong here
        self.covid_disrup_affected = self.c_data.covid_disrup_affected
        self.vmmc_disrup_covid = self.c_data.vmmc_disrup_covid
        self.circ_increase_rate = self.c_data.circ_increase_rate.sample()
        self.circ_rate_change_post_2013 = self.c_data.circ_rate_change_post_2013.sample()
        self.circ_rate_change_15_19 = self.c_data.circ_rate_change_15_19.sample()
        self.circ_rate_change_20_29 = self.c_data.circ_rate_change_20_29.sample()
        self.circ_rate_change_30_49 = self.c_data.circ_rate_change_30_49.sample()
        self.prob_birth_circ = self.c_data.prob_birth_circ.sample()

        # age range and grouping variables
        self.min_vmmc_age = 10
        self.vmmc_cutoff_age = self.min_vmmc_age + 5
        self.vmmc_age_bound_1 = 20
        self.vmmc_age_bound_2 = 30
        self.max_vmmc_age = 50

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
                                                lambda x: date - timedelta(days=(x-0.25)*365))

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
        if (not self.covid_disrup_affected) & (not self.vmmc_disrup_covid):
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
        self.date = pop.date

        # no further circumcision
        if self.vmmc_disrup_covid \
           | ((self.policy_intervention_year <= self.date.year)
              & (self.circ_policy_scenario == 2)) \
           | ((self.policy_intervention_year + 5 <= self.date.year)
              & (self.circ_policy_scenario == 4)):
            self.prob_circ_after_test = 0

        # only begin VMMC after a specific year
        elif self.vmmc_start_year <= self.date.year:

            # circumcision stops in 10-14 year olds
            if ((self.circ_policy_scenario == 1)
                | (self.circ_policy_scenario == 3)
                | (self.circ_policy_scenario == 4)) \
               & (self.policy_intervention_year <= self.date.year):
                uncirc_male_population = pop.data.index[(pop.data[col.SEX] == SexType.Male)
                                                        & ~pop.data[col.CIRCUMCISED]
                                                        & (pop.data[col.AGE] >= self.vmmc_cutoff_age)
                                                        & (pop.data[col.AGE] < self.max_vmmc_age)]
            # get uncircumcised male population of specific ages
            else:
                uncirc_male_population = pop.data.index[(pop.data[col.SEX] == SexType.Male)
                                                        & ~pop.data[col.CIRCUMCISED]
                                                        & (pop.data[col.AGE] >= self.min_vmmc_age)
                                                        & (pop.data[col.AGE] < self.max_vmmc_age)]

            # continue if uncircumcised males are present this timestep
            if len(uncirc_male_population) > 0:

                # group males by age groups
                age_groups = np.digitize(pop.data.loc[uncirc_male_population, col.AGE],
                                         [self.min_vmmc_age,
                                          self.vmmc_age_bound_1,
                                          self.vmmc_age_bound_2,
                                          self.max_vmmc_age])
                # TODO: change age group col name to be more descriptive
                pop.data.loc[uncirc_male_population, col.AGE_GROUP] = age_groups
                # calculate vmmc outcomes
                circumcision = pop.transform_group([col.AGE_GROUP], self.calc_circ_outcomes,
                                                   sub_pop=uncirc_male_population)
                pop.data.loc[uncirc_male_population, col.CIRCUMCISED] = circumcision
                pop.data.loc[uncirc_male_population, col.VMMC] = circumcision
                # newly circumcised males get the current date set as their circumcision date
                new_circ_males = pop.data.index[pop.data[col.CIRCUMCISED]
                                                & pop.data[col.CIRCUMCISION_DATE].isnull()]
                pop.data.loc[new_circ_males, col.CIRCUMCISION_DATE] = self.date

    def calc_prob_circ(self, age_group):
        """
        Calculates the circumcision probability for a given
        age group and returns it.
        """
        age_mod = 1
        # age group 1 has no modifier in most cases
        if age_group == 2:
            age_mod = self.circ_rate_change_20_29
        elif age_group == 3:
            age_mod = self.circ_rate_change_30_49

        calc_date = self.date.year
        # cap date at prob_circ_calc_cutoff_year (2019 by default) for calculations
        if self.prob_circ_calc_cutoff_year < self.date.year:
            calc_date = self.prob_circ_calc_cutoff_year

        # circumcision probability for a given age group
        # year is after circ_rate_change_year (2013 by default)
        if self.circ_rate_change_year < self.date.year:
            # case where age group 1 has a modifier
            ag1_has_mod = (age_group == 1) & (self.circ_policy_scenario == 1) \
                          & (self.policy_intervention_year <= self.date.year)
            if ag1_has_mod:
                prob_circ = ((self.circ_rate_change_year - self.vmmc_start_year)
                             + (calc_date - self.circ_rate_change_year)
                             * self.circ_rate_change_post_2013 * self.circ_rate_change_15_19) \
                             * self.circ_increase_rate
            else:
                prob_circ = ((self.circ_rate_change_year - self.vmmc_start_year)
                             + (calc_date - self.circ_rate_change_year)
                             * self.circ_rate_change_post_2013) * self.circ_increase_rate * age_mod
        # year is before circ_rate_change_year (2013 by default)
        else:
            prob_circ = (calc_date - self.vmmc_start_year) * self.circ_increase_rate * age_mod

        return min(prob_circ, 1)

    def calc_circ_outcomes(self, age_group, size):
        """
        Uses the circumcision probability for a given
        age group to return VMMC outcomes.
        """
        prob_circ = self.calc_prob_circ(age_group)
        # outcomes
        r = rng.uniform(size=size)
        circumcision = r < prob_circ

        return circumcision
