import importlib.resources
from datetime import timedelta

import numpy as np

import hivpy.column_names as col

from .common import SexType, rng
from .pregnancy_data import PregnancyData


class PregnancyModule:

    def __init__(self, **kwargs):

        # init pregnancy data
        with importlib.resources.path("hivpy.data", "pregnancy.yaml") as data_path:
            self.c_data = PregnancyData(data_path)

        self.can_be_pregnant = self.c_data.can_be_pregnant
        self.fold_preg = self.c_data.fold_preg
        self.inc_cat = self.c_data.inc_cat.sample()
        self.rate_birth_with_infected_child = self.c_data.rate_birth_with_infected_child.sample()
        self.prob_pregnancy_base = self.generate_prob_pregnancy_base()

    def generate_prob_pregnancy_base(self):
        """
        Determine the base probability of pregnancy and
        return it (rounded to 3 decimal places).
        """
        prob_pregnancy_base = 0.06 + rng.uniform() * 0.05
        if self.inc_cat == 1:
            prob_pregnancy_base *= 1.75
        elif self.inc_cat == 3:
            prob_pregnancy_base /= 1.75
        return round(prob_pregnancy_base, 3)

    def init_fertility(self, pop):
        """
        Initialise who has a nonzero chance of getting pregnant
        for the entire female population.
        """
        female_population = pop.data.index[pop.data[col.SEX] == SexType.Female]
        r = rng.uniform(size=len(female_population))
        fertility = r > self.can_be_pregnant
        pop.data.loc[female_population, col.LOW_FERTILITY] = fertility

    def update_pregnancy(self, pop):
        """
        Monitor pregnancies and model childbirth.
        """
        # get sexually active female population
        # to check for new pregnancies
        # TODO: need to also check nobody selected has been pregnant in the last 6 months
        active_female_population = pop.data.index[(pop.data[col.SEX] == SexType.Female)
                                                  & (pop.data[col.AGE] >= 15)
                                                  & (pop.data[col.AGE] < 55)
                                                  & (~pop.data[col.LOW_FERTILITY])
                                                  & (pop.data[col.PREGNANCY_DATE].isnull())
                                                  & ((pop.data[col.NUM_PARTNERS] > 0)
                                                     | pop.data[col.LONG_TERM_PARTNER])]
        # get population ready for childbirth
        pregnant_population = pop.data[pop.data[col.PREGNANCY_DATE].notnull()]
        # TODO: may need to include time step period in the check below
        birthing_population = pregnant_population.index[pop.date -
                                                        pregnant_population[col.PREGNANCY_DATE]
                                                        >= timedelta(days=270)]

        # continue if sexually active females are present this timestep
        if len(active_female_population) > 0:

            # group females by age groups
            age_groups = np.digitize(pop.data.loc[active_female_population, col.AGE],
                                     [15, 25, 35, 45, 55])
            # TODO: change age group col name to be more descriptive
            pop.data.loc[active_female_population, col.AGE_GROUP] = age_groups
            # calculate pregnancy outcomes
            pregnancy = pop.transform_group([col.AGE_GROUP], self.calc_preg_outcomes,
                                            sub_pop=active_female_population)
            # use pregnancy outcomes as mask to assign current date as pregnancy date
            pop.data.loc[active_female_population[pregnancy], col.PREGNANCY_DATE] = pop.date

        # continue if births occur this time step
        if len(birthing_population) > 0:
            # remove pregnancy status
            pop.data.loc[birthing_population, col.PREGNANCY_DATE] = None

    def calc_prob_preg(self, age_group):
        """
        Calculates the probability of getting pregnant
        for a given age group and returns it.
        """
        prob_preg = self.prob_pregnancy_base * self.fold_preg[int(age_group)-1]
        # if want no more children, prob_preg/5
        return prob_preg

    def calc_preg_outcomes(self, age_group, size):
        """
        Uses the pregnancy probability for a given
        age group to return pregnancy outcomes.
        """
        prob_preg = self.calc_prob_preg(age_group)
        # outcomes
        r = rng.uniform(size=size)
        pregnancy = r < prob_preg

        return pregnancy
