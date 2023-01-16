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
        self.rate_want_no_children = 0.005
        self.max_children = 10

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
        active_female_population = pop.data[(pop.data[col.SEX] == SexType.Female)
                                            & (pop.data[col.AGE] >= 15)
                                            & (pop.data[col.AGE] < 55)
                                            & (~pop.data[col.LOW_FERTILITY])
                                            & (~pop.data[col.PREGNANT])
                                            & (pop.data[col.NUM_CHILDREN] < self.max_children)
                                            & ((pop.data[col.NUM_PARTNERS] > 0)
                                                | pop.data[col.LONG_TERM_PARTNER])]
        # get population ready for childbirth
        pregnant_population = pop.data[pop.data[col.PREGNANT]]
        # TODO: may need to include time step period in the check below
        birthing_population = pregnant_population.index[pop.date -
                                                        pregnant_population[col.LAST_PREGNANCY_DATE]
                                                        >= timedelta(days=270)]

        # continue if sexually active females are present this timestep
        if len(active_female_population) > 0:

            # at least 6 months must pass since last childbirth
            # (15 months since last pregnancy start date)
            previously_pregnant = active_female_population[active_female_population[col.LAST_PREGNANCY_DATE].notnull()]
            pregnancy_ready = ~active_female_population[col.LOW_FERTILITY]
            if len(previously_pregnant) > 0:
                # mask
                pregnancy_ready = pregnancy_ready & (previously_pregnant[col.LAST_PREGNANCY_DATE] + timedelta(days=450) <= pop.date)

            # group females by age groups
            age_groups = np.digitize(pop.data.loc[active_female_population[pregnancy_ready].index, col.AGE],
                                     [15, 25, 35, 45, 55])
            # TODO: change age group col name to be more descriptive
            pop.data.loc[active_female_population[pregnancy_ready].index, col.AGE_GROUP] = age_groups
            # calculate pregnancy outcomes
            pregnancy = pop.transform_group([col.AGE_GROUP], self.calc_preg_outcomes,
                                            sub_pop=active_female_population[pregnancy_ready].index)
            # TODO: wanting no more children decreases pregnancy probability by 80%
            # assign outcomes
            pop.data.loc[active_female_population[pregnancy_ready].index, col.PREGNANT] = pregnancy
            # use pregnancy outcomes as mask to assign current date as pregnancy date
            pop.data.loc[active_female_population[pregnancy_ready].index[pregnancy], col.LAST_PREGNANCY_DATE] = pop.date

        # continue if births occur this time step
        if len(birthing_population) > 0:
            # remove pregnancy status
            pop.data.loc[birthing_population, col.PREGNANT] = False
            # add to children
            pop.data.loc[birthing_population, col.NUM_CHILDREN] += 1

        # increase number of women that want no more children
        want_children_population = pop.data.index[(pop.data[col.SEX] == SexType.Female)
                                                  & (pop.data[col.AGE] >= 25)
                                                  & (pop.data[col.AGE] < 55)
                                                  & (~pop.data[col.WANT_NO_CHILDREN])]
        # continue if those who want children are present this time step
        if len(want_children_population) > 0:
            # calculate outcomes
            r = rng.uniform(size=len(want_children_population))
            want_no_children = r < self.rate_want_no_children
            # assign outcomes
            pop.data.loc[want_children_population, col.WANT_NO_CHILDREN] = want_no_children

    def calc_prob_preg(self, age_group):
        """
        Calculates the probability of getting pregnant
        for a given age group and returns it.
        """
        return self.prob_pregnancy_base * self.fold_preg[int(age_group)-1]

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
