import importlib.resources
from datetime import timedelta
from math import pow

import numpy as np

import hivpy.column_names as col

from .common import SexType, rng
from .pregnancy_data import PregnancyData


class PregnancyModule:

    def __init__(self, **kwargs):

        # init pregnancy data
        with importlib.resources.path("hivpy.data", "pregnancy.yaml") as data_path:
            self.p_data = PregnancyData(data_path)

        self.can_be_pregnant = self.p_data.can_be_pregnant
        self.rate_want_no_children = self.p_data.rate_want_no_children  # dependent on time step length
        self.date_pmtct = self.p_data.date_pmtct
        self.pmtct_inc_rate = self.p_data.pmtct_inc_rate
        self.fold_preg = self.p_data.fold_preg
        self.inc_cat = self.p_data.inc_cat.sample()
        self.rate_testanc_inc = self.p_data.rate_testanc_inc.sample()
        self.prob_pregnancy_base = self.generate_prob_pregnancy_base()  # dependent on time step length
        self.rate_birth_with_infected_child = self.p_data.rate_birth_with_infected_child.sample()
        self.max_children = self.p_data.max_children
        self.init_num_children_distributions = self.p_data.init_num_children_distributions
        self.prob_anc = 0

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

    def init_num_children(self, pop):
        """
        Initialise the number of children each female individual
        at or above the age of 15 starts out with.
        """
        # get fertile female population above age 14
        female_population = pop.data.index[(pop.data[col.SEX] == SexType.Female)
                                           & (~pop.data[col.LOW_FERTILITY])
                                           & (pop.data[col.AGE] >= 15)]
        # group females by age groups
        age_groups = np.digitize(pop.data.loc[female_population, col.AGE],
                                 [15, 25, 35, 45])
        pop.data.loc[female_population, col.AGE_GROUP] = age_groups
        # outcomes
        num_children = pop.transform_group([col.AGE_GROUP], self.calc_init_num_children_outcomes,
                                           sub_pop=female_population)
        # assign outcomes
        pop.data.loc[female_population, col.NUM_CHILDREN] = num_children
        # give everyone with a child a pregnancy date before the start of the simulation
        pop.data.loc[pop.data[col.NUM_CHILDREN] > 0, col.LAST_PREGNANCY_DATE] = pop.date - timedelta(days=270)

    def calc_init_num_children_outcomes(self, age_group, size):
        """
        Uses the probability distribution for a given age group to return
        outcomes for each individual's initial number of children.
        """
        index = int(age_group)-1
        outcomes = self.init_num_children_distributions[index].sample(size)
        return outcomes

    def update_pregnancy(self, pop):
        """
        Monitor pregnancies and model childbirth.
        """
        # TODO: this is needed for stp prob preg reduction but should
        # probably be moved elsewhere so it's only initialised once
        self.fold_tr_newp = pop.hiv_status.fold_tr_newp
        # get sexually active female population to check for new pregnancies
        active_female_population = pop.data[(pop.data[col.SEX] == SexType.Female)
                                            & (pop.data[col.AGE] >= 15)
                                            & (pop.data[col.AGE] < 55)
                                            & (~pop.data[col.LOW_FERTILITY])
                                            & (~pop.data[col.PREGNANT])
                                            & (pop.data[col.NUM_CHILDREN] < self.max_children)
                                            & ((pop.data[col.NUM_PARTNERS] > 0)
                                                | pop.data[col.LONG_TERM_PARTNER])]

        # continue if sexually active females are present this timestep
        if len(active_female_population) > 0:

            # at least 6 months must pass since last childbirth
            # (15 months since last pregnancy start date)
            previously_pregnant = active_female_population[active_female_population[col.LAST_PREGNANCY_DATE].notnull()]
            pregnancy_ready = ~active_female_population[col.LOW_FERTILITY]
            if len(previously_pregnant) > 0:
                # mask
                pregnancy_ready = pregnancy_ready & (previously_pregnant[col.LAST_PREGNANCY_DATE]
                                                     + timedelta(days=450) <= pop.date)

            # group females by age groups
            age_groups = np.digitize(pop.data.loc[active_female_population[pregnancy_ready].index, col.AGE],
                                     [15, 25, 35, 45, 55])
            # TODO: change age group col name to be more descriptive
            pop.data.loc[active_female_population[pregnancy_ready].index, col.AGE_GROUP] = age_groups
            # calculate pregnancy outcomes
            pregnancy = pop.transform_group([col.AGE_GROUP, col.LONG_TERM_PARTNER,
                                             col.NUM_PARTNERS, col.WANT_NO_CHILDREN],
                                            self.calc_preg_outcomes,
                                            sub_pop=active_female_population[pregnancy_ready].index)
            # assign outcomes
            pop.data.loc[active_female_population[pregnancy_ready].index, col.PREGNANT] = pregnancy
            # use pregnancy outcomes as mask to assign current date as pregnancy date
            pop.data.loc[active_female_population[pregnancy_ready].index[pregnancy], col.LAST_PREGNANCY_DATE] = pop.date

        self.update_antenatal_care_testing(pop)
        self.update_births(pop)
        self.update_want_no_children(pop)

    def update_antenatal_care_testing(self, pop):
        """
        Determine who is in antenatal care and receiving
        prevention of mother to child transmission care.
        """
        # get pregnant population
        pregnant_population = pop.data[pop.data[col.PREGNANT]]
        # update probability of antenatal care attendance
        self.prob_anc = min(max(self.prob_anc, 0.1) + self.rate_testanc_inc, 0.975)

        # anc outcomes
        r = rng.uniform(size=len(pregnant_population))
        anc = r < self.prob_anc
        pop.data.loc[pregnant_population.index, col.ANC] = anc

        # update pregnant population
        pregnant_population = pop.data[pop.data[col.PREGNANT]]

        # FIXME: this should probably only be applied to HIV+ individuals
        if pop.date.year >= self.date_pmtct:
            # probability of prevention of mother to child transmission care
            prob_pmtct = min((pop.date.year - self.date_pmtct) * self.pmtct_inc_rate, 0.975)
            # FIXME: NVP use hasn't been modelled yet and neither has drug resistance
            in_anc = pregnant_population[pregnant_population[col.ART_NAIVE]
                                         & pregnant_population[col.ANC]]
            # pmtct outcomes
            r = rng.uniform(size=len(in_anc))
            pmtct = r < prob_pmtct
            pop.data.loc[in_anc.index, col.PMTCT] = pmtct

    def update_births(self, pop):
        """
        Model pregnancies that come to term, including
        births with infected children.
        """
        # get pregnant population
        pregnant_population = pop.data[pop.data[col.PREGNANT]]
        # get population ready for childbirth
        birthing_population = pregnant_population[pop.date -
                                                  pregnant_population[col.LAST_PREGNANCY_DATE]
                                                  >= timedelta(days=270)]

        # continue if births occur this time step
        if len(birthing_population) > 0:
            # remove pregnancy status
            pop.data.loc[birthing_population.index, col.PREGNANT] = False
            # add to children
            pop.data.loc[birthing_population.index, col.NUM_CHILDREN] += 1
            # birth with infected child
            infected_birthing_population = birthing_population[birthing_population[col.HIV_STATUS]]
            # calculate infected pregnancy outcomes
            infected_children = pop.transform_group([col.VIRAL_LOAD_GROUP],
                                                    self.calc_infected_birth_outcomes,
                                                    sub_pop=infected_birthing_population.index)
            # add to infected children
            pop.data.loc[infected_birthing_population[infected_children].index, col.NUM_HIV_CHILDREN] += 1
            # FIXME: drug resistance in HIV mutations not modelled yet

    def update_want_no_children(self, pop):
        """
        Increase the number of female individuals that
        don't want any more children each time step.
        """
        # TODO: should this have an init?
        # ideally yes, but task is low-priority
        # could apply rate at the start a number of times proportional to age
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

    def calc_prob_preg(self, age_group, ltp, stp, want_no_children):
        """
        Calculates the probability of getting pregnant for a group
        with specific characteristics and returns it. Age, number
        of condomless sex partners, and the desire to have no more
        children all affect groupings and pregnancy probability.
        """
        # initial values (no chance of pregnancy)
        ltp_prob_no_preg = 1
        stp_prob_no_preg = 1
        # base probability adjusted according to age factor
        base_prob_adjusted = self.prob_pregnancy_base * self.fold_preg[int(age_group)-1]
        # wanting no more children decreases pregnancy probability by 80%
        if want_no_children:
            base_prob_adjusted *= 0.2
        # chance of not getting pregnant from a long-term partner
        if ltp:
            ltp_prob_no_preg = 1 - base_prob_adjusted
        # chance of not getting pregnant from all short-term partners
        if stp > 0:
            # apply short-term partner reduction
            stp_prob_no_preg = pow(1 - base_prob_adjusted * self.fold_tr_newp, stp)
        # total probability of no pregnancy
        prob_all_no_preg = ltp_prob_no_preg * stp_prob_no_preg
        # probability of at least one encounter resulting in pregnancy
        prob_preg = 1 - prob_all_no_preg
        return min(prob_preg, 1)

    def calc_preg_outcomes(self, age_group, ltp, stp, want_no_children, size):
        """
        Uses the pregnancy probability for a given
        age group to return pregnancy outcomes.
        """
        prob_preg = self.calc_prob_preg(age_group, ltp, stp, want_no_children)
        # outcomes
        r = rng.uniform(size=size)
        pregnancy = r < prob_preg

        return pregnancy

    def calc_infected_birth_outcomes(self, viral_load, size):
        """
        Determines whether a birth results in an infected
        child based on the mother's viral load.
        """
        vl_multiplier = 1
        if viral_load <= 3:
            vl_multiplier = 1000
        elif viral_load <= 4:
            vl_multiplier = 2
        elif viral_load > 5:
            vl_multiplier = 0.5
        # outcomes
        r = rng.uniform(size=size) * vl_multiplier
        infected_children = r < self.rate_birth_with_infected_child

        return infected_children
