from __future__ import annotations

import importlib.resources
import operator as op
from datetime import timedelta
from math import pow
from typing import TYPE_CHECKING

import numpy as np

import hivpy.column_names as col

from . import output
from .common import SexType, rng
from .pregnancy_data import PregnancyData

if TYPE_CHECKING:
    from .population import Population


class PregnancyModule:

    def __init__(self, **kwargs):

        self.output = output.simulation_output
        # init pregnancy data
        with importlib.resources.path("hivpy.data", "pregnancy.yaml") as data_path:
            self.p_data = PregnancyData(data_path)

        self.can_be_pregnant = self.p_data.can_be_pregnant
        self.rate_want_no_children = self.p_data.rate_want_no_children  # dependent on time step length
        self.date_pmtct = self.p_data.date_pmtct
        self.pmtct_inc_rate = self.p_data.pmtct_inc_rate
        self.fertility_factor = self.p_data.fertility_factor
        self.inc_cat = self.p_data.inc_cat.sample()
        self.prob_pregnancy_base = self.generate_prob_pregnancy_base()  # dependent on time step length
        self.rate_test_anc_inc = self.p_data.rate_test_anc_inc.sample()
        self.prob_birth_with_infected_child = self.p_data.prob_birth_with_infected_child.sample()
        self.max_children = self.p_data.max_children
        self.init_num_children_distributions = self.p_data.init_num_children_distributions
        self.prob_anc = 0
        self.prob_pmtct = 0

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

    def init_pregnancy(self, pop: Population):
        self.init_fertility(pop)
        pop.init_variable(col.PREGNANT, False)
        pop.init_variable(col.LAST_PREGNANCY_DATE, None)
        pop.init_variable(col.ANC, False)
        pop.init_variable(col.PMTCT, False)
        pop.init_variable(col.ART_NAIVE, True)
        self.init_num_children(pop)
        pop.init_variable(col.NUM_HIV_CHILDREN, 0)
        pop.init_variable(col.WANT_NO_CHILDREN, False)

    def init_fertility(self, pop: Population):
        """
        Initialise who has a nonzero chance of getting pregnant
        for the entire female population.
        """
        pop.init_variable(col.LOW_FERTILITY, False)
        female_population = pop.get_sub_pop([(col.SEX, op.eq, SexType.Female)])
        r = rng.uniform(size=len(female_population))
        fertility = r > self.can_be_pregnant
        pop.set_present_variable(col.LOW_FERTILITY, fertility, female_population)

    def init_num_children(self, pop: Population):
        """
        Initialise the number of children each female individual
        at or above the age of 15 starts out with.
        """
        pop.init_variable(col.NUM_CHILDREN, 0)
        # get fertile female population above age 14
        female_population = pop.get_sub_pop([(col.SEX, op.eq, SexType.Female),
                                             (col.LOW_FERTILITY, op.eq, False),
                                             (col.AGE, op.ge, 15)])
        # group females by age groups
        age_groups = np.digitize(pop.get_variable(col.AGE, female_population),
                                 [15, 25, 35, 45])
        pop.set_present_variable(col.AGE_GROUP, age_groups, female_population)
        # outcomes
        pop.set_variable_by_group(col.NUM_CHILDREN, [col.AGE_GROUP], self.calc_init_num_children_outcomes,
                                  sub_pop=female_population)
        # give everyone with a child a pregnancy date before the start of the simulation
        pop.set_present_variable(col.LAST_PREGNANCY_DATE, pop.date - timedelta(days=270),
                                 sub_pop=pop.get_sub_pop([(col.NUM_CHILDREN, op.gt, 0)]))

    def calc_init_num_children_outcomes(self, age_group, size):
        """
        Uses the probability distribution for a given age group to return
        outcomes for each individual's initial number of children.
        """
        index = int(age_group)-1
        outcomes = self.init_num_children_distributions[index].sample(size)
        return outcomes

    def update_pregnancy(self, pop: Population, time_step: timedelta):
        """
        Monitor pregnancies and model childbirth.
        """
        # TODO: this is needed for stp prob preg reduction but should
        # probably be moved elsewhere so it's only initialised once
        self.stp_transmission_factor = pop.hiv_status.stp_transmission_factor

        # get sexually active female population to check for new pregnancies
        can_get_pregnant = pop.get_sub_pop([(col.SEX, op.eq, SexType.Female),
                                            (col.AGE, op.ge, 15),
                                            (col.AGE, op.lt, 55),
                                            (col.LOW_FERTILITY, op.eq, False),
                                            (col.PREGNANT, op.eq, False),
                                            (col.NUM_CHILDREN, op.lt, self.max_children),
                                            [(col.NUM_PARTNERS, op.gt, 0), (col.LONG_TERM_PARTNER, op.eq, True)],
                                            [(col.LAST_PREGNANCY_DATE, op.eq, None),
                                             (col.LAST_PREGNANCY_DATE, op.le, pop.date - timedelta(days=450))
                                             ]
                                            ])

        # continue if there are women who can become pregnant in this time step
        if len(can_get_pregnant) > 0:
            # group females by age groups
            age_groups = np.digitize(pop.get_variable(col.AGE, can_get_pregnant),
                                     [15, 25, 35, 45, 55])
            # TODO: change age group col name to be more descriptive
            pop.set_present_variable(col.AGE_GROUP, age_groups, can_get_pregnant)
            # calculate pregnancy outcomes
            pregnancy = pop.transform_group([col.AGE_GROUP, col.LONG_TERM_PARTNER,
                                             col.NUM_PARTNERS, col.WANT_NO_CHILDREN],
                                            self.calc_preg_outcomes,
                                            sub_pop=can_get_pregnant)
            # assign outcomes
            pop.set_present_variable(col.PREGNANT, pregnancy, can_get_pregnant)
            # use pregnancy outcomes as mask to assign current date as pregnancy date
            pop.set_present_variable(col.LAST_PREGNANCY_DATE,
                                     pop.date,
                                     pop.apply_bool_mask(pregnancy, can_get_pregnant))

        self.update_antenatal_care(pop, time_step)
        self.update_births(pop)
        self.update_want_no_children(pop)

    def update_antenatal_care(self, pop: Population, time_step: timedelta):
        """
        Determine who is in antenatal care and receiving
        prevention of mother to child transmission care.
        """
        # get population that became pregnant this time step
        pregnant_population = pop.get_sub_pop([(col.PREGNANT, op.eq, True),
                                               (col.LAST_PREGNANCY_DATE, op.eq, pop.date)])
        # update probability of antenatal care attendance
        self.prob_anc = min(max(self.prob_anc, 0.1) + self.rate_test_anc_inc, 0.975)

        # anc outcomes
        r = rng.uniform(size=len(pregnant_population))
        anc = r < self.prob_anc
        pop.set_present_variable(col.ANC, anc, pregnant_population)

        # anc testing
        pop.hiv_testing.update_anc_hiv_testing(pop, time_step)

        # FIXME: this should probably only be applied to HIV diagnosed individuals?
        # If date is after introduction of prevention of mother to child transmission
        if pop.date.year >= self.date_pmtct:
            # probability of prevention of mother to child transmission care
            self.prob_pmtct = min((pop.date.year - self.date_pmtct) * self.pmtct_inc_rate, 0.975)
            # FIXME: NVP use hasn't been modelled yet and neither has drug resistance
            # this expression assumed ANC can only be true if pregnant
            in_anc = pop.get_sub_pop([(col.ART_NAIVE, op.eq, True),
                                      (col.ANC, op.eq, True)])
            # pmtct outcomes
            r = rng.uniform(size=len(in_anc))
            pmtct = r < self.prob_pmtct
            pop.set_present_variable(col.PMTCT, pmtct, in_anc)

    def update_births(self, pop: Population):
        """
        Model pregnancies that come to term, including
        births with infected children.
        """
        # get population who give birth this time step
        birthing_population = pop.get_sub_pop([(col.PREGNANT, op.eq, True),
                                               (col.LAST_PREGNANCY_DATE, op.le, pop.date - timedelta(days=270))])

        # continue if births occur this time step
        if len(birthing_population) > 0:
            # remove pregnancy status
            pop.set_present_variable(col.PREGNANT, False, birthing_population)
            # remove from antenatal care
            pop.set_present_variable(col.ANC, False, birthing_population)
            # add to children
            pop.set_present_variable(col.NUM_CHILDREN, pop.get_variable(col.NUM_CHILDREN)+1, birthing_population)
            # birth with infected child
            infected_birthing_pop = pop.get_sub_pop_intersection(birthing_population,
                                                                 pop.get_sub_pop([(col.HIV_STATUS, op.eq, True)]))
            # calculate infected pregnancy outcomes
            infected_children = pop.transform_group([col.VIRAL_LOAD_GROUP],
                                                    self.calc_infected_birth_outcomes,
                                                    sub_pop=infected_birthing_pop)
            # add to infected children
            pop.set_present_variable(col.NUM_HIV_CHILDREN,
                                     pop.get_variable(col.NUM_HIV_CHILDREN)+1,
                                     pop.apply_bool_mask(infected_children, infected_birthing_pop))
            # infected newborns
            self.output.infected_newborns = len(infected_children)

            # FIXME: drug resistance in HIV mutations not modelled yet

    def update_want_no_children(self, pop: Population):
        """
        Increase the number of female individuals that
        don't want any more children each time step.
        """
        # TODO: should this have an init?
        # ideally yes, but task is low-priority
        # could apply rate at the start a number of times proportional to age
        want_children_population = pop.get_sub_pop([(col.SEX, op.eq, SexType.Female),
                                                    (col.AGE, op.ge, 25),
                                                    (col.AGE, op.lt, 55),
                                                    (col.WANT_NO_CHILDREN, op.eq, False)])
        # continue if those who want children are present this time step
        if len(want_children_population) > 0:
            # calculate outcomes
            r = rng.uniform(size=len(want_children_population))
            want_no_children = r < self.rate_want_no_children
            # assign outcomes
            pop.set_present_variable(col.WANT_NO_CHILDREN, want_no_children, want_children_population)

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
        base_prob_adjusted = self.prob_pregnancy_base * self.fertility_factor[int(age_group)-1]
        # wanting no more children decreases pregnancy probability by 80%
        if want_no_children:
            base_prob_adjusted *= 0.2
        # chance of not getting pregnant from a long-term partner
        if ltp:
            ltp_prob_no_preg = 1 - base_prob_adjusted
        # chance of not getting pregnant from all short-term partners
        if stp > 0:
            # apply short-term partner reduction
            stp_prob_no_preg = pow(1 - base_prob_adjusted * self.stp_transmission_factor, stp)
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
        infected_children = r < self.prob_birth_with_infected_child

        return infected_children
