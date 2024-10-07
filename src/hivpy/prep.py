from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .population import Population

import importlib.resources
import operator as op
from enum import IntEnum

import pandas as pd

import hivpy.column_names as col

from .common import AND, COND, OR, SexType, date, rng, timedelta
from .prep_data import PrEPData


class PrEPType(IntEnum):
    Oral = 0
    Cabotegravir = 1  # injectable
    Lenacapavir = 2  # injectable
    VaginalRing = 3


class PrEPModule:

    def __init__(self, **kwargs):

        # init prep data
        with importlib.resources.path("hivpy.data", "prep.yaml") as data_path:
            self.p_data = PrEPData(data_path)

        self.prep_strategy = self.p_data.prep_strategy.sample()
        self.date_prep_intro = [date(self.p_data.date_prep_oral_intro),
                                date(self.p_data.date_prep_cab_intro),
                                date(self.p_data.date_prep_len_intro),
                                date(self.p_data.date_prep_vr_intro)]
        self.prob_risk_informed_prep = self.p_data.prob_risk_informed_prep
        self.prob_greater_risk_informed_prep = self.p_data.prob_greater_risk_informed_prep
        self.prob_suspect_risk_prep = self.p_data.prob_suspect_risk_prep

        self.prep_oral_pref_beta = rng.choice([1.1, 1.3, 1.5])
        self.prep_cab_pref_beta = self.prep_oral_pref_beta + 0.3
        self.prep_len_pref_beta = self.prep_cab_pref_beta
        self.prep_vr_pref_beta = self.prep_oral_pref_beta - 0.1
        self.vl_prevalence_affects_prep = rng.choice([True, False], p=[1/3, 2/3])
        self.vl_prevalence_prep_threshold = rng.choice([0.005, 0.01])

        self.rate_test_onprep_any = self.p_data.rate_test_onprep_any
        self.prep_willing_threshold = self.p_data.prep_willing_threshold
        self.prob_test_prep_start = self.p_data.prob_test_prep_start.sample()
        self.prob_prep_restart = self.p_data.prob_prep_restart.sample()
        # FIXME: add to yaml
        # probability of starting prep in people who are eligible and willing
        # tested for HIV according to base rate of testing
        self.prob_base_prep_start = rng.choice([0.05, 0.1, 0.2])

    def init_prep_variables(self, pop: Population):
        pop.init_variable(col.PREP_ORAL_PREF, 0)
        pop.init_variable(col.PREP_CAB_PREF, 0)
        pop.init_variable(col.PREP_LEN_PREF, 0)
        pop.init_variable(col.PREP_VR_PREF, 0)
        pop.init_variable(col.PREP_ORAL_RANK, 0)
        pop.init_variable(col.PREP_CAB_RANK, 0)
        pop.init_variable(col.PREP_LEN_RANK, 0)
        pop.init_variable(col.PREP_VR_RANK, 0)
        pop.init_variable(col.PREP_ORAL_WILLING, False)
        pop.init_variable(col.PREP_CAB_WILLING, False)
        pop.init_variable(col.PREP_LEN_WILLING, False)
        pop.init_variable(col.PREP_VR_WILLING, False)
        pop.init_variable(col.PREP_ANY_WILLING, False)
        pop.init_variable(col.R_PREP, 1.0)
        pop.init_variable(col.PREP_ELIGIBLE, False)
        pop.init_variable(col.PREP_TYPE, None)
        pop.init_variable(col.EVER_PREP, False)
        pop.init_variable(col.FIRST_ORAL_START_DATE, None)
        pop.init_variable(col.FIRST_CAB_START_DATE, None)
        pop.init_variable(col.FIRST_LEN_START_DATE, None)
        pop.init_variable(col.FIRST_VR_START_DATE, None)
        pop.init_variable(col.LAST_PREP_START_DATE, None)
        pop.init_variable(col.PREP_JUST_STARTED, False)
        pop.init_variable(col.PREP_ORAL_TESTED, False)
        pop.init_variable(col.PREP_CAB_TESTED, False)
        pop.init_variable(col.PREP_LEN_TESTED, False)
        pop.init_variable(col.PREP_VR_TESTED, False)
        pop.init_variable(col.LTP_HIV_STATUS, False)
        pop.init_variable(col.LTP_HIV_DIAGNOSED, False)
        pop.init_variable(col.LTP_ON_ART, False)

    def reroll_r_prep(self, pop: Population):
        """
        Reroll the r_prep value for each individual that was ineligible for PrEP last time step.
        """
        ineligible_pop = pop.get_sub_pop(COND(col.PREP_ELIGIBLE, op.eq, False))
        pop.set_present_variable(col.R_PREP, rng.uniform(size=len(ineligible_pop)), sub_pop=ineligible_pop)

    def get_at_risk_pop(self, pop: Population):
        """
        Return the sub-population that either has one or more short-term partners or
        has a diagnosed long-term partner who is not on ART.
        """
        return pop.get_sub_pop(OR(COND(col.NUM_PARTNERS, op.ge, 1),
                                  AND(COND(col.LTP_HIV_DIAGNOSED, op.eq, True),
                                      COND(col.LTP_ON_ART, op.eq, False))))

    def get_risk_informed_pop(self, pop: Population, prob_risk_informed_prep):
        """
        Return the sub-population that has a long-term partner who is not on ART
        and pass the probability to fulfill the criteria for risk-informed PrEP.
        """
        return pop.get_sub_pop(AND(COND(col.LONG_TERM_PARTNER, op.eq, True),
                                   COND(col.LTP_ON_ART, op.eq, False),
                                   COND(col.LTP_HIV_STATUS, op.eq, False),
                                   COND(col.R_PREP, op.lt, prob_risk_informed_prep)))

    def get_suspect_risk_pop(self, pop: Population):
        """
        Return the sub-population that has a long-term partner who is not on ART but is infected
        and pass the higher probability to fulfill the criteria for risk-informed PrEP.
        """
        return pop.get_sub_pop(AND(COND(col.LONG_TERM_PARTNER, op.eq, True),
                                   COND(col.LTP_ON_ART, op.eq, False),
                                   COND(col.LTP_HIV_STATUS, op.eq, True),
                                   COND(col.R_PREP, op.lt, self.prob_suspect_risk_prep)))

    def get_presumed_hiv_neg_pop(self, pop: Population):
        """
        Return the sub-population that has been tested and is HIV positive but
        received a false negative result.
        """
        false_neg_pop = pop.get_sub_pop(AND(COND(col.EVER_TESTED, op.eq, True),
                                            COND(col.HIV_STATUS, op.eq, True)))

        # general test sensitivity
        eff_test_sens = pop.hiv_diagnosis.test_sens_general
        if not pop.hiv_diagnosis.init_prep_inj_na:
            # infected up to 3 months ago
            recently_infected_pop = pop.get_sub_pop_intersection(
                pop.get_sub_pop(COND(col.DATE_HIV_INFECTION, op.le, pop.date - timedelta(months=3))), false_neg_pop)

            # expand sensitivity into a list
            eff_test_sens = [pop.hiv_diagnosis.test_sens_general] * len(false_neg_pop)
            false_neg_list = list(false_neg_pop)
            # find indices in false_neg_pop that correspond to people belonging to recently_infected_pop
            common_i = [false_neg_list.index(i) for i in false_neg_list if i in recently_infected_pop]

            # FIXME: is there a better way to do this?
            for i in common_i:
                # lower test sensitivity used to mimic more people starting prep when they have hiv
                eff_test_sens[i] = pop.hiv_diagnosis.test_sens_primary_ab

        # false negative outcomes
        r = rng.uniform(size=len(false_neg_pop))
        mask = r > eff_test_sens

        return pop.apply_bool_mask(mask, false_neg_pop)

    def set_prep_preference(self, pop: Population, date_intro, pref_beta, pref_col, willing_col, sub_pop_mod=None):
        """
        Set preference values for a specific type of PrEP and determine willingness.
        """
        if pop.date >= date_intro:
            # find those who turned 15 this time step
            sub_pop = pop.get_sub_pop([(col.AGE, op.eq, 15)])
            # unless the current date is the introduction date
            if pop.date == date_intro:
                # then find all over 15s
                sub_pop = pop.get_sub_pop([(col.AGE, op.ge, 15)])
            # find intersection if further modifications should be made to the sub-pop
            if sub_pop_mod is not None:
                sub_pop = pop.get_sub_pop_intersection(sub_pop, sub_pop_mod)

            # random preference beta distribution
            pref = rng.beta(pref_beta, 5, size=len(sub_pop))
            pop.set_present_variable(pref_col, pref, sub_pop)
            # determine willingness by comparing to threshold
            willingness = pref > self.prep_willing_threshold
            pop.set_present_variable(willing_col, willingness, sub_pop)
            pop.set_present_variable(col.PREP_ANY_WILLING, True, pop.apply_bool_mask(willingness, sub_pop))

    def prep_willingness(self, pop: Population):
        """
        Determine which individuals are willing to take PrEP, as well as their PrEP preferences.
        """
        # initial preference values
        init_prefs = pop.data[[col.PREP_ORAL_PREF, col.PREP_CAB_PREF, col.PREP_LEN_PREF, col.PREP_VR_PREF]]
        # oral prep pref + willingness
        self.set_prep_preference(pop, self.date_prep_intro[PrEPType.Oral], self.prep_oral_pref_beta,
                                 col.PREP_ORAL_PREF, col.PREP_ORAL_WILLING)
        # injectable prep pref + willingness
        # FIXME: should Cab be controlled by an availability flag instead of introduction date?
        self.set_prep_preference(pop, self.date_prep_intro[PrEPType.Cabotegravir], self.prep_cab_pref_beta,
                                 col.PREP_CAB_PREF, col.PREP_CAB_WILLING)
        self.set_prep_preference(pop, self.date_prep_intro[PrEPType.Lenacapavir], self.prep_len_pref_beta,
                                 col.PREP_LEN_PREF, col.PREP_LEN_WILLING)
        # vr prep pref + willingness (women only)
        self.set_prep_preference(pop, self.date_prep_intro[PrEPType.VaginalRing], self.prep_vr_pref_beta,
                                 col.PREP_VR_PREF, col.PREP_VR_WILLING,
                                 sub_pop_mod=pop.get_sub_pop([(col.SEX, op.eq, SexType.Female)]))

        # new preference values
        new_prefs = pop.data[[col.PREP_ORAL_PREF, col.PREP_CAB_PREF, col.PREP_LEN_PREF, col.PREP_VR_PREF]]
        # find people whose preference has changed this time step
        changed_pref_pop = new_prefs.compare(init_prefs).index

        if len(changed_pref_pop) > 0:
            # get ranking outcomes
            # FIXME: not sure if transform group is the best way to do this, but it works for now
            pref_ranks = pop.transform_group([col.PREP_ORAL_PREF, col.PREP_CAB_PREF,
                                              col.PREP_LEN_PREF, col.PREP_VR_PREF],
                                             self.calc_prep_pref_ranks, sub_pop=changed_pref_pop, use_size=False)
            # set ranks for each prep type
            pop.set_present_variable(col.PREP_ORAL_RANK, [i[0] for i in pref_ranks], changed_pref_pop)
            pop.set_present_variable(col.PREP_CAB_RANK, [i[1] for i in pref_ranks], changed_pref_pop)
            pop.set_present_variable(col.PREP_LEN_RANK, [i[2] for i in pref_ranks], changed_pref_pop)
            pop.set_present_variable(col.PREP_VR_RANK, [i[3] for i in pref_ranks], changed_pref_pop)

        gen_pop = len(pop.get_sub_pop([(col.AGE, op.ge, 15), (col.AGE, op.lt, 50)]))
        # find prevalence of people with a viral load of over 1000
        vl_prevalence = (len(pop.get_sub_pop([(col.VIRAL_LOAD, op.gt, 1000),
                                              (col.AGE, op.ge, 15),
                                              (col.AGE, op.lt, 50)])) / gen_pop
                         if gen_pop > 0 else 0)

        # there's a chance nobody is willing to take PrEP if unsuppressed viral load prevalence is too low
        if self.vl_prevalence_affects_prep and vl_prevalence < self.vl_prevalence_prep_threshold:
            pop.set_present_variable(col.PREP_ORAL_WILLING, False)
            pop.set_present_variable(col.PREP_CAB_WILLING, False)
            pop.set_present_variable(col.PREP_LEN_WILLING, False)
            pop.set_present_variable(col.PREP_VR_WILLING, False)
            pop.set_present_variable(col.PREP_ANY_WILLING, False)

    def calc_prep_pref_ranks(self, oral_pref, cab_pref, len_pref, vr_pref):
        """
        Returns PrEP preference rankings based on all preference values.
        """
        ranks = [0, 0, 0, 0]
        prefs = [oral_pref, cab_pref, len_pref, vr_pref]
        # reverse sort preference values (position indicates rank, value indicates prep type)
        sorted_pref_indices = sorted(range(len(prefs)), key=lambda x: prefs[x], reverse=True)
        # assign rank per prep type (position indicates prep type, value indicates rank)
        for i in range(len(prefs)):
            ranks[sorted_pref_indices[i]] = i+1
        return [ranks]

    def prep_eligibility(self, pop: Population):
        """
        Mark people who are eligible for PrEP this time step.
        """
        # start when first type of prep is introduced
        if pop.date >= min(self.date_prep_intro):

            prob_risk_informed_prep = (self.prob_greater_risk_informed_prep
                                       if (8 <= self.prep_strategy <= 11 or self.prep_strategy == 14)
                                       else self.prob_risk_informed_prep)

            # nobody is eligible by default
            prep_eligible_pop = pd.Index([], dtype="int64")
            # reroll r_prep for those that were ineligible last time step
            self.reroll_r_prep(pop)
            # reset old prep eligibility
            pop.set_present_variable(col.PREP_ELIGIBLE, False)

            # female sex workers + adolescent girls and young women
            if self.prep_strategy == 1:
                fsw_agyw_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                                   COND(col.SEX, op.eq, SexType.Female),
                                                   OR(COND(col.SEX_WORKER, op.eq, True),
                                                      AND(COND(col.AGE, op.ge, 15),
                                                          COND(col.AGE, op.lt, 25)))))
                # fsw_agyw AND (at_risk OR risk_informed OR suspect_risk)
                prep_eligible_pop = pop.get_sub_pop_intersection(
                    fsw_agyw_pop, pop.get_sub_pop_union(
                        self.get_at_risk_pop(pop),
                        self.get_risk_informed_pop(pop, prob_risk_informed_prep),
                        self.get_suspect_risk_pop(pop)))
            # female sex workers
            elif self.prep_strategy == 2:
                fsw_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                              COND(col.SEX, op.eq, SexType.Female),
                                              COND(col.SEX_WORKER, op.eq, True)))
                # fsw AND (at_risk OR risk_informed OR suspect_risk)
                prep_eligible_pop = pop.get_sub_pop_intersection(
                    fsw_pop, pop.get_sub_pop_union(
                        self.get_at_risk_pop(pop),
                        self.get_risk_informed_pop(pop, prob_risk_informed_prep),
                        self.get_suspect_risk_pop(pop)))
            # adolescent girls and young women
            elif self.prep_strategy == 3:
                agyw_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                               COND(col.SEX, op.eq, SexType.Female),
                                               COND(col.AGE, op.ge, 15),
                                               COND(col.AGE, op.lt, 25)))
                # agyw AND (at_risk OR risk_informed OR suspect_risk)
                prep_eligible_pop = pop.get_sub_pop_intersection(
                    agyw_pop, pop.get_sub_pop_union(
                        self.get_at_risk_pop(pop),
                        self.get_risk_informed_pop(pop, prob_risk_informed_prep),
                        self.get_suspect_risk_pop(pop)))
            # general at risk population and informed women
            elif self.prep_strategy == 4 or self.prep_strategy == 8:
                gen_fem_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                                  COND(col.SEX, op.eq, SexType.Female),
                                                  COND(col.AGE, op.ge, 15),
                                                  COND(col.AGE, op.lt, 50)))
                # at_risk OR (gen_fem AND (risk_informed OR suspect_risk))
                prep_eligible_pop = pop.get_sub_pop_union(
                    pop.get_sub_pop_intersection(
                        pop.get_sub_pop(COND(col.HIV_DIAGNOSED, op.eq, False)), self.get_at_risk_pop(pop)),
                    pop.get_sub_pop_intersection(
                        gen_fem_pop, pop.get_sub_pop_union(
                            self.get_risk_informed_pop(pop, prob_risk_informed_prep), self.get_suspect_risk_pop(pop))))
            # general recently active and informed population
            elif self.prep_strategy == 5 or self.prep_strategy == 9:
                gen_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                              COND(col.AGE, op.ge, 15),
                                              COND(col.AGE, op.lt, 50)))
                active_stp_pop = pop.get_sub_pop(COND(col.LAST_STP_DATE, op.gt, pop.date - timedelta(months=9)))
                # gen AND (active_stp OR risk_informed OR suspect_risk)
                prep_eligible_pop = pop.get_sub_pop_intersection(
                    gen_pop, pop.get_sub_pop_union(
                        active_stp_pop,
                        self.get_risk_informed_pop(pop, prob_risk_informed_prep),
                        self.get_suspect_risk_pop(pop)))
            # at risk and informed women
            elif self.prep_strategy == 6 or self.prep_strategy == 10:
                gen_fem_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                                  COND(col.SEX, op.eq, SexType.Female)))
                gen_age_pop = pop.get_sub_pop(AND(COND(col.AGE, op.ge, 15),
                                                  COND(col.AGE, op.lt, 50)))
                # gen_fem AND (at_risk OR (gen_age AND (risk_informed OR suspect_risk)))
                prep_eligible_pop = pop.get_sub_pop_intersection(
                    gen_fem_pop, pop.get_sub_pop_union(
                        self.get_at_risk_pop(pop), pop.get_sub_pop_intersection(
                            gen_age_pop, pop.get_sub_pop_union(
                                self.get_risk_informed_pop(pop, prob_risk_informed_prep),
                                self.get_suspect_risk_pop(pop)))))
            # recently active and informed women
            elif self.prep_strategy == 7 or self.prep_strategy == 11:
                gen_fem_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                                  COND(col.SEX, op.eq, SexType.Female),
                                                  COND(col.AGE, op.ge, 15),
                                                  COND(col.AGE, op.lt, 50)))
                active_stp_pop = pop.get_sub_pop(COND(col.LAST_STP_DATE, op.gt, pop.date - timedelta(months=9)))
                # gen_fem AND (active_stp OR risk_informed OR suspect_risk)
                prep_eligible_pop = pop.get_sub_pop_intersection(
                    gen_fem_pop, pop.get_sub_pop_union(
                        active_stp_pop,
                        self.get_risk_informed_pop(pop, prob_risk_informed_prep),
                        self.get_suspect_risk_pop(pop)))
            # general recently active population
            elif self.prep_strategy == 12:
                gen_pop = pop.get_sub_pop(COND(col.HIV_DIAGNOSED, op.eq, False))
                active_pop = pop.get_sub_pop(OR(COND(col.LONG_TERM_PARTNER, op.eq, True),
                                                COND(col.LAST_STP_DATE, op.gt, pop.date - timedelta(months=9))))
                # gen AND active
                prep_eligible_pop = pop.get_sub_pop_intersection(gen_pop, active_pop)
            # recently active women
            elif self.prep_strategy == 13:
                gen_fem_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                                  COND(col.SEX, op.eq, SexType.Female)))
                active_pop = pop.get_sub_pop(OR(COND(col.LONG_TERM_PARTNER, op.eq, True),
                                                COND(col.LAST_STP_DATE, op.gt, pop.date - timedelta(months=9))))
                # gen_fem AND active
                prep_eligible_pop = pop.get_sub_pop_intersection(gen_fem_pop, active_pop)
            # general active at risk population and informed women
            elif self.prep_strategy == 14:
                gen_fem_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                                  COND(col.SEX, op.eq, SexType.Female),
                                                  COND(col.AGE, op.ge, 15),
                                                  COND(col.AGE, op.lt, 50)))
                active_at_risk_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                                         OR(COND(col.LAST_STP_DATE, op.gt,
                                                                 pop.date - timedelta(months=6)),
                                                            AND(COND(col.LTP_HIV_DIAGNOSED, op.eq, True),
                                                                COND(col.LTP_ON_ART, op.eq, False)))))
                # active_at_risk OR (gen_fem AND (risk_informed OR suspect_risk))
                prep_eligible_pop = pop.get_sub_pop_union(
                    active_at_risk_pop, pop.get_sub_pop_intersection(
                        gen_fem_pop, pop.get_sub_pop_union(
                            self.get_risk_informed_pop(pop, prob_risk_informed_prep), self.get_suspect_risk_pop(pop))))
            # serodiscordant couples
            elif self.prep_strategy == 15:
                gen_ltp_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                                  COND(col.LONG_TERM_PARTNER, op.eq, True),
                                                  COND(col.LTP_HIV_DIAGNOSED, op.eq, False),
                                                  COND(col.AGE, op.ge, 15),
                                                  COND(col.AGE, op.lt, 50),
                                                  OR(COND(col.R_PREP, op.lt, 0.01),  # (alt) risk informed
                                                     AND(COND(col.R_PREP, op.lt, self.prob_suspect_risk_prep),
                                                         COND(col.LTP_HIV_STATUS, op.eq, True)))))  # (alt) suspect risk
                at_risk_ltp_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                                      COND(col.LTP_HIV_DIAGNOSED, op.eq, True),
                                                      COND(col.LTP_ON_ART, op.eq, False)))
                # at_risk_ltp OR gen_ltp
                prep_eligible_pop = pop.get_sub_pop_union(at_risk_ltp_pop, gen_ltp_pop)
            # pregnant and lactating/breastfeeding women
            elif self.prep_strategy == 16:
                plw_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                              COND(col.SEX, op.eq, SexType.Female),
                                              OR(COND(col.PREGNANT, op.eq, True),
                                                 COND(col.BREASTFEEDING, op.eq, True))))
                # plw AND (at_risk OR risk_informed OR suspect_risk)
                prep_eligible_pop = pop.get_sub_pop_intersection(
                    plw_pop, pop.get_sub_pop_union(
                        self.get_at_risk_pop(pop),
                        self.get_risk_informed_pop(pop, prob_risk_informed_prep),
                        self.get_suspect_risk_pop(pop)))

            if len(prep_eligible_pop) > 0:
                pop.set_present_variable(col.PREP_ELIGIBLE, True, prep_eligible_pop)

    def tested_start_prep(self, pop: Population, prep_eligible_pop, prep_type, prep_tested_col, first_start_col):
        """
        Update people starting PrEP for the first time after testing to start PrEP.
        """
        # only start if specific prep type has been introduced
        if pop.date >= self.date_prep_intro[prep_type]:
            # tested explicitly to start prep
            starting_prep_pop = pop.get_sub_pop_intersection(
                prep_eligible_pop, pop.get_sub_pop(COND(prep_tested_col, op.eq, True)))

            if len(starting_prep_pop) > 0:
                pop.set_present_variable(col.PREP_TYPE, prep_type, starting_prep_pop)
                pop.set_present_variable(col.EVER_PREP, True, starting_prep_pop)
                pop.set_present_variable(col.LAST_PREP_START_DATE, pop.date, starting_prep_pop)
                pop.set_present_variable(first_start_col, pop.date, starting_prep_pop)

    def general_start_prep(self, pop: Population, prep_eligible_pop):
        """
        Update people starting PrEP for the first time without specifically testing to start PrEP.
        """
        # not tested explicitly to start any prep
        starting_prep_pop = pop.get_sub_pop_intersection(
           prep_eligible_pop, pop.get_sub_pop(AND(COND(col.PREP_ORAL_TESTED, op.eq, False),
                                                  COND(col.PREP_CAB_TESTED, op.eq, False),
                                                  COND(col.PREP_LEN_TESTED, op.eq, False),
                                                  COND(col.PREP_VR_TESTED, op.eq, False))))

        if len(starting_prep_pop) > 0:
            # FIXME: can we pass the date to transform_group in a better way?
            self.date = pop.date
            # starting prep outcomes
            prep_types = pop.transform_group([col.PREP_ORAL_RANK, col.PREP_CAB_RANK,
                                              col.PREP_LEN_RANK, col.PREP_VR_RANK,
                                              col.PREP_ORAL_WILLING, col.PREP_CAB_WILLING,
                                              col.PREP_LEN_WILLING, col.PREP_VR_WILLING],
                                             self.calc_willing_start_prep, sub_pop=starting_prep_pop)

            pop.set_present_variable(col.PREP_TYPE, prep_types, starting_prep_pop)
            pop.set_present_variable(col.EVER_PREP, True, starting_prep_pop)
            pop.set_present_variable(col.LAST_PREP_START_DATE, pop.date, starting_prep_pop)

            def set_prep_start_date(pop: Population, starting_prep_pop, prep_type, start_date_col):
                """
                Set a specific start date column for the population starting a corresponding PrEP type.
                """
                pop.set_present_variable(start_date_col, pop.date,
                                         pop.get_sub_pop_intersection(
                                             starting_prep_pop,
                                             pop.get_sub_pop(COND(col.PREP_TYPE, op.eq, prep_type))))

            set_prep_start_date(pop, starting_prep_pop, PrEPType.Oral, col.FIRST_ORAL_START_DATE)
            set_prep_start_date(pop, starting_prep_pop, PrEPType.Cabotegravir, col.FIRST_CAB_START_DATE)
            set_prep_start_date(pop, starting_prep_pop, PrEPType.Lenacapavir, col.FIRST_LEN_START_DATE)
            set_prep_start_date(pop, starting_prep_pop, PrEPType.VaginalRing, col.FIRST_VR_START_DATE)

    def calc_willing_start_prep(self, oral_pref, cab_pref, len_pref, vr_pref,
                                oral_willing, cab_willing, len_willing, vr_willing, size):
        """
        Returns PrEP types for people starting PrEP for the first time without explicitly
        testing to start PrEP. Individual preferences and availability are taken into account.
        """
        # group pref ranks and willingness
        prefs = [oral_pref, cab_pref, len_pref, vr_pref]
        willing = [oral_willing, cab_willing, len_willing, vr_willing]
        # zip prep type and willingness together and sort by pref rank
        sorted_zipped = sorted(enumerate(willing), key=lambda x: prefs[x[0]])
        sorted_dict = dict(sorted_zipped)

        starting_prep = None
        # find prep type someone is willing to take with the highest pref that is currently available
        for prep_type in sorted_dict:
            willing = sorted_dict[prep_type]
            if self.date >= self.date_prep_intro[prep_type] and willing:
                starting_prep = prep_type
                break

        # outcomes
        r = rng.uniform(size=size)
        # FIXME: may need different probabilities for different prep types
        starting = r < self.prob_base_prep_start
        prep = [starting_prep if s else None for s in starting]

        return prep

    def prep_usage(self, pop: Population):
        """
        Update PrEP usage for people starting, restarting, and stopping PrEP.
        """
        # starting prep for the first time
        eligible = pop.get_sub_pop([(col.HARD_REACH, op.eq, False),
                                    (col.HIV_DIAGNOSED, op.eq, False),
                                    (col.PREP_ELIGIBLE, op.eq, True),
                                    (col.PREP_ANY_WILLING, op.eq, True),
                                    (col.EVER_PREP, op.eq, False),
                                    (col.LAST_TEST_DATE, op.eq, pop.date)])
        # factor in both true and false negatives in hiv status
        starting_prep_pop = pop.get_sub_pop_intersection(
            eligible, pop.get_sub_pop_union(
                pop.get_sub_pop(COND(col.HIV_STATUS, op.eq, False)), self.get_presumed_hiv_neg_pop(pop)))

        # starting oral prep after testing
        self.tested_start_prep(
            pop, starting_prep_pop, PrEPType.Oral, col.PREP_ORAL_TESTED, col.FIRST_ORAL_START_DATE)
        # starting injectable cab prep after testing
        self.tested_start_prep(
            pop, starting_prep_pop, PrEPType.Cabotegravir, col.PREP_CAB_TESTED, col.FIRST_CAB_START_DATE)
        # starting injectable len prep after testing
        self.tested_start_prep(
            pop, starting_prep_pop, PrEPType.Lenacapavir, col.PREP_LEN_TESTED, col.FIRST_LEN_START_DATE)
        # starting vr prep after testing
        self.tested_start_prep(
            pop, starting_prep_pop, PrEPType.VaginalRing, col.PREP_VR_TESTED, col.FIRST_VR_START_DATE)

        # not tested explicitly to start prep
        self.general_start_prep(pop, starting_prep_pop)
