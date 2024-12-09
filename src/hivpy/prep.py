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
    Lenacapavir = 2   # injectable
    VaginalRing = 3
    NoPrep = 5


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
        self.cab_available = True
        self.prob_risk_informed_prep = self.p_data.prob_risk_informed_prep
        self.prob_greater_risk_informed_prep = self.p_data.prob_greater_risk_informed_prep
        self.prob_suspect_risk_prep = self.p_data.prob_suspect_risk_prep

        self.prep_oral_pref_beta = self.p_data.prep_oral_pref_beta.sample()
        self.prep_cab_pref_beta = self.prep_oral_pref_beta + 0.3
        self.prep_len_pref_beta = self.prep_cab_pref_beta
        self.prep_vr_pref_beta = self.prep_oral_pref_beta - 0.1
        self.vl_prevalence_affects_prep = rng.choice([True, False], p=[1/3, 2/3])
        self.vl_prevalence_prep_threshold = self.p_data.vl_prevalence_prep_threshold.sample()

        self.rate_test_onprep_any = self.p_data.rate_test_onprep_any
        self.prep_willing_threshold = self.p_data.prep_willing_threshold
        self.prob_test_prep_start = self.p_data.prob_test_prep_start.sample()
        # probability of starting prep in people who are eligible, willing,
        # and tested for HIV according to base rate of testing
        self.prob_base_prep_start = self.p_data.prob_base_prep_start.sample()
        # FIXME: add 4-year scale up for these probabilities
        self.prob_oral_prep_start = self.prob_base_prep_start
        self.prob_cab_prep_start = self.prob_base_prep_start
        self.prob_len_prep_start = self.prob_base_prep_start
        self.prob_vr_prep_start = self.prob_base_prep_start
        self.prob_prep_restart = self.p_data.prob_prep_restart.sample()
        # FIXME: stop probabilities dependent on time step length
        self.prob_oral_prep_stop = self.p_data.prob_base_prep_stop.sample()
        self.prob_cab_prep_stop = self.p_data.prob_base_prep_stop.sample()
        self.prob_len_prep_stop = self.prob_cab_prep_stop
        self.prob_vr_prep_stop = self.p_data.prob_base_prep_stop_nonuniform.sample()

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
        pop.init_variable(col.FAVOURED_PREP_TYPE, PrEPType.NoPrep)
        pop.init_variable(col.R_PREP, 1.0)
        pop.init_variable(col.PREP_ELIGIBLE, False)

        pop.init_variable(col.PREP_TYPE, PrEPType.NoPrep)
        pop.init_variable(col.PREP_ORAL_TESTED, False)
        pop.init_variable(col.PREP_CAB_TESTED, False)
        pop.init_variable(col.PREP_LEN_TESTED, False)
        pop.init_variable(col.PREP_VR_TESTED, False)

        pop.init_variable(col.EVER_PREP, False)
        pop.init_variable(col.FIRST_ORAL_START_DATE, None)
        pop.init_variable(col.FIRST_CAB_START_DATE, None)
        pop.init_variable(col.FIRST_LEN_START_DATE, None)
        pop.init_variable(col.FIRST_VR_START_DATE, None)
        pop.init_variable(col.LAST_PREP_START_DATE, None)
        pop.init_variable(col.LAST_PREP_STOP_DATE, None)
        pop.init_variable(col.PREP_JUST_STARTED, False)

        pop.init_variable(col.CONT_ON_PREP, timedelta(months=0))
        pop.init_variable(col.CONT_ACTIVE_ON_PREP, timedelta(months=0))
        pop.init_variable(col.CUMULATIVE_PREP_ORAL, timedelta(months=0))
        pop.init_variable(col.CUMULATIVE_PREP_CAB, timedelta(months=0))
        pop.init_variable(col.CUMULATIVE_PREP_LEN, timedelta(months=0))
        pop.init_variable(col.CUMULATIVE_PREP_VR, timedelta(months=0))

        pop.init_variable(col.LTP_ON_ART, False)

    # FIXME: should this function be in another module?
    def get_vl_prevalence(self, pop: Population):
        """
        Return the prevalence of people between 15 and 50 years old with a viral load of over 3.0.
        Affects willingness to take PrEP.
        """
        gen_pop = len(pop.get_sub_pop([(col.AGE, op.ge, 15), (col.AGE, op.lt, 50)]))
        # find prevalence of people with a viral load of over 3.0
        return (len(pop.get_sub_pop([(col.VIRAL_LOAD, op.ge, 3.0),
                                     (col.AGE, op.ge, 15),
                                     (col.AGE, op.lt, 50)])) / gen_pop
                if gen_pop > 0 else 0)

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
                                  AND(COND(col.LTP_DIAGNOSED, op.eq, True),
                                      COND(col.LTP_ON_ART, op.eq, False))))

    def get_risk_informed_pop(self, pop: Population, prob_risk_informed_prep):
        """
        Return the sub-population that has a long-term partner who is not on ART
        and pass the probability to fulfill the criteria for risk-informed PrEP.
        """
        return pop.get_sub_pop(AND(COND(col.LONG_TERM_PARTNER, op.eq, True),
                                   COND(col.LTP_ON_ART, op.eq, False),
                                   COND(col.LTP_STATUS, op.eq, False),
                                   COND(col.R_PREP, op.lt, prob_risk_informed_prep)))

    def get_suspect_risk_pop(self, pop: Population):
        """
        Return the sub-population that has a long-term partner who is not on ART but is infected
        and pass the higher probability to fulfill the criteria for risk-informed PrEP.
        """
        return pop.get_sub_pop(AND(COND(col.LONG_TERM_PARTNER, op.eq, True),
                                   COND(col.LTP_ON_ART, op.eq, False),
                                   COND(col.LTP_STATUS, op.eq, True),
                                   COND(col.R_PREP, op.lt, self.prob_suspect_risk_prep)))

    # FIXME: this function may be removed if there are no issues with
    # updating PrEP after HIV diagnosis during population evolution
    def get_presumed_hiv_neg_pop(self, pop: Population):
        """
        Return the sub-population that has been tested and is HIV positive but
        received a false negative result.
        """
        false_neg_pop = pop.get_sub_pop(AND(COND(col.EVER_TESTED, op.eq, True),
                                            COND(col.HIV_DIAGNOSED, op.eq, False),
                                            COND(col.HIV_STATUS, op.eq, True)))

        # general test sensitivity
        eff_test_sens = pop.hiv_diagnosis.test_sens_general
        if not pop.hiv_diagnosis.init_prep_inj_na:
            # infected up to 3 months ago
            recently_infected_pop = pop.get_sub_pop_intersection(
                pop.get_sub_pop(COND(col.DATE_HIV_INFECTION, op.ge, pop.date - timedelta(months=3))), false_neg_pop)

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

    def prep_preference(self, pop: Population):
        """
        Determine PrEP preferences for all PrEP types.
        """
        # oral prep pref
        self.set_prep_preference(pop, self.date_prep_intro[PrEPType.Oral],
                                 self.prep_oral_pref_beta, col.PREP_ORAL_PREF)
        # injectable prep pref
        self.set_prep_preference(pop, self.date_prep_intro[PrEPType.Cabotegravir],
                                 self.prep_cab_pref_beta, col.PREP_CAB_PREF)
        self.set_prep_preference(pop, self.date_prep_intro[PrEPType.Lenacapavir],
                                 self.prep_len_pref_beta, col.PREP_LEN_PREF)
        # vr prep pref (women only)
        self.set_prep_preference(pop, self.date_prep_intro[PrEPType.VaginalRing], self.prep_vr_pref_beta,
                                 col.PREP_VR_PREF, sub_pop_mod=pop.get_sub_pop([(col.SEX, op.eq, SexType.Female)]))

    def set_prep_preference(self, pop: Population, date_intro, pref_beta, pref_col, sub_pop_mod=None):
        """
        Set preference values for a specific type of PrEP.
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

    def prep_willingness(self, pop: Population):
        """
        Determine PrEP willingness for all PrEP types.
        """
        vl_prevalence = self.get_vl_prevalence(pop)
        # there's a chance nobody is willing to take PrEP if unsuppressed viral load prevalence is too low
        if self.vl_prevalence_affects_prep and vl_prevalence < self.vl_prevalence_prep_threshold:
            pop.set_present_variable(col.PREP_ORAL_WILLING, False)
            pop.set_present_variable(col.PREP_CAB_WILLING, False)
            pop.set_present_variable(col.PREP_LEN_WILLING, False)
            pop.set_present_variable(col.PREP_VR_WILLING, False)
            pop.set_present_variable(col.PREP_ANY_WILLING, False)
        # otherwise set willingness as normal
        else:
            self.set_prep_willingness(pop, col.PREP_ORAL_PREF, col.PREP_ORAL_WILLING)
            self.set_prep_willingness(pop, col.PREP_CAB_PREF, col.PREP_CAB_WILLING)
            self.set_prep_willingness(pop, col.PREP_LEN_PREF, col.PREP_LEN_WILLING)
            self.set_prep_willingness(pop, col.PREP_VR_PREF, col.PREP_VR_WILLING)

    def set_prep_willingness(self, pop: Population, pref_col, willing_col):
        """
        Set willingness values for a specific type of PrEP.
        """
        # determine willingness by comparing to threshold
        willingness = pop.get_variable(pref_col) > self.prep_willing_threshold
        pop.set_present_variable(willing_col, willingness)
        pop.set_present_variable(col.PREP_ANY_WILLING, True, pop.apply_bool_mask(willingness))

    def prep_pref_ranks(self, pop: Population, sub_pop=None):
        """
        Rank PrEP preferences.
        """
        # get ranking outcomes
        pref_ranks = pop.col_apply([col.PREP_ORAL_PREF, col.PREP_CAB_PREF,
                                    col.PREP_LEN_PREF, col.PREP_VR_PREF],
                                   self.calc_prep_pref_ranks, sub_pop=sub_pop)
        # set ranks for each prep type
        pop.set_present_variable(col.PREP_ORAL_RANK, [i[0] for i in pref_ranks], sub_pop)
        pop.set_present_variable(col.PREP_CAB_RANK, [i[1] for i in pref_ranks], sub_pop)
        pop.set_present_variable(col.PREP_LEN_RANK, [i[2] for i in pref_ranks], sub_pop)
        pop.set_present_variable(col.PREP_VR_RANK, [i[3] for i in pref_ranks], sub_pop)

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
        return ranks

    def favoured_prep(self, pop: Population, sub_pop=None):
        """
        Determine favoured PrEP type using preference ranks. Favoured PrEP is the type of PrEP an individual
        is willing to take with the highest preference value that is also currently available.
        """
        # FIXME: can we pass the date to transform_group in a better way?
        self.date = pop.date
        # find prep type with highest preference an individual is willing to take that is also currently available
        favoured_prep = pop.transform_group([col.PREP_ORAL_RANK, col.PREP_CAB_RANK,
                                             col.PREP_LEN_RANK, col.PREP_VR_RANK,
                                             col.PREP_ORAL_WILLING, col.PREP_CAB_WILLING,
                                             col.PREP_LEN_WILLING, col.PREP_VR_WILLING],
                                            self.calc_favoured_prep, sub_pop=sub_pop, use_size=False)
        pop.set_present_variable(col.FAVOURED_PREP_TYPE, favoured_prep, sub_pop)

    def calc_favoured_prep(self, oral_rank, cab_rank, len_rank, vr_rank,
                           oral_willing, cab_willing, len_willing, vr_willing):
        """
        Returns favoured PrEP type based on willingness, preference rank and availability.
        """
        # group pref ranks and willingness
        prefs = [oral_rank, cab_rank, len_rank, vr_rank]
        willing = [oral_willing, cab_willing, len_willing, vr_willing]
        # zip prep type and willingness together and sort by pref rank
        sorted_zipped = sorted(enumerate(willing), key=lambda x: prefs[x[0]])
        sorted_dict = dict(sorted_zipped)

        favoured_prep = PrEPType.NoPrep
        # find prep type someone is willing to take with the highest pref that is currently available
        for prep_type in sorted_dict:
            willing = sorted_dict[prep_type]
            if self.date >= self.date_prep_intro[prep_type] and willing:
                if PrEPType(prep_type) is not PrEPType.Cabotegravir or self.cab_available:
                    favoured_prep = prep_type
                    break

        return favoured_prep

    def prep_propensity(self, pop: Population):
        """
        Determine PrEP preference values, willingness to take PrEP, PrEP preference ranks, and favoured PrEP type.
        """
        # store initial preference values
        init_prefs = pop.data[[col.PREP_ORAL_PREF, col.PREP_CAB_PREF, col.PREP_LEN_PREF, col.PREP_VR_PREF]]
        # set preference values
        self.prep_preference(pop)
        # set willingness values
        self.prep_willingness(pop)
        # get new preference values
        new_prefs = pop.data[[col.PREP_ORAL_PREF, col.PREP_CAB_PREF, col.PREP_LEN_PREF, col.PREP_VR_PREF]]
        # find people whose preference has changed this time step
        changed_pref_pop = new_prefs.compare(init_prefs).index
        if len(changed_pref_pop) > 0:
            # update preference ranks
            self.prep_pref_ranks(pop, changed_pref_pop)
        # update favoured prep
        self.favoured_prep(pop, changed_pref_pop)

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
                                                            AND(COND(col.LTP_DIAGNOSED, op.eq, True),
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
                                                  COND(col.LTP_DIAGNOSED, op.eq, False),
                                                  COND(col.AGE, op.ge, 15),
                                                  COND(col.AGE, op.lt, 50),
                                                  OR(COND(col.R_PREP, op.lt, 0.01),  # (alt) risk informed
                                                     AND(COND(col.R_PREP, op.lt, self.prob_suspect_risk_prep),
                                                         COND(col.LTP_STATUS, op.eq, True)))))  # (alt) suspect risk
                at_risk_ltp_pop = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                                      COND(col.LTP_DIAGNOSED, op.eq, True),
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

    def tested_start_prep(self, pop: Population, prep_eligible_pop, prep_type,
                          prep_tested_col, first_start_col, time_step):
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
                pop.set_present_variable(col.PREP_JUST_STARTED, True, starting_prep_pop)
                # set start dates
                pop.set_present_variable(col.LAST_PREP_START_DATE, pop.date, starting_prep_pop)
                pop.set_present_variable(first_start_col, pop.date, starting_prep_pop)
                # set continuous use
                pop.set_present_variable(col.CONT_ON_PREP, time_step, starting_prep_pop)
                pop.set_present_variable(col.CONT_ACTIVE_ON_PREP, time_step, starting_prep_pop)
                # increment cumulative use
                self.set_all_prep_cumulative(pop, starting_prep_pop, time_step)

    def general_start_prep(self, pop: Population, prep_eligible_pop, time_step):
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
            # starting prep outcomes
            prep_types = pop.transform_group([col.FAVOURED_PREP_TYPE], self.calc_starting_prep,
                                             sub_pop=starting_prep_pop)
            pop.set_present_variable(col.PREP_TYPE, prep_types, starting_prep_pop)
            pop.set_present_variable(col.EVER_PREP, True, starting_prep_pop)
            pop.set_present_variable(col.PREP_JUST_STARTED, True, starting_prep_pop)
            # set start dates
            self.set_all_prep_start_dates(pop, starting_prep_pop)
            # set continuous use
            pop.set_present_variable(col.CONT_ON_PREP, time_step, starting_prep_pop)
            pop.set_present_variable(col.CONT_ACTIVE_ON_PREP, time_step, starting_prep_pop)
            # increment cumulative use
            self.set_all_prep_cumulative(pop, starting_prep_pop, time_step)

    def set_all_prep_start_dates(self, pop: Population, starting_prep_pop):
        """
        Set current PrEP start date and all first PrEP start date columns.
        """
        pop.set_present_variable(col.LAST_PREP_START_DATE, pop.date, starting_prep_pop)
        self.set_prep_first_start_date(pop, starting_prep_pop, PrEPType.Oral, col.FIRST_ORAL_START_DATE)
        self.set_prep_first_start_date(pop, starting_prep_pop, PrEPType.Cabotegravir, col.FIRST_CAB_START_DATE)
        self.set_prep_first_start_date(pop, starting_prep_pop, PrEPType.Lenacapavir, col.FIRST_LEN_START_DATE)
        self.set_prep_first_start_date(pop, starting_prep_pop, PrEPType.VaginalRing, col.FIRST_VR_START_DATE)

    def set_prep_first_start_date(self, pop: Population, starting_prep_pop, prep_type, first_start_date_col):
        """
        Set a specific start date column for the population starting a corresponding
        PrEP type for the first time. Only overwrites if first start date is None.
        """
        pop.set_present_variable(
            first_start_date_col, pop.date,
            pop.get_sub_pop_intersection(
                starting_prep_pop, pop.get_sub_pop(AND(COND(col.PREP_TYPE, op.eq, prep_type),
                                                       COND(first_start_date_col, op.eq, None)))))

    def set_all_prep_cumulative(self, pop: Population, using_prep_pop, time_step):
        """
        Increment all cumulative PrEP usage columns for active PrEP users.
        """
        self.set_prep_cumulative(pop, using_prep_pop, PrEPType.Oral, col.CUMULATIVE_PREP_ORAL, time_step)
        self.set_prep_cumulative(pop, using_prep_pop, PrEPType.Cabotegravir, col.CUMULATIVE_PREP_CAB, time_step)
        self.set_prep_cumulative(pop, using_prep_pop, PrEPType.Lenacapavir, col.CUMULATIVE_PREP_LEN, time_step)
        self.set_prep_cumulative(pop, using_prep_pop, PrEPType.VaginalRing, col.CUMULATIVE_PREP_VR, time_step)

    def set_prep_cumulative(self, pop: Population, using_prep_pop, prep_type, cumulative_col, time_step):
        """
        Increment a specific cumulative PrEP usage column for the population using a
        corresponding PrEP type. Applies to those starting, continuing, or switching PrEP.
        """
        prep_cont = pop.get_variable(cumulative_col) + time_step
        pop.set_present_variable(
            cumulative_col, prep_cont,
            pop.get_sub_pop_intersection(
                using_prep_pop, pop.get_sub_pop(COND(col.PREP_TYPE, op.eq, prep_type))))

    def calc_starting_prep(self, favoured_prep, size):
        """
        Returns PrEP types for people starting PrEP for the first time without explicitly
        testing to start PrEP. Individual preferences and availability are taken into account.
        """
        # outcomes
        r = rng.uniform(size=size)
        if PrEPType(favoured_prep) is PrEPType.Oral:
            starting = r < self.prob_oral_prep_start
        elif PrEPType(favoured_prep) is PrEPType.Cabotegravir:
            starting = r < self.prob_cab_prep_start
        elif PrEPType(favoured_prep) is PrEPType.Lenacapavir:
            starting = r < self.prob_len_prep_start
        elif PrEPType(favoured_prep) is PrEPType.VaginalRing:
            starting = r < self.prob_vr_prep_start
        else:
            starting = [False] * size
        prep = [favoured_prep if s else PrEPType.NoPrep for s in starting]

        return prep

    def start_prep(self, pop: Population, time_step):
        """
        Update PrEP usage for people starting PrEP for the first time.
        """
        # clear just_started flag
        pop.set_present_variable(col.PREP_JUST_STARTED, False,
                                 pop.get_sub_pop([(col.LAST_PREP_START_DATE, op.ne, pop.date)]))
        # find people eligible to start for the first time
        eligible = pop.get_sub_pop([(col.HARD_REACH, op.eq, False),
                                    (col.HIV_DIAGNOSED, op.eq, False),
                                    (col.PREP_ELIGIBLE, op.eq, True),
                                    (col.PREP_ANY_WILLING, op.eq, True),
                                    (col.EVER_PREP, op.eq, False),
                                    (col.LAST_TEST_DATE, op.eq, pop.date)])

        # starting oral prep after testing
        self.tested_start_prep(
            pop, eligible, PrEPType.Oral, col.PREP_ORAL_TESTED, col.FIRST_ORAL_START_DATE, time_step)
        # starting injectable cab prep after testing
        self.tested_start_prep(
            pop, eligible, PrEPType.Cabotegravir, col.PREP_CAB_TESTED, col.FIRST_CAB_START_DATE, time_step)
        # starting injectable len prep after testing
        self.tested_start_prep(
            pop, eligible, PrEPType.Lenacapavir, col.PREP_LEN_TESTED, col.FIRST_LEN_START_DATE, time_step)
        # starting vr prep after testing
        self.tested_start_prep(
            pop, eligible, PrEPType.VaginalRing, col.PREP_VR_TESTED, col.FIRST_VR_START_DATE, time_step)
        # not tested explicitly to start prep
        self.general_start_prep(pop, eligible, time_step)

    def continue_prep(self, pop: Population, time_step):
        """
        Update PrEP usage for people continuing PrEP.
        """
        # people who have used prep before but not yet started this time step
        eligible = pop.get_sub_pop(AND(COND(col.PREP_ELIGIBLE, op.eq, True),
                                       COND(col.EVER_PREP, op.eq, True),
                                       COND(col.PREP_JUST_STARTED, op.eq, False),
                                       COND(col.LAST_PREP_STOP_DATE, op.eq, None),
                                       OR(COND(col.LAST_TEST_DATE, op.ne, pop.date),
                                          AND(COND(col.LAST_TEST_DATE, op.eq, pop.date),
                                              COND(col.HIV_DIAGNOSED, op.eq, False)))))

        if len(eligible) > 0:
            # continuous prep outcomes
            prep_types = pop.transform_group([col.PREP_TYPE, col.FAVOURED_PREP_TYPE],
                                             self.calc_current_prep, sub_pop=eligible)
            # find various sub-populations
            # people who are continuing current prep
            continuing_prep_mask = pop.get_variable(col.PREP_TYPE, eligible) == prep_types
            continuing_prep_pop = pop.apply_bool_mask(continuing_prep_mask, eligible)
            # people who are switching prep
            switching_prep_mask = (pop.get_variable(col.PREP_TYPE, eligible) != prep_types) & (prep_types != PrEPType.NoPrep)
            switching_prep_pop = pop.apply_bool_mask(switching_prep_mask, eligible)
            # people who are either continuing or switching prep
            using_prep_pop = pop.apply_bool_mask(prep_types != PrEPType.NoPrep, eligible)
            # people who are stopping prep
            stopping_prep_pop = pop.apply_bool_mask(prep_types == PrEPType.NoPrep, eligible)

            if len(continuing_prep_pop) > 0:
                prep_cont = pop.get_variable(col.CONT_ON_PREP, eligible) + time_step
                prep_active_cont = pop.get_variable(col.CONT_ACTIVE_ON_PREP, eligible) + time_step
                # increment continuous use
                pop.set_present_variable(col.CONT_ON_PREP, prep_cont, continuing_prep_pop)
                pop.set_present_variable(col.CONT_ACTIVE_ON_PREP, prep_active_cont, continuing_prep_pop)

            if len(switching_prep_pop) > 0:
                # set new prep types
                pop.set_present_variable(col.PREP_TYPE, prep_types, switching_prep_pop)
                # set start dates
                self.set_all_prep_start_dates(pop, switching_prep_pop)
                # reset continuous use
                pop.set_present_variable(col.CONT_ON_PREP, time_step, switching_prep_pop)
                pop.set_present_variable(col.CONT_ACTIVE_ON_PREP, time_step, switching_prep_pop)

            if len(using_prep_pop) > 0:
                # increment cumulative use
                self.set_all_prep_cumulative(pop, using_prep_pop, time_step)

            if len(stopping_prep_pop) > 0:
                # stop continuous use
                pop.set_present_variable(col.CONT_ON_PREP, timedelta(months=0), stopping_prep_pop)
                pop.set_present_variable(col.CONT_ACTIVE_ON_PREP, timedelta(months=0), stopping_prep_pop)
                # set stop date
                pop.set_present_variable(col.LAST_PREP_STOP_DATE, pop.date, stopping_prep_pop)

    def calc_current_prep(self, prep_type, favoured_prep, size):
        """
        Returns PrEP types for people continuing PrEP.
        Individual preferences and availability are taken into account.
        """
        # outcomes
        r = rng.uniform(size=size)
        if PrEPType(prep_type) is PrEPType.Oral:
            continuing = r < (1 - self.prob_oral_prep_stop)
        elif PrEPType(prep_type) is PrEPType.Cabotegravir:
            continuing = r < (1 - self.prob_cab_prep_stop)
        elif PrEPType(prep_type) is PrEPType.Lenacapavir:
            continuing = r < (1 - self.prob_len_prep_stop)
        elif PrEPType(prep_type) is PrEPType.VaginalRing:
            continuing = r < (1 - self.prob_vr_prep_stop)
        else:
            continuing = [False] * size
        prep = [favoured_prep if c else PrEPType.NoPrep for c in continuing]

        return prep

    def restart_prep(self, pop: Population, time_step):
        """
        Update PrEP usage for people restarting PrEP.
        """
        # people who have used prep before and previously stopped using it
        eligible = pop.get_sub_pop(AND(COND(col.HIV_DIAGNOSED, op.eq, False),
                                       COND(col.PREP_ELIGIBLE, op.eq, True),
                                       COND(col.EVER_PREP, op.eq, True),
                                       COND(col.LAST_PREP_STOP_DATE, op.lt, pop.date),
                                       COND(col.LAST_TEST_DATE, op.eq, pop.date)))

        if len(eligible) > 0:
            # starting prep outcomes
            prep_types = pop.transform_group([col.FAVOURED_PREP_TYPE], self.calc_restarting_prep,
                                             sub_pop=eligible)
            # people who are restarting prep
            restarting_prep_pop = pop.apply_bool_mask(prep_types != PrEPType.NoPrep, eligible)

            if len(restarting_prep_pop) > 0:
                # set prep types
                pop.set_present_variable(col.PREP_TYPE, prep_types, restarting_prep_pop)
                pop.set_present_variable(col.PREP_JUST_STARTED, True, restarting_prep_pop)
                # set start dates
                self.set_all_prep_start_dates(pop, restarting_prep_pop)
                # set continuous use
                pop.set_present_variable(col.CONT_ON_PREP, time_step, restarting_prep_pop)
                pop.set_present_variable(col.CONT_ACTIVE_ON_PREP, time_step, restarting_prep_pop)
                # increment cumulative use
                self.set_all_prep_cumulative(pop, restarting_prep_pop, time_step)
                # unset stop date
                pop.set_present_variable(col.LAST_PREP_STOP_DATE, None, restarting_prep_pop)

    def calc_restarting_prep(self, favoured_prep, size):
        """
        Returns PrEP types for people restarting PrEP.
        Individual preferences and availability are taken into account.
        """
        # outcomes
        r = rng.uniform(size=size)
        restarting = r < self.prob_prep_restart
        prep = [favoured_prep if r else PrEPType.NoPrep for r in restarting]

        return prep

    def prep_usage(self, pop: Population, time_step):
        """
        Update PrEP usage for people starting, continuing, switching, restarting, and stopping PrEP.
        """
        # starting prep for the first time
        self.start_prep(pop, time_step)
        # continuing prep
        self.continue_prep(pop, time_step)
        # restarting prep
        self.restart_prep(pop, time_step)
