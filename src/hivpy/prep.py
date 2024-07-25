from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .population import Population

import operator as op
from enum import IntEnum

import hivpy.column_names as col

from .common import AND, COND, OR, date, rng


class PrEPType(IntEnum):
    Oral = 0
    Cabotegravir = 1  # injectable
    Lenacapavir = 2  # injectable
    VaginalRing = 3


class PrEPModule:

    def __init__(self, **kwargs):
        # FIXME: move these to data file
        self.prep_strategy = rng.choice([4, 8, 14])
        self.date_prep_intro = [date(2018, 3), date(2027), date(3000), date(2100)]
        self.prob_risk_informed_prep = 0.05
        self.prob_greater_risk_informed_prep = 0.1
        self.prob_suspect_risk_prep = 0.5

        self.rate_test_onprep_any = 1
        self.prep_willing_threshold = 0.2
        self.prob_test_prep_start = rng.choice([0.25, 0.50, 0.75])
        self.prob_prep_restart = rng.choice([0.05, 0.10, 0.20])

    def init_prep_variables(self, pop: Population):
        pop.init_variable(col.PREP_ELIGIBLE, False)
        pop.init_variable(col.PREP_TYPE, None)
        pop.init_variable(col.PREP_JUST_STARTED, False)
        pop.init_variable(col.LTP_HIV_STATUS, False)
        pop.init_variable(col.LTP_HIV_DIAGNOSED, False)
        pop.init_variable(col.LTP_ON_ART, False)

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
        # FIXME: is it correct to include the LTP_HIV_STATUS == False condition here
        # to avoid potential double-dipping with get_suspect_risk_pop?
        risk_informed_pop = pop.get_sub_pop(AND(COND(col.LONG_TERM_PARTNER, op.eq, True),
                                                COND(col.LTP_ON_ART, op.eq, False),
                                                COND(col.LTP_HIV_STATUS, op.eq, False)))
        r = rng.uniform(size=len(risk_informed_pop))
        rip_mask = r < prob_risk_informed_prep
        return pop.apply_bool_mask(rip_mask, risk_informed_pop)

    def get_suspect_risk_pop(self, pop: Population):
        """
        Return the sub-population that has a long-term partner who is not on ART but is infected
        and pass the higher probability to fulfill the criteria for risk-informed PrEP.
        """
        suspect_risk_pop = pop.get_sub_pop(AND(COND(col.LONG_TERM_PARTNER, op.eq, True),
                                               COND(col.LTP_ON_ART, op.eq, False),
                                               COND(col.LTP_HIV_STATUS, op.eq, True)))
        r = rng.uniform(size=len(suspect_risk_pop))
        srp_mask = r < self.prob_suspect_risk_prep
        return pop.apply_bool_mask(srp_mask, suspect_risk_pop)
