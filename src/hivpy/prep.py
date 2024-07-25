from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .population import Population

from enum import IntEnum

import hivpy.column_names as col

from .common import date, rng


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
