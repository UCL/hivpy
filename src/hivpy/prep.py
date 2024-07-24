from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .population import Population

from enum import IntEnum

import hivpy.column_names as col


class PrEPType(IntEnum):
    Oral = 0
    Cabotegravir = 1  # injectable
    Lenacapavir = 2  # injectable
    VaginalRing = 3


class PrEPModule:

    def __init__(self, **kwargs):
        # FIXME: move these to data file
        self.rate_test_onprep_any = 1
        self.prep_willing_threshold = 0.2

    def init_prep_variables(self, pop: Population):
        pop.init_variable(col.PREP_TYPE, None)
        pop.init_variable(col.PREP_JUST_STARTED, False)
