from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .population import Population

import operator as op

import hivpy.column_names as col

from .common import SexType, rng

class PrepModule:
    def __init__(self, **kwargs):
        self.prep_uptake_rate = 0.05
        self.prep_post_test_factor = 5
        return

    def init_prep_variables(self, pop: Population):
        pop.init_variable(col.ON_PREP, False)
        pop.init_variable(col.RECENTLY_TESTED, False)

    def update_prep_status(self, pop: Population):
        def new_prep_uptake(recently_tested, size):
            prob_prep = self.prep_uptake_rate
            if recently_tested:
                prob_prep *= self.prep_post_test_factor
            new_users = rng.uniform(size=size) < prob_prep
            return new_users

        pop.set_present_variable(col.RECENTLY_TESTED, pop.get_variable(col.LAST_TEST_DATE) == pop.date)

        prep_eligible = pop.get_sub_pop([(col.AGE, op.ge, 15),
                                             (col.AGE, op.lt, 65),
                                             (col.ON_PREP, op.eq, False)])

        pop.set_variable_by_group(col.ON_PREP,
                                  [col.RECENTLY_TESTED],
                                  new_prep_uptake,
                                  sub_pop=prep_eligible)
