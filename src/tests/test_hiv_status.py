import operator
from datetime import date

import pandas as pd

from hivpy.hiv_status import HIVStatusModule
from hivpy.population import Population
from hivpy.sexual_behaviour import selector


def test_initial_hiv():
    """Check no one under 15 or over 65 has HIV initially, but some people do"""
    pop = Population(size=1000, start_date=date(1989, 1, 1)).data
    HIV_module = HIVStatusModule()
    pop["HIV_status"] = HIV_module.initial_HIV_status(pop)
    index_young = selector(pop, HIV_status=(operator.eq, True), age=(operator.le, 15))
    assert not any(index_young)
    index_old = selector(pop, HIV_status=(operator.eq, True), age=(operator.ge, 65))
    assert not any(index_old)
    index_pos = selector(pop, HIV_status=(operator.eq, True))
    assert any(index_pos)


def test_hiv_update():
    pd.set_option('display.max_columns', None)
    pop_size = 100000
    pop = Population(size=pop_size, start_date=date(1989, 1, 1))
    data = pop.data
    prev_status = data["HIV_status"].copy()
    for i in range(10):
        pop.hiv_status.update_HIV_status(pop.data)

    new_cases = data["HIV_status"] & (~ prev_status)
    print(sum(new_cases))
    miracles = (~ data["HIV_status"]) & (prev_status)
    under_15s_idx = selector(data, HIV_status=(operator.eq, True), age=(operator.le, 15))
    assert not any(miracles)
    assert any(new_cases)
    assert not any(under_15s_idx)
