import operator
from datetime import date

import pandas as pd
import numpy as np
from hivpy.common import SexType

from hivpy.hiv_status import HIVStatusModule
from hivpy.population import Population
from hivpy.sexual_behaviour import selector
import hivpy.column_names as col


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
        pop.hiv_status.update_HIV_status(pop)

    new_cases = data["HIV_status"] & (~ prev_status)
    miracles = (~ data["HIV_status"]) & (prev_status)
    under_15s_idx = selector(data, HIV_status=(operator.eq, True), age=(operator.le, 15))
    assert not any(miracles)
    assert any(new_cases)
    assert not any(under_15s_idx)

def test_hiv_update2():
    N = 100000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    hiv_module = pop.hiv_status
    # create some easy to calculate scenarios 
    # let everyone have one partner so P(HIV) is just the fraction with HIV 
    pop.data[col.SEX] = np.array([SexType.Male, SexType.Female] * (N//2))
    pop.data[col.AGE] = 30
    pop.data[col.HIV_STATUS] = False
    pop.data[col.HIV_STATUS][:10000] = np.array([True] * 10000)  # 10% HIV+
    pop.data[col.NUM_PARTNERS] = 1
    pop.sexual_behaviour.assign_stp_ages(pop)
    pop.data[col.VIRAL_LOAD_GROUP] = 5  # set everyone to same viral load group
    prev_hiv_status = pop.data[col.HIV_STATUS].copy()
    hiv_module.update_HIV_status(pop)
    expected_transmission_prob = 0.1 * 0.1  # 10% probability of HIV+ and 10% probability of transmission given viral load
    pop.data["delta_HIV"] = pop.data[col.HIV_STATUS] ^ prev_hiv_status
    n_new_male = sum(pop.data.loc[pop.data[col.SEX]==SexType.Male, "delta_HIV"])  # count number of new transmissions for men 
    exp_new_male = 45000 * expected_transmission_prob
    sig = np.sqrt(45000 * expected_transmission_prob * (1 - expected_transmission_prob))
    print(n_new_male, exp_new_male, sig)