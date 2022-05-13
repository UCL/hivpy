import operator
from datetime import date, timedelta
from random import randint

import numpy as np
import pandas as pd
import yaml

from hivpy.common import SexType
from hivpy.population import Population
from hivpy.sexual_behaviour import (SexBehaviours, SexualBehaviourModule,
                                    selector)


def check_prob_sums(sex, trans_matrix):
    SBM = SexualBehaviourModule()
    SBM.sex_behaviour_trans[sex] = trans_matrix
    (dim, _) = trans_matrix.shape
    for i in range(0, dim):
        ages = np.array([15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65])
        sexes = np.array([sex]*len(ages))
        pop = pd.DataFrame({"age": ages, "sex": sexes, "HIV_status": False,
                           "HIV_Diagnosis_Date": None})
        SBM.init_risk_factors(pop)
        assert len(pop["rred"]) == 11
        assert (0 < SBM.new_partner_factor <= 2)
        tot_prob = np.array([0.0]*len(ages))  # probability for each age range
        for j in range(0, dim):
            tot_prob += SBM.prob_transition(sex, pop["rred"], i, j)
        assert(np.allclose(tot_prob, 1.0))


def test_transition_probabilities():
    with open("data/sex_behaviour.yaml", 'r') as file:
        yaml_data = yaml.safe_load(file)
    for trans_matrix in np.array(yaml_data["sex_behaviour_transition_options"]["Male"]):
        assert (trans_matrix.shape == (4, 4))
        check_prob_sums(SexType.Male, trans_matrix)
    for trans_matrix in np.array(yaml_data["sex_behaviour_transition_options"]["Female"]):
        assert (trans_matrix.shape == (2, 2))
        check_prob_sums(SexType.Female, trans_matrix)


def check_num_partners(row):
    sex = row["sex"]
    group = row["sex_behaviour"]
    n = row["num_partners"]
    age = row["age"]
    if age <= 15:  # no sexual partners for under 16s
        return n == 0
    if sex == SexType.Male:
        if group == 0:
            return n == 0
        elif group == 1:
            return (n > 0) & (n <= 3)
        elif group == 2:
            return (n > 3) & (n <= 9)
        else:
            return n in [10, 15, 20, 25, 30, 35]
    else:
        if group == 0:
            return n == 0
        else:
            return n in range(1, 10)


def test_num_partners():
    """Check that number of partners are reasonable"""
    pop_data = Population(size=1000, start_date=date(1989, 1, 1)).data
    assert(any(pop_data["num_partners"] > 0))
    # Check the num_partners column
    checks = pop_data.apply(check_num_partners, axis=1)
    assert np.all(checks)


def test_behaviour_updates():
    """Check that at least one person changes sexual behaviour groups"""
    pop = Population(size=1000, start_date=date(1989, 1, 1))
    initial_groupings = pop.data["sex_behaviour"].copy()
    for i in range(1):
        pop.sexual_behaviour.update_sex_groups(pop.data)
    subsequent_groupings = pop.data["sex_behaviour"]
    assert(any(initial_groupings != subsequent_groupings))


def test_initial_sex_behaviour_groups():
    """Check that the number of people initialised into each sexual behaviour group
    is within 3-sigma of the expectation value, as calculated by a binomial distribution."""
    N = 10000
    pop_data = Population(size=N, start_date=date(1989, 1, 1)).data
    with open("data/sex_behaviour.yaml", 'r') as file:
        yaml_data = yaml.safe_load(file)
    probs = {SexType.Male:
             yaml_data["initial_sex_behaviour_probabilities"]["Male"]["Probability"],
             SexType.Female:
             yaml_data["initial_sex_behaviour_probabilities"]["Female"]["Probability"]}
    for sex in SexType:
        index_sex = selector(pop_data, sex=(operator.eq, sex))
        n_sex = sum(index_sex)
        Prob_sex = np.array(probs[sex])
        Prob_sex /= sum(Prob_sex)
        for g in SexBehaviours[sex]:
            index_group = selector(pop_data, sex_behaviour=(operator.eq, g))
            p = Prob_sex[g]
            sigma = np.sqrt(p*(1-p)*float(n_sex))
            Expectation = float(n_sex)*p
            N_g = sum(index_sex & index_group)
            assert Expectation - sigma*3 <= N_g <= Expectation + sigma*3


def test_rred_long_term_partner():
    N = 1000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    # pick some indices and give those people LTPs
    indices = [randint(0, N-1) for i in range(15)]
    pop.data["partnered"] = False
    pop.data.loc[indices, "partnered"] = True
    SBM = SexualBehaviourModule()
    SBM.calc_rred_long_term_partnered(pop.data)
    for i in indices:
        assert pop.data.loc[i, "rred_long_term_partnered"] == SBM.ltp_risk_factor


def test_rred_adc():
    N = 20
    pop = Population(size=N, start_date=date(1989, 1, 1))
    pop.data["HIV_status"] = False  # make test independent of how HIV is initialised
    init_HIV_idx = [randint(0, N-1) for i in range(5)]
    pop.data.loc[init_HIV_idx, "HIV_status"] = True
    expected_rred = np.ones(N)
    expected_rred[init_HIV_idx] = 0.2
    SBM = SexualBehaviourModule()
    # initialise rred_adc
    SBM.init_risk_factors(pop.data)
    # Check only people with HIV have rred_adc = 0.2
    assert np.all(pop.data["rred_adc"] == expected_rred)
    # Assign more people with HIV
    add_HIV_idx = [randint(0, N-1) for i in range(5)]
    pop.data.loc[add_HIV_idx, "HIV_status"] = True
    # Update rred factors
    SBM.update_sex_behaviour(pop)
    expected_rred[add_HIV_idx] = 0.2
    assert np.all(pop.data["rred_adc"] == expected_rred)


def test_rred_diagnosis():
    N = 20
    pop = Population(size=N, start_date=date(1989, 1, 1))
    pop.data["HIV_status"] = False  # init everyone HIV negative
    SBM = SexualBehaviourModule()
    SBM.rred_diagnosis = 4
    SBM.rred_diagnosis_period = timedelta(700)
    SBM.update_rred_diagnosis(pop.data, pop.date)
    assert np.all(pop.data["rred_diagnosis"] == 1)
    # give up some people HIV and advance the date
    HIV_idx = [randint(0, N-1) for i in range(5)]
    pop.data.loc[HIV_idx, "HIV_status"] = True
    pop.data.loc[HIV_idx, "HIV_Diagnosis_Date"] = pop.date
    SBM.update_rred_diagnosis(pop.data, pop.date)
    for i in HIV_idx:
        assert pop.data.loc[i, "rred_diagnosis"] == 4
    pop.date += timedelta(days=365)
    SBM.update_rred_diagnosis(pop.data, pop.date)
    for i in HIV_idx:
        assert pop.data.loc[i, "rred_diagnosis"] == 4
    pop.date += timedelta(days=500)
    SBM.update_rred_diagnosis(pop.data, pop.date)
    for i in HIV_idx:
        assert pop.data.loc[i, "rred_diagnosis"] == 2
