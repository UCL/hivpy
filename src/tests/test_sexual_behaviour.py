import importlib.resources
import operator
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
import yaml

from hivpy.common import SexType, rng
from hivpy.population import Population
from hivpy.sexual_behaviour import (SexBehaviours, SexualBehaviourModule,
                                    selector)


@pytest.fixture(scope="module")
def yaml_data():
    with importlib.resources.open_text("hivpy.data", "sex_behaviour.yaml") as file:
        return yaml.safe_load(file)


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


def test_transition_probabilities(yaml_data):
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


def test_initial_sex_behaviour_groups(yaml_data):
    """Check that the number of people initialised into each sexual behaviour group
    is within 3-sigma of the expectation value, as calculated by a binomial distribution."""
    N = 10000
    pop_data = Population(size=N, start_date=date(1989, 1, 1)).data
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
    indices = rng.integers(0, N, size=15)
    pop.data["partnered"] = False
    pop.data.loc[indices, "partnered"] = True
    SBM = SexualBehaviourModule()
    SBM.update_rred_long_term_partnered(pop.data)
    assert all(pop.data.loc[indices, "rred_long_term_partnered"] == SBM.ltp_risk_factor)


def test_rred_adc():
    N = 20
    pop = Population(size=N, start_date=date(1989, 1, 1))
    pop.data["HIV_status"] = False  # make test independent of how HIV is initialised
    init_HIV_idx = rng.integers(0, N, size=5)
    pop.data.loc[init_HIV_idx, "HIV_status"] = True
    expected_rred = np.ones(N)
    expected_rred[init_HIV_idx] = 0.2
    SBM = SexualBehaviourModule()
    # initialise rred_adc
    SBM.init_risk_factors(pop.data)
    # Check only people with HIV have rred_adc = 0.2
    assert np.all(pop.data["rred_adc"] == expected_rred)
    # Assign more people with HIV
    add_HIV_idx = rng.integers(0, N, size=5)
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
    HIV_idx = rng.integers(0, N, size=5)
    pop.data.loc[HIV_idx, "HIV_status"] = True
    pop.data.loc[HIV_idx, "HIV_Diagnosis_Date"] = pop.date
    SBM.update_rred_diagnosis(pop.data, pop.date)
    assert all(pop.data.loc[HIV_idx, "rred_diagnosis"] == 4)
    pop.date += timedelta(days=365)
    SBM.update_rred_diagnosis(pop.data, pop.date)
    assert all(pop.data.loc[HIV_idx, "rred_diagnosis"] == 4)
    pop.date += timedelta(days=500)
    SBM.update_rred_diagnosis(pop.data, pop.date)
    assert all(pop.data.loc[HIV_idx, "rred_diagnosis"] == 2)
    undiagnosed = [x for x in range(0, N) if x not in HIV_idx]
    assert all(pop.data.loc[undiagnosed, "rred_diagnosis"] == 1)


def test_rred_balance():
    N = 1000
    n_partners = 1000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    SBM = SexualBehaviourModule()
    # need to set the number of new partners to test for risk reduction cases
    pop.data["num_partners"] = 0
    men = selector(pop.data, sex=(operator.eq, SexType.Male))
    women = selector(pop.data, sex=(operator.eq, SexType.Female))

    # equal numbers of partners
    n_partners_men = n_partners_women = n_partners//2
    # distribute partners amongst the men and women
    men_idx = pop.data.loc[men].index
    women_idx = pop.data.loc[women].index
    for rand_man in rng.choice(men_idx, size=n_partners_men):
        pop.data.loc[rand_man, "num_partners"] += 1
    for rand_woman in rng.choice(women_idx, size=n_partners_women):
        pop.data.loc[rand_woman, "num_partners"] += 1
    SBM.update_rred_balance(pop.data)
    assert np.all(pop.data.loc[men, "rred_balance"] == 1)
    assert np.all(pop.data.loc[women, "rred_balance"] == 1)

    # ~ -1 % discrepancy
    pop.data["num_partners"] = 0
    n_partners_men = int(n_partners * 0.495)
    n_partners_women = int(n_partners * 0.505)
    for i in range(n_partners_men):
        pop.data.loc[rng.choice(men_idx), "num_partners"] += 1
    for i in range(n_partners_women):
        pop.data.loc[rng.choice(women_idx), "num_partners"] += 1
    SBM.update_rred_balance(pop.data)
    assert np.allclose(pop.data.loc[men, "rred_balance"], 1/0.7)
    assert np.allclose(pop.data.loc[women, "rred_balance"], 0.7)

    # > 10 % discrepancy
    pop.data["num_partners"] = 0
    n_partners_men = int(n_partners * 0.6)
    n_partners_women = int(n_partners * 0.45)
    for i in range(n_partners_men):
        pop.data.loc[rng.choice(men_idx), "num_partners"] += 1
    for i in range(n_partners_women):
        pop.data.loc[rng.choice(women_idx), "num_partners"] += 1
    SBM.update_rred_balance(pop.data)
    assert np.all(pop.data.loc[men, "rred_balance"] == 0.1)
    assert np.all(pop.data.loc[women, "rred_balance"] == 10)


def test_rred_art_adherence():
    N = 1000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    SBM = SexualBehaviourModule()
    SBM.init_risk_factors(pop.data)
    # art adherence flag off
    SBM.use_rred_art_adherence = False
    pop.data["art_adherence"] = rng.random(size=N)
    above_idx = selector(pop.data, art_adherence=(operator.ge, 0.8))
    below_idx = selector(pop.data, art_adherence=(operator.lt, 0.8))
    SBM.update_rred(pop)
    assert np.allclose(pop.data["rred_art_adherence"], 1)

    # art adherence flag on
    SBM.use_rred_art_adherence = True
    SBM.update_rred(pop)
    assert np.all(pop.data.loc[above_idx, "rred_art_adherence"] == 1)
    assert np.all(pop.data.loc[below_idx, "rred_art_adherence"] == 2)


def test_rred_art_adherence_usage():
    # check that art adherence risk factor is considered in correct fraction
    # of simulations
    count = 0
    for i in range(100):
        SBM_temp = SexualBehaviourModule()
        count += SBM_temp.use_rred_art_adherence
    assert 0.1 < count/100 < 0.3


def test_rred_population():
    N = 100
    pop = Population(size=N, start_date=date(1989, 1, 1))
    SBM = SexualBehaviourModule()
    SBM.init_risk_factors(pop.data)
    assert(SBM.rred_population == 1)
    pop.date = date(1995, 1, 1)
    SBM.update_rred(pop)
    assert np.isclose(SBM.rred_population, 1)
    pop.date = date(1996, 1, 1)
    SBM.update_rred(pop)
    assert np.isclose(SBM.rred_population, SBM.yearly_risk_change["1990s"], 0.001)
    pop.date = date(1998, 1, 1)
    SBM.update_rred(pop)
    assert np.isclose(SBM.rred_population, SBM.yearly_risk_change["1990s"]**3, 0.001)
    pop.date = date(2000, 1, 1)
    SBM.update_rred(pop)
    assert np.isclose(SBM.rred_population, SBM.yearly_risk_change["1990s"]**5, 0.001)
    pop.date = date(2010, 1, 1)
    SBM.update_rred(pop)
    assert np.isclose(SBM.rred_population, SBM.yearly_risk_change["1990s"]**5, 0.001)
    pop.date = date(2011, 1, 1)
    SBM.update_rred(pop)
    assert np.isclose(SBM.rred_population,
                      SBM.yearly_risk_change["1990s"]**5 *
                      SBM.yearly_risk_change["2010s"], 0.001)
    pop.date = date(2020, 1, 1)
    SBM.update_rred(pop)
    assert np.isclose(SBM.rred_population,
                      SBM.yearly_risk_change["1990s"]**5 *
                      SBM.yearly_risk_change["2010s"]**10, 0.001)
    pop.date = date(2022, 1, 1)
    SBM.update_rred(pop)
    assert np.isclose(SBM.rred_population,
                      SBM.yearly_risk_change["1990s"]**5 *
                      SBM.yearly_risk_change["2010s"]**11, 0.001)


def test_rred_personal():
    N = 100
    pop = Population(size=N, start_date=date(1989, 1, 1))
    # Count how many times we initialise with each threshold 0.3, 0.5, 0.7
    count03, count05, count07 = (0, 0, 0)
    for i in range(100):
        SBM = SexualBehaviourModule()
        SBM.init_risk_factors(pop.data)
        risk_count = sum(selector(pop.data, rred_personal=(operator.lt, 1)))
        count03 += (0.2 < (risk_count/N) < 0.4)  # check consistency with threshold
        count05 += (0.4 < (risk_count/N) < 0.6)  # check consistency with threshold
        count07 += (0.6 < (risk_count/N) < 0.8)  # check consistency with threshold
        assert all([x == 1 or x == 1e-5 for x in pop.data["rred_personal"]])
    assert(1/6 < count03/100 < 1/2)  # check frequency of threshold from initialisations
    assert(1/6 < count05/100 < 1/2)  # check frequency of threshold from initialisations
    assert(1/6 < count07/100 < 1/2)  # check frequency of threshold from initialisations


def test_rred_age():
    N = 11
    pop = Population(size=2*N, start_date=date(1989, 1, 1))
    ages = np.array([12, 17, 22, 27, 32, 37, 42, 47, 52, 57, 62]*2)
    pop.data["age"] = ages
    pop.data["sex"] = np.array([SexType.Male]*N + [SexType.Female]*N)
    pop.sexual_behaviour.init_risk_factors(pop.data)
    # select a particular risk matrix
    risk_factors = np.array(pop.data["rred_age"])
    assert risk_factors[0] == 1  # no update for under 15
    assert risk_factors[11] == 1
    expected_risk_male, expected_risk_female = pop.sexual_behaviour.age_based_risk.T
    assert np.allclose(risk_factors[1:11], expected_risk_male)
    assert np.allclose(risk_factors[12:22], expected_risk_female)
    dt = timedelta(days=90)
    for i in range(20):
        pop.evolve(dt)
    risk_factors = np.array(pop.data["rred_age"])
    expected_risk_male = np.append(expected_risk_male, expected_risk_male[-1])
    expected_risk_female = np.append(expected_risk_female, expected_risk_female[-1])
    assert np.allclose(risk_factors, np.append(expected_risk_male, expected_risk_female))
