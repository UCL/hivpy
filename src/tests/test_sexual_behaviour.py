import importlib.resources
import operator
from datetime import date, timedelta

import numpy as np
import pytest
import yaml

import hivpy.column_names as col
from hivpy.common import SexType, rng, selector
from hivpy.population import Population
from hivpy.sexual_behaviour import SexBehaviours, SexualBehaviourModule


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
        population = Population(11, date(1989, 1, 1))
        population.set_present_variable(col.AGE, ages)
        population.set_present_variable(col.SEX, sexes)
        population.set_present_variable(col.HIV_STATUS, False)
        population.set_present_variable(col.HIV_DIAGNOSIS_DATE, None)
        SBM.init_risk_factors(population)
        rred = population.get_variable(col.RRED)
        assert len(rred) == 11
        assert (0 < SBM.new_partner_factor <= 2)
        tot_prob = np.array([0.0]*len(ages))  # probability for each age range
        for j in range(0, dim):
            tot_prob += SBM.prob_transition(sex, rred, i, j)
        assert (np.allclose(tot_prob, 1.0))


def test_transition_probabilities(yaml_data):
    for trans_matrix in np.array(yaml_data["sex_behaviour_transition_options"]["Male"]):
        assert (trans_matrix.shape == (4, 4))
        check_prob_sums(SexType.Male, trans_matrix)
    for trans_matrix in np.array(yaml_data["sex_behaviour_transition_options"]["Female"]):
        assert (trans_matrix.shape == (2, 2))
        check_prob_sums(SexType.Female, trans_matrix)


def test_sex_behaviour_transition(yaml_data):
    N = 100000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    pop.set_present_variable(col.AGE, 25)  # make whole population active
    pop.set_present_variable(col.RRED, 1)  # rred factors can be tested elsewhere
    # set population to each group
    trans_matrix = pop.sexual_behaviour.sex_behaviour_trans
    for s in SexType:
        num_sex = sum(pop.data[col.SEX] == s)
        for g in SexBehaviours[s]:
            print(g)
            pop.set_present_variable(col.SEX_BEHAVIOUR, 0)
            pop.set_present_variable(col.SEX_BEHAVIOUR, g,
                                     pop.get_sub_pop([(col.SEX, operator.eq, s)]))
            pop.sexual_behaviour.update_sex_groups(pop)
            for g2 in SexBehaviours[s]:
                num = len(pop.data[(pop.data[col.SEX_BEHAVIOUR] == g2) & (pop.data[col.SEX] == s)])
                p = trans_matrix[s][g][g2] / (sum(trans_matrix[s][g]))
                E = p * num_sex
                sig = np.sqrt(E * (1-p))
                lower = E - 5*sig
                upper = E + 5*sig
                assert (lower <= num <= upper)


def check_num_partners(row):
    sex = row["sex"]
    group = row["sex_behaviour"]
    n = row["num_partners"]
    age = row["age"]
    if age < 15 or age >= 65:  # no sexual partners for under 16s
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
    pop = Population(size=100000, start_date=date(1989, 1, 1))
    pop.sexual_behaviour.num_short_term_partners(pop)
    assert (any(pop.data["num_partners"] > 0))
    # Check the num_partners column
    checks = pop.data.apply(check_num_partners, axis=1)
    assert np.all(checks)


def test_num_partner_for_behaviour_group():
    """Check that approx correct number of partners are generated for each
    sex, age, and sexual behaviour group"""
    N = 100000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    for sex in SexType:
        np_probs = pop.sexual_behaviour.short_term_partners[sex]
        for g in range(len(np_probs)):
            for (n, p) in zip(np_probs[g].data, np_probs[g].probs):
                sub_pop = pop.data.loc[(pop.data[col.SEX] == sex) & (
                    pop.data[col.SEX_BEHAVIOUR] == g) & (pop.data[col.AGE] >= 15)]
                N_g = len(sub_pop)
                E = p * N_g
                sig = np.sqrt(p * (1-p) * N)
                X = sum(sub_pop[col.NUM_PARTNERS] == n)
                assert (E - 3*sig <= X <= E + 3*sig)


def test_behaviour_updates():
    """Check that at least one person changes sexual behaviour groups"""
    pop = Population(size=100000, start_date=date(1989, 1, 1))
    initial_groupings = pop.data["sex_behaviour"].copy()
    for i in range(5):
        pop.sexual_behaviour.update_sex_groups(pop)
    subsequent_groupings = pop.data["sex_behaviour"]
    assert (any(initial_groupings != subsequent_groupings))


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
    pop.set_present_variable(col.LONG_TERM_PARTNER, False)
    pop.set_present_variable(col.LONG_TERM_PARTNER, True, indices)
    SBM = SexualBehaviourModule()
    SBM.update_rred_long_term_partnered(pop)
    assert all(pop.data.loc[indices, "rred_long_term_partnered"] == SBM.ltp_risk_factor)


def test_rred_adc():
    N = 20
    pop = Population(size=N, start_date=date(1989, 1, 1))
    pop.data["HIV_status"] = False  # make test independent of how HIV is initialised
    init_HIV_idx = rng.integers(0, N, size=5)
    init_ADC_idx = init_HIV_idx[[0, 2, 4]]
    pop.data.loc[init_HIV_idx, col.HIV_STATUS] = True
    pop.data.loc[init_ADC_idx, col.ADC] = True
    expected_rred = np.ones(N)
    expected_rred[init_ADC_idx] = 0.2
    SBM = SexualBehaviourModule()
    # initialise rred_adc
    SBM.init_risk_factors(pop)
    # Check only people with HIV have rred_adc = 0.2
    assert np.all(pop.data["rred_adc"] == expected_rred)
    # Assign more people with HIV
    add_HIV_idx = rng.integers(0, N, size=5)
    add_ADC_idx = add_HIV_idx[[0, 1, 2]]
    pop.data.loc[add_HIV_idx, col.HIV_STATUS] = True
    pop.data.loc[add_ADC_idx, col.ADC] = True
    # Update rred factors
    SBM.update_sex_behaviour(pop)
    expected_rred[add_ADC_idx] = 0.2
    assert np.all(pop.data["rred_adc"] == expected_rred)


def test_rred_diagnosis():
    N = 20
    pop = Population(size=N, start_date=date(1989, 1, 1))
    pop.data["HIV_status"] = False  # init everyone HIV negative
    SBM = SexualBehaviourModule()
    SBM.rred_diagnosis = 4
    SBM.rred_diagnosis_period = timedelta(700)
    SBM.update_rred_diagnosis(pop)
    assert np.all(pop.data["rred_diagnosis"] == 1)
    # give up some people HIV and advance the date
    HIV_idx = rng.integers(0, N, size=5)
    pop.data.loc[HIV_idx, "HIV_status"] = True
    pop.data.loc[HIV_idx, "HIV_Diagnosis_Date"] = pop.date
    SBM.update_rred_diagnosis(pop)
    assert all(pop.data.loc[HIV_idx, "rred_diagnosis"] == 4)
    pop.date += timedelta(days=365)
    SBM.update_rred_diagnosis(pop)
    assert all(pop.data.loc[HIV_idx, "rred_diagnosis"] == 4)
    pop.date += timedelta(days=500)
    SBM.update_rred_diagnosis(pop)
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
    SBM.update_rred_balance(pop)
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
    SBM.update_rred_balance(pop)
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
    SBM.update_rred_balance(pop)
    assert np.all(pop.data.loc[men, "rred_balance"] == 0.1)
    assert np.all(pop.data.loc[women, "rred_balance"] == 10)


def test_rred_art_adherence():
    N = 1000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    SBM = SexualBehaviourModule()
    pop.init_variable(col.ART_ADHERENCE, rng.random(size=N))
    SBM.init_risk_factors(pop)
    # art adherence flag off
    SBM.use_rred_art_adherence = False
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
    SBM.init_risk_factors(pop)
    assert (SBM.rred_population == 1)
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
        SBM.init_risk_factors(pop)
        risk_count = sum(selector(pop.data, rred_personal=(operator.lt, 1)))
        count03 += (0.2 < (risk_count/N) < 0.4)  # check consistency with threshold
        count05 += (0.4 < (risk_count/N) < 0.6)  # check consistency with threshold
        count07 += (0.6 < (risk_count/N) < 0.8)  # check consistency with threshold
        assert all([x == 1 or x == 1e-5 for x in pop.data["rred_personal"]])
    assert (1/6 < count03/100 < 1/2)  # check frequency of threshold from initialisations
    assert (1/6 < count05/100 < 1/2)  # check frequency of threshold from initialisations
    assert (1/6 < count07/100 < 1/2)  # check frequency of threshold from initialisations


def test_rred_age():
    N = 11
    pop = Population(size=2*N, start_date=date(1989, 1, 1))
    ages = np.array([12, 17, 22, 27, 32, 37, 42, 47, 52, 57, 62]*2)
    pop.data["age"] = ages
    pop.data["sex"] = np.array([SexType.Male]*N + [SexType.Female]*N)
    pop.sexual_behaviour.init_sex_behaviour_groups(pop)
    pop.sexual_behaviour.num_short_term_partners(pop)
    pop.sexual_behaviour.init_risk_factors(pop)
    # select a particular risk matrix
    risk_factors = np.array(pop.data[col.RRED_AGE])
    assert risk_factors[0] == 1  # no update for under 15
    assert risk_factors[11] == 1
    expected_risk_male, expected_risk_female = pop.sexual_behaviour.age_based_risk.T
    print(expected_risk_male)
    print("********************************************")
    print(expected_risk_female)
    print("*************************************************")
    print(risk_factors)
    assert np.allclose(risk_factors[1:11], expected_risk_male)
    assert np.allclose(risk_factors[12:22], expected_risk_female)
    dt = timedelta(days=90)
    for i in range(20):
        pop.evolve(dt)
    risk_factors = np.array(pop.data[col.RRED_AGE])
    print("*************************************************")
    print(risk_factors)
    print("*************************************************")
    print(np.array(pop.data[col.AGE]))
    expected_risk_male = np.append(expected_risk_male, expected_risk_male[-1])
    expected_risk_female = np.append(expected_risk_female, expected_risk_female[-1])
    assert np.allclose(risk_factors, np.append(expected_risk_male, expected_risk_female))


# Test long term partnerships
# test the start of new ltps, end of ltps, and longevity.

def test_start_ltp():
    N = 10000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    sb_mod = pop.sexual_behaviour
    assert (0.1 * np.exp(0.25 * (-5))) <= sb_mod.new_ltp_rate <= (0.1 * np.exp(0.25 * 5))

    for age, modifier, longevity_prob in [(20, 1, [0.3, 0.3, 0.4]),
                                          (40, 2, [0.3, 0.3, 0.4]),
                                          (50, 3, [0.3, 0.5, 0.2]),
                                          (60, 5, [0.3, 0.7, 0.0])]:
        pop.data[col.LONG_TERM_PARTNER] = False
        pop.data[col.AGE] = age

        sb_mod.update_long_term_partners(pop)
        # check that the rate of new partnerships is in the expected range

        num_ltp = sum(pop.data[col.LONG_TERM_PARTNER])
        expected_ltp = sb_mod.new_ltp_rate * N / modifier
        sigma_ltp = np.sqrt(expected_ltp * (1 - sb_mod.new_ltp_rate / modifier))
        assert (expected_ltp - 3 * sigma_ltp) <= num_ltp <= (expected_ltp + 3 * sigma_ltp)

        # test longevity
        partnered_idx = pop.data.index[pop.data[col.LONG_TERM_PARTNER]]
        partnered_pop = pop.data.loc[partnered_idx]
        n_partnered = len(partnered_idx)
        longevity_totals = [sum(partnered_pop[col.LTP_LONGEVITY] == v) for v in [1, 2, 3]]
        for prob, total in zip(longevity_prob, longevity_totals):
            expected_total = prob * n_partnered
            sigma_total = np.sqrt((1 - prob) * expected_total)
            assert (expected_total - 3 * sigma_total) <= total <= (expected_total + 3 * sigma_total)

        # TODO: add checks for correct balancing factors & dates


@pytest.mark.parametrize(["longevity", "rate_change"], [(1, 0.25), (2, 0.05), (3, 0.02)])
def test_end_ltp(longevity, rate_change):
    # TODO: will need to be updated when addition ltp factors added
    # e.g. for diagnosis & balancing

    N = 10000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    sb_mod = pop.sexual_behaviour
    pop.data[col.AGE] = 20  # make everyone sexually active

    pop.data[col.LONG_TERM_PARTNER] = True
    pop.data[col.LTP_LONGEVITY] = longevity
    expected_end_prob = rate_change  # TODO: update for date changes
    expected_ends = expected_end_prob * N
    sigma_ends = (1 - expected_end_prob) * expected_ends
    sb_mod.update_long_term_partners(pop)
    n_ends = N - sum(pop.data[col.LONG_TERM_PARTNER])
    assert (expected_ends - 3 * sigma_ends) <= n_ends <= (expected_ends + 3 * sigma_ends)
