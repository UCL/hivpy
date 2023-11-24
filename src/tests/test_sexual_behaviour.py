import importlib.resources
import operator
from datetime import date, timedelta

import numpy as np
import pytest
import yaml

import hivpy.column_names as col
from hivpy.common import SexType, rng, seedManager
from hivpy.population import Population
from hivpy.sexual_behaviour import (SexBehaviourClass, SexBehaviours,
                                    SexualBehaviourModule)


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
        risk = population.get_variable(col.RISK)
        assert len(risk) == 11
        assert (0 < SBM.new_partner_factor <= 2)
        tot_prob = np.array([0.0]*len(ages))  # probability for each age range
        for j in range(0, dim):
            tot_prob += SBM.prob_transition(sex, risk, i, j)
        assert (np.allclose(tot_prob, 1.0))


def test_transition_probabilities(yaml_data):
    for trans_matrix in np.array(yaml_data["sex_behaviour_transition_options"]["Male"]):
        assert (trans_matrix.shape == (4, 4))
        check_prob_sums(SexType.Male, trans_matrix)
    for trans_matrix in np.array(yaml_data["sex_behaviour_transition_options"]["Female"]):
        assert (trans_matrix.shape == (2, 2))
        check_prob_sums(SexType.Female, trans_matrix)


def test_sex_behaviour_transition(yaml_data):
    N = 10000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    pop.set_present_variable(col.AGE, 25)  # make whole population active
    pop.set_present_variable(col.RISK, 1)  # risk factors can be tested elsewhere
    # set population to each group
    trans_matrix = pop.sexual_behaviour.sex_behaviour_trans
    for s in SexBehaviourClass:
        pop.set_present_variable(col.SEX_BEHAVIOUR_CLASS, s)
        for g in SexBehaviours[s]:
            pop.set_present_variable(col.SEX_BEHAVIOUR, g)
            pop.sexual_behaviour.update_sex_groups(pop)
            for g2 in SexBehaviours[s]:
                num = len(pop.data[(pop.data[col.SEX_BEHAVIOUR] == g2)])
                p = trans_matrix[s][g][g2] / (sum(trans_matrix[s][g]))
                E = p * N
                sig = np.sqrt(E * (1-p))
                lower = E - 5*sig
                upper = E + 5*sig
                assert (lower <= num <= upper)


def check_num_partners(row):
    sex = row["sex"]
    sex_worker = row[col.SEX_WORKER]
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
        # Female
        if not sex_worker:
            if group == 0:
                return n == 0
            else:
                if age < 25:
                    return n in range(1, 10)
                else:
                    return n in range(1, 4)
        else:
            if group == 0:
                return n == 0
            elif group == 1:
                return n in range(1, 7)
            elif group == 2:
                return n in range(7, 21)
            elif group == 3:
                return n in range(21, 51 if age < 30 else 31)
            elif group == 4:
                return (n in range(51, 150)) if age < 30 else (n == 30)


def test_num_partners():
    """
    Check that number of partners are reasonable.
    """
    pop = Population(size=10000, start_date=date(1989, 1, 1))
    # test sex worker program separately
    pop.sexual_behaviour.sex_worker_program = False
    pop.sexual_behaviour.num_short_term_partners(pop)
    assert (any(pop.data["num_partners"] > 0))
    # Check the num_partners column
    checks = pop.data.apply(check_num_partners, axis=1)
    assert np.all(checks)

    # set all women to sex workers
    pop.data[col.AGE] = 35
    pop.set_present_variable(col.SEX_WORKER,
                             True,
                             pop.get_sub_pop([(col.SEX,
                                               operator.eq,
                                               SexType.Female)]))
    pop.sexual_behaviour.update_sex_behaviour_class(pop)
    pop.sexual_behaviour.init_sex_behaviour_groups(pop)
    pop.sexual_behaviour.num_short_term_partners(pop)
    checks = pop.data.apply(check_num_partners, axis=1)
    assert np.all(checks)


def test_num_partner_for_behaviour_group():
    """
    Check that approx correct number of partners are generated for each
    sex class, age, and sexual behaviour group.
    """
    N = 10000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    for sex_class in SexBehaviourClass:
        np_probs = pop.sexual_behaviour.short_term_partners[sex_class]
        for g in range(len(np_probs)):
            pop.data[col.SEX_BEHAVIOUR_CLASS] = sex_class
            pop.data[col.SEX_BEHAVIOUR] = g
            pop.sexual_behaviour.num_short_term_partners(pop)
            if sex_class == SexBehaviourClass.SEX_WORKER_O30:
                if 30 in np_probs[g].data:
                    found = False
                    i = 0
                    while not found:
                        if np_probs[g].data[i] == 30:
                            found = True
                        else:
                            i += 1
                    prob_ge_30 = sum(np_probs[g].probs[i:])

            for (n, p) in zip(np_probs[g].data, np_probs[g].probs):
                sub_pop = pop.data.loc[(pop.data[col.SEX_BEHAVIOUR_CLASS] == sex_class) & (
                    pop.data[col.SEX_BEHAVIOUR] == g) & (pop.data[col.AGE] >= 15)]
                if sex_class == SexBehaviourClass.SEX_WORKER_O30:
                    if n > 30:
                        p = 0
                    elif n == 30:
                        p = prob_ge_30

                N_g = len(sub_pop)
                E = p * N_g
                sig = np.sqrt(p * (1-p) * N)
                X = sum(sub_pop[col.NUM_PARTNERS] == n)
                assert (E - 3*sig <= X <= E + 3*sig)


def test_behaviour_updates():
    """
    Check that at least one person changes sexual behaviour groups.
    """
    pop = Population(size=100000, start_date=date(1989, 1, 1))
    initial_groupings = pop.data["sex_behaviour"].copy()
    for i in range(5):
        pop.sexual_behaviour.update_sex_groups(pop)
    subsequent_groupings = pop.data["sex_behaviour"]
    assert (any(initial_groupings != subsequent_groupings))


def test_initial_sex_behaviour_groups(yaml_data):
    """
    Check that the number of people initialised into each sexual behaviour group
    is within 3-sigma of the expectation value, as calculated by a binomial distribution.
    """
    N = 10000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    probs = {SexType.Male:
             yaml_data["initial_sex_behaviour_probabilities"]["Male"]["Probability"],
             SexType.Female:
             yaml_data["initial_sex_behaviour_probabilities"]["Female"]["Probability"]}
    for sex in SexType:
        sex_sub_pop = pop.get_sub_pop([(col.SEX, operator.eq, sex)])
        n_sex = len(sex_sub_pop)
        Prob_sex = np.array(probs[sex])
        Prob_sex /= sum(Prob_sex)
        for g in SexBehaviours[sex]:
            group_sub_pop = pop.get_sub_pop([(col.SEX_BEHAVIOUR, operator.eq, g)])
            p = Prob_sex[g]
            sigma = np.sqrt(p*(1-p)*float(n_sex))
            Expectation = float(n_sex)*p
            N_g = len(pop.get_sub_pop_intersection(sex_sub_pop, group_sub_pop))
            assert Expectation - sigma*3 <= N_g <= Expectation + sigma*3


def test_risk_long_term_partner():
    N = 1000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    # pick some indices and give those people LTPs
    indices = rng.integers(0, N, size=15)
    pop.set_present_variable(col.LONG_TERM_PARTNER, False)
    pop.set_present_variable(col.LONG_TERM_PARTNER, True, indices)
    SBM = SexualBehaviourModule()
    SBM.update_risk_long_term_partnered(pop)
    assert all(pop.data.loc[indices, "risk_long_term_partnered"] == SBM.ltp_risk_factor)


def test_risk_adc():
    N = 20
    pop = Population(size=N, start_date=date(1989, 1, 1))
    pop.data["HIV_status"] = False  # make test independent of how HIV is initialised
    init_HIV_idx = rng.integers(0, N, size=5)
    init_ADC_idx = init_HIV_idx[[0, 2, 4]]
    pop.data.loc[init_HIV_idx, col.HIV_STATUS] = True
    pop.data.loc[init_ADC_idx, col.ADC] = True
    expected_risk = np.ones(N)
    expected_risk[init_ADC_idx] = 0.2
    SBM = SexualBehaviourModule()
    # initialise risk_adc
    SBM.init_risk_factors(pop)
    # Check only people with HIV have risk_adc = 0.2
    assert np.all(pop.data["risk_adc"] == expected_risk)
    # Assign more people with HIV
    add_HIV_idx = rng.integers(0, N, size=5)
    add_ADC_idx = add_HIV_idx[[0, 1, 2]]
    pop.data.loc[add_HIV_idx, col.HIV_STATUS] = True
    pop.data.loc[add_ADC_idx, col.ADC] = True
    # Update risk factors
    SBM.update_sex_behaviour(pop)
    expected_risk[add_ADC_idx] = 0.2
    assert np.all(pop.data["risk_adc"] == expected_risk)


def test_risk_diagnosis():
    N = 20
    pop = Population(size=N, start_date=date(1989, 1, 1))
    pop.data["HIV_status"] = False  # init everyone HIV negative
    SBM = SexualBehaviourModule()
    SBM.risk_diagnosis = 4
    SBM.risk_diagnosis_period = timedelta(700)
    SBM.update_risk_diagnosis(pop)
    assert np.all(pop.data["risk_diagnosis"] == 1)
    # give up some people HIV and advance the date
    HIV_idx = rng.integers(0, N, size=5)
    pop.data.loc[HIV_idx, "HIV_status"] = True
    pop.data.loc[HIV_idx, "HIV_Diagnosis_Date"] = pop.date
    SBM.update_risk_diagnosis(pop)
    assert all(pop.data.loc[HIV_idx, "risk_diagnosis"] == 4)
    pop.date += timedelta(days=365)
    SBM.update_risk_diagnosis(pop)
    assert all(pop.data.loc[HIV_idx, "risk_diagnosis"] == 4)
    pop.date += timedelta(days=500)
    SBM.update_risk_diagnosis(pop)
    assert all(pop.data.loc[HIV_idx, "risk_diagnosis"] == 2)
    undiagnosed = [x for x in range(0, N) if x not in HIV_idx]
    assert all(pop.data.loc[undiagnosed, "risk_diagnosis"] == 1)


def test_risk_balance():
    N = 2000
    n_partners = 1000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    SBM = SexualBehaviourModule()
    # need to set the number of new partners to test for risk reduction cases
    men = pop.get_sub_pop([(col.SEX, operator.eq, SexType.Male)])
    women = pop.get_sub_pop([(col.SEX, operator.eq, SexType.Female)])

    def test_partner_ratios(male_ratio, balance_male, balance_female):
        pop.data["num_partners"] = 0
        n_partners_men = int(n_partners * male_ratio)
        n_partners_women = int(n_partners * (1-male_ratio))
        # distribute partners amongst the men and women
        random_men = rng.choice(men, size=n_partners_men, replace=False)
        pop.set_present_variable(col.NUM_PARTNERS,
                                 pop.get_variable(col.NUM_PARTNERS, random_men) + 1,
                                 random_men)
        random_women = rng.choice(women, size=n_partners_women, replace=False)
        pop.set_present_variable(col.NUM_PARTNERS,
                                 pop.get_variable(col.NUM_PARTNERS, random_women) + 1,
                                 random_women)
        SBM.update_risk_balance(pop)
        assert np.all(pop.data.loc[men, "risk_balance"] == balance_male)
        assert np.all(pop.data.loc[women, "risk_balance"] == balance_female)

    # equal numbers of partners
    test_partner_ratios(0.5, 1, 1)

    # ~ -1 % discrepancy
    test_partner_ratios(0.495, 1/0.7, 0.7)

    # > 10 % discrepancy
    test_partner_ratios(0.6, 0.1, 10)


def test_risk_art_adherence():
    N = 1000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    SBM = SexualBehaviourModule()
    pop.init_variable(col.ART_ADHERENCE, rng.random(size=N))
    SBM.init_risk_factors(pop)
    # art adherence flag off
    SBM.use_risk_art_adherence = False
    pop_over_threshold = pop.get_sub_pop([(col.ART_ADHERENCE, operator.ge, 0.8)])
    pop_under_threshold = pop.get_sub_pop([(col.ART_ADHERENCE, operator.lt, 0.8)])
    SBM.update_risk(pop)
    assert np.allclose(pop.data["risk_art_adherence"], 1)

    # art adherence flag on
    SBM.use_risk_art_adherence = True
    SBM.update_risk(pop)
    assert np.all(pop.get_variable(col.RISK_ART_ADHERENCE, pop_over_threshold) == 1)
    assert np.all(pop.get_variable(col.RISK_ART_ADHERENCE, pop_under_threshold) == 2)


def test_risk_art_adherence_usage():
    # check that art adherence risk factor is considered in correct fraction
    # of simulations
    count = 0
    for i in range(100):
        SBM_temp = SexualBehaviourModule()
        count += SBM_temp.use_risk_art_adherence
    assert 0.1 < count/100 < 0.3


def test_risk_population():
    N = 100
    pop = Population(size=N, start_date=date(1989, 1, 1))
    SBM = SexualBehaviourModule()
    SBM.init_risk_factors(pop)
    assert (SBM.risk_population == 1)
    pop.date = date(1995, 1, 1)
    SBM.update_risk(pop)
    assert np.isclose(SBM.risk_population, 1)
    pop.date = date(1996, 1, 1)
    SBM.update_risk(pop)
    assert np.isclose(SBM.risk_population, SBM.yearly_risk_change["1990s"], 0.001)
    pop.date = date(1998, 1, 1)
    SBM.update_risk(pop)
    assert np.isclose(SBM.risk_population, SBM.yearly_risk_change["1990s"]**3, 0.001)
    pop.date = date(2000, 1, 1)
    SBM.update_risk(pop)
    assert np.isclose(SBM.risk_population, SBM.yearly_risk_change["1990s"]**5, 0.001)
    pop.date = date(2010, 1, 1)
    SBM.update_risk(pop)
    assert np.isclose(SBM.risk_population, SBM.yearly_risk_change["1990s"]**5, 0.001)
    pop.date = date(2011, 1, 1)
    SBM.update_risk(pop)
    assert np.isclose(SBM.risk_population,
                      SBM.yearly_risk_change["1990s"]**5 *
                      SBM.yearly_risk_change["2010s"], 0.001)
    pop.date = date(2020, 1, 1)
    SBM.update_risk(pop)
    assert np.isclose(SBM.risk_population,
                      SBM.yearly_risk_change["1990s"]**5 *
                      SBM.yearly_risk_change["2010s"]**10, 0.001)
    pop.date = date(2022, 1, 1)
    SBM.update_risk(pop)
    assert np.isclose(SBM.risk_population,
                      SBM.yearly_risk_change["1990s"]**5 *
                      SBM.yearly_risk_change["2010s"]**11, 0.001)


def test_risk_personal():
    N = 100
    pop = Population(size=N, start_date=date(1989, 1, 1))
    # Count how many times we initialise with each threshold 0.3, 0.5, 0.7
    count03, count05, count07 = (0, 0, 0)
    universal_seed = seedManager.UniversalSeed
    for i in range(100):
        seedManager.UniversalSeed = rng.integers(0, 1000)
        SBM = SexualBehaviourModule()
        SBM.init_risk_factors(pop)
        risk_count = len(pop.get_sub_pop([(col.RISK_PERSONAL, operator.lt, 1)]))
        count03 += (0.2 < (risk_count/N) < 0.4)  # check consistency with threshold
        count05 += (0.4 < (risk_count/N) < 0.6)  # check consistency with threshold
        count07 += (0.6 < (risk_count/N) < 0.8)  # check consistency with threshold
        seedManager.UniversalSeed = universal_seed  # put seed back in case assert fails
        assert ([x == 1 or x == 1e-5 for x in pop.data["risk_personal"]])
    assert (1/6 < count03/100 < 1/2)  # check frequency of threshold from initialisations
    assert (1/6 < count05/100 < 1/2)  # check frequency of threshold from initialisations
    assert (1/6 < count07/100 < 1/2)  # check frequency of threshold from initialisations
    

def test_risk_age():
    N = 11
    # rng.set_seed(42)
    pop = Population(size=2*N, start_date=date(1989, 1, 1))
    ages = np.array([12, 17, 22, 27, 32, 37, 42, 47, 52, 57, 62]*2)
    pop.set_present_variable(col.SEX_BEHAVIOUR_CLASS, 0)
    pop.set_present_variable(col.AGE, ages)
    pop.set_present_variable(col.SEX, np.array([SexType.Male]*N + [SexType.Female]*N))
    # pop.set_present_variable(col.SEX_WORKER, False)
    pop.sexual_behaviour.update_sex_behaviour_class(pop)
    pop.sexual_behaviour.init_sex_behaviour_groups(pop)
    pop.sexual_behaviour.num_short_term_partners(pop)
    pop.sexual_behaviour.init_risk_factors(pop)
    # select a particular risk matrix
    risk_factors = np.array(pop.data[col.RISK_AGE])
    assert risk_factors[0] == 1  # no update for under 15
    assert risk_factors[11] == 1
    expected_risk_male, expected_risk_female = pop.sexual_behaviour.age_based_risk.T
    assert np.allclose(risk_factors[1:11], expected_risk_male)
    assert np.allclose(risk_factors[12:22], expected_risk_female)
    dt = timedelta(days=90)
    for i in range(20):
        pop.evolve(dt)
    risk_factors = np.array(pop.data[col.RISK_AGE])
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


# Test Sex Work related code

@pytest.mark.parametrize(["life_sex_risk", "p_start"],
                         [(1, np.array([0., 0., 0., 0.])),
                          (2, np.array([0.005, 0.01, 0.02, 0.03])),
                          (3, np.array([0.05, 0.1, 0.2, 0.3]))])
def test_start_sex_work(life_sex_risk, p_start):
    N = 10000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    sb_module = pop.sexual_behaviour
    pop.set_present_variable(col.SEX_WORKER, False)
    pop.set_present_variable(col.LIFE_SEX_RISK, life_sex_risk)
    # dummy sex behaviour variables
    pop.set_present_variable(col.AGE, [17, 22, 27, 37, 50]*2000)
    sb_module.risk_population = 4
    sb_module.base_start_sw = 0.005
    sb_module.risk_sex_worker_age = np.array([0.5, 1.0, 2.0, 3.0])

    sb_module.update_sex_worker_status(pop)
    # assert any(pop.get_variable(col.SEX_WORKER))

    # Men and under 15s / over 50s should not be sex workers
    men = pop.get_sub_pop([(col.SEX, operator.eq, SexType.Male)])
    under_15 = pop.get_sub_pop([(col.AGE, operator.lt, 15)])
    over_50s = pop.get_sub_pop([(col.AGE, operator.ge, 50)])
    men_sex_worker = pop.get_variable(col.SEX_WORKER, men)
    assert not any(men_sex_worker)
    assert not any(pop.get_variable(col.SEX_WORKER, under_15))
    assert not any(pop.get_variable(col.SEX_WORKER, over_50s))

    # Proportion of sex workers
    n_15to19 = sum(pop.get_variable(col.SEX_WORKER, pop.get_sub_pop([(col.AGE, operator.eq, 17)])))
    n_20to24 = sum(pop.get_variable(col.SEX_WORKER, pop.get_sub_pop([(col.AGE, operator.eq, 22)])))
    n_25to34 = sum(pop.get_variable(col.SEX_WORKER, pop.get_sub_pop([(col.AGE, operator.eq, 27)])))
    n_34to49 = sum(pop.get_variable(col.SEX_WORKER, pop.get_sub_pop([(col.AGE, operator.eq, 37)])))
    E = p_start * (N * 0.1)  # only 1/5 of the population in each age group and half female
    sigma = np.sqrt(E * (1-p_start))
    assert (E[0] - 3*sigma[0] <= n_15to19 <= E[0] + 3*sigma[0])
    assert (E[1] - 3*sigma[1] <= n_20to24 <= E[1] + 3*sigma[1])
    assert (E[2] - 3*sigma[2] <= n_25to34 <= E[2] + 3*sigma[2])
    assert (E[3] - 3*sigma[3] <= n_34to49 <= E[3] + 3*sigma[3])


def test_stopping_sex_work():
    N = 10000
    pop = Population(size=N, start_date=date(1989, 1, 1))
    sb_module = pop.sexual_behaviour
    # dummy sex behaviour variables
    sb_module.risk_population = 4
    sb_module.base_stop_sw = 0.1

    pop.set_present_variable(col.SEX_WORKER, True)

    # Age >= 50 means everyone should stop sex work
    pop.set_present_variable(col.AGE, 50)
    sb_module.update_sex_worker_status(pop)
    assert not any(pop.get_variable(col.SEX_WORKER))
    stop_sex_work_age = pop.get_variable(col.AGE_STOP_SEX_WORK)
    assert all(stop_sex_work_age == 50)
    date_stop_sex_work = pop.get_variable(col.DATE_STOP_SW)
    assert all(date_stop_sex_work == date(1989, 1, 1))  # date hasn't changed

    # Test for ages between 40 and 50
    pop.set_present_variable(col.SEX_WORKER, True)
    pop.set_present_variable(col.AGE, 45)
    sb_module.update_sex_worker_status(pop)
    num_stop = N - sum(pop.get_variable(col.SEX_WORKER))
    P_stop_over40 = 0.15  # based on dummy variables above
    E = P_stop_over40 * N
    sigma = np.sqrt(E * (1 - P_stop_over40))
    assert ((E - 3 * sigma) < num_stop < (E + 3 * sigma))

    # Test for ages under 40
    pop.set_present_variable(col.SEX_WORKER, True)
    pop.set_present_variable(col.AGE, 30)
    sb_module.update_sex_worker_status(pop)
    num_stop = N - sum(pop.get_variable(col.SEX_WORKER))
    P_stop_under40 = 0.05  # based on dummy variables above
    E = P_stop_under40 * N
    sigma = np.sqrt(E * (1 - P_stop_under40))
    assert ((E - 3 * sigma) < num_stop < (E + 3 * sigma))
