import operator
from datetime import date

import numpy as np
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
        tot_prob = np.array([0.0]*len(ages))  # probability for each age range
        for j in range(0, dim):
            tot_prob += SBM.prob_transition(sex, ages, i, j)
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
        Prob_sex = np.array(probs[sex].copy())
        Prob_sex /= sum(Prob_sex)
        for g in SexBehaviours[sex]:
            index_group = selector(pop_data, sex_behaviour=(operator.eq, g))
            p = Prob_sex[g]
            sigma = np.sqrt(p*(1-p)*float(n_sex))
            Expectation = float(n_sex)*p
            N_g = sum(index_sex & index_group)
            assert Expectation - sigma*3 <= N_g <= Expectation + sigma*3
