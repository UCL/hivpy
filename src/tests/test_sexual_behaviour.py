from datetime import date

import numpy as np

from hivpy import sex_behaviour_data as sbd
from hivpy.demographics import SexType
from hivpy.population import Population
from hivpy.sexual_behaviour import SexualBehaviourModule


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
    for trans_matrix in sbd.sex_behaviour_trans_male_options:
        assert (trans_matrix.shape == (4, 4))
        check_prob_sums(SexType.Male, trans_matrix)
    # for trans_matrix in sbd.sex_behaviour_trans_female_options:
    #     assert (trans_matrix.shape == (2, 2))
    #     check_prob_sums(1, trans_matrix)


def check_num_partners(row):
    sex = row["sex"]
    group = row["sex_behaviour"]
    n = row["num_partners"]
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
    print(pop_data)
    # Check the num_partners column
    checks = pop_data.apply(check_num_partners, axis=1)
    assert np.all(checks)
