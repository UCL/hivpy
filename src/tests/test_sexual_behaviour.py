import numpy as np
import pytest
from hivpy import sex_behaviour_data as sbd
from hivpy import sexual_behaviour
from hivpy.sexual_behaviour import SexualBehaviourModule


def check_prob_sums(sex, trans_matrix):
    SBM = SexualBehaviourModule()
    if(sex == 0):
        SBM.sex_behaviour_trans_male = trans_matrix
    else:
        SBM.sex_behaviour_trans_female = trans_matrix
    (dim, _) = trans_matrix.shape
    for i in range(0, dim):
        for age in [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]:
            tot_prob = 0
            for j in range(0, dim):
                tot_prob += SBM.prob_transition(sex, age, i, j)
            assert(np.isclose(tot_prob, 1.0))


def test_transition_probabilities():
    for trans_matrix in sbd.sex_behaviour_trans_male_options:
        assert (trans_matrix.shape == (4, 4))
        check_prob_sums(0, trans_matrix)
    for trans_matrix in sbd.sex_behaviour_trans_female_options:
        assert (trans_matrix.shape == (2, 2))
        check_prob_sums(1, trans_matrix)

