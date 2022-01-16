import numpy as np

from hivpy.sexual_behaviour import (prob_transition,
                                    sex_behaviour_trans_female,
                                    sex_behaviour_trans_male)


def check_prob_sums(gender, trans_matrices):
    (n_mat, dim, _) = trans_matrices.shape
    for i in range(0, dim):
        for age in [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]:
            tot_prob = 0
            for j in range(0, dim):
                tot_prob += prob_transition(gender, age, i, j)
            assert(np.isclose(tot_prob, 1.0))


def test_transition_probabilities():
    assert (sex_behaviour_trans_male.shape == (15, 4, 4))
    assert (sex_behaviour_trans_female.shape == (15, 2, 2))
    check_prob_sums(1, sex_behaviour_trans_female)
    check_prob_sums(0, sex_behaviour_trans_male)
