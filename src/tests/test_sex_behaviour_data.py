import importlib.resources
from pathlib import Path

import numpy as np
import scipy.linalg as sla

from hivpy.sex_behaviour_data import SexualBehaviourData


def test_probability_loading():
    # Load example file
    data_path = Path(__file__).parent / "test_data" / "sbd_testing.yaml"
    SBD = SexualBehaviourData(data_path)
    dists = SBD._get_discrete_dist_list("Test_Example_List")

    # explicit distribution
    d0 = dists[0]
    N = 10000
    pop0 = d0.sample(size=N)
    for i in range(5):
        v = i + 1
        p = 0.1*i
        count = sum(pop0 == v)
        E_count = p * N
        var = N * p * (1-p)
        assert E_count - 2*var <= count <= E_count + 2*var

    # uniform over range
    d1 = dists[1]
    pop1 = d1.sample(size=N)
    for i in range(1, 11):
        count = sum(pop1 == i)
        E_count = 0.1 * N
        var = N * 0.09
        assert E_count - 2*var <= count <= E_count + 2*var


def test_sex_behaviour_matrices_diagonalisable():
    """Checks that all of the sexual behaviour transition matrices are diagonalisable
    and therefore can be used to calculate transition matrices for variable time-steps.
    This only needs to be run if the matrices change, and doesn't need to be run every time."""
    with importlib.resources.path("hivpy.data", "sex_behaviour.yaml") as data_path:
        SBD = SexualBehaviourData(data_path)
    count_bad = 0
    bad_matrices = []
    all_matrices = [SBD.sex_behaviour_transition_options["Male"],
                    SBD.sex_behaviour_transition_options["Female"]]
    for matList in all_matrices:
        for m in matList:
            print("m = ", m)
            T = np.array(m).transpose()
            evals, evecs = sla.eig(T)
            print(evals)
            print(evecs)
            # Check that eigenvectors are linearly independent
            # i.e. that no eigenvector is a multiple of any other
            for i in range(0, evals.size):
                for j in range(i + 1, evals.size):
                    ratio = evecs[i] / evecs[j]
                    print((i, j), ratio)
                    if np.allclose(ratio, ratio[0]):
                        count_bad += 1
                        bad_matrices.append(T)
            print("***************************************************")
    print(count_bad)
    print(bad_matrices)
    assert (count_bad == 0)
