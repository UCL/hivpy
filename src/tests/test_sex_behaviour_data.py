from pathlib import Path

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

    # uniform over specific values
    d2 = dists[2]
    pop2 = d2.sample(size=N)
    for i in range(5):
        v = i + 1
        p = 0.2
        count = sum(pop2 == v)
        E_count = N * p
        var = N * p * (1-p)
        assert E_count - 2*var <= count <= E_count + 2*var
