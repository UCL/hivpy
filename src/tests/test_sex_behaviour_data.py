from hivpy.sex_behaviour_data import SexualBehaviourData


def test_probability_loading():
    # Load example file
    SBD = SexualBehaviourData("src/tests/test_data/sbd_testing.yaml")
    dists = SBD._get_discrete_dist_list(["Test_Example_List"])

    # explicit distribution
    d0 = dists[0]
    N = 10000
    pop0 = d0.rvs(size=N)
    for i in range(5):
        v = i + 1
        p = 0.1*i
        count = sum(pop0 == v)
        E_count = p * N
        var = N * p * (1-p)
        assert E_count - 2*var <= count <= E_count + 2*var

    # uniform over range
    d1 = dists[1]
    pop1 = d1.rvs(size=N)
    for i in range(1, 11):
        count = sum(pop1 == i)
        E_count = 0.1 * N
        var = N * 0.09
        assert E_count - 2*var <= count <= E_count + 2*var
