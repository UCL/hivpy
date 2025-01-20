from math import isclose

from hivpy.common import date, timedelta
from hivpy.population import Population


def test_matrix_val_retrieval():
    pop = Population(size=1, start_date=date(2000, 1, 1))
    res = pop.resistance
    max_viral_load = 4

    # get vl_matrix[0][0][0] -> max_viral_load
    assert res.get_matrix_val(res.get_viral_load_matrix(max_viral_load),
                              0, timedelta(months=0), 0, 0) == max_viral_load
    # get vl_matrix[5][1][2][1] -> max_viral_load - 1.05
    assert res.get_matrix_val(res.get_viral_load_matrix(max_viral_load),
                              1.25, timedelta(months=4), 1, 0.6) == max_viral_load - 1.05
    # get vl_matrix[9][1][2][2] -> 1.4
    assert res.get_matrix_val(res.get_viral_load_matrix(max_viral_load),
                              2.25, timedelta(months=5), 0.8, 1.2) == 1.4
    # get vl_matrix[12][2][2] -> min_vl_on_art
    assert res.get_matrix_val(res.get_viral_load_matrix(max_viral_load),
                              3, timedelta(months=6), 0.8, 0.8) == res.min_vl_on_art

    # get cd4_matrix[0][0][0] -> -18
    assert res.get_matrix_val(res.cd4_delta_matrix, 0, timedelta(months=0), 0, 0) == -18
    # get cd4_matrix[5][1][2][1] -> -7
    assert res.get_matrix_val(res.cd4_delta_matrix, 1.25, timedelta(months=4), 1, 0.6) == -7
    # get cd4_matrix[9][1][2][2] -> 23
    assert res.get_matrix_val(res.cd4_delta_matrix, 2.25, timedelta(months=5), 0.8, 1.2) == 23
    # get cd4_matrix[12][2][2] -> 30
    assert res.get_matrix_val(res.cd4_delta_matrix, 3, timedelta(months=6), 0.8, 0.8) == 30

    # get nm_matrix[0][0][0] -> 0.05
    assert res.get_matrix_val(res.new_mutation_matrix, 0, timedelta(months=0), 0, 0) == 0.05
    # get nm_matrix[5][0->1][2][1] -> 0.3
    assert res.get_matrix_val(res.new_mutation_matrix, 1.25, timedelta(months=2), 1, 0.6, on_nev=True) == 0.3
    # get nm_matrix[5][2][1] -> 0.35
    assert res.get_matrix_val(res.new_mutation_matrix, 1.25, timedelta(months=7), 0.6, 0.6) == 0.35
    # get nm_matrix[9][1][2][2] -> 0.05
    assert res.get_matrix_val(res.new_mutation_matrix, 2.25, timedelta(months=5), 0.8, 1.2) == 0.05
    # get nm_matrix[12][2][2] -> 0.002
    assert res.get_matrix_val(res.new_mutation_matrix, 3, timedelta(months=6), 0.8, 0.8) == 0.002

def test_calc_viral_load():
    pop = Population(size=1, start_date=date(2000, 1, 1))
    res = pop.resistance
    max_viral_load = 4

    # max_viral_load + vl_stdev_on_art * rng.normal()
    assert (max_viral_load - res.vl_stdev_on_art * 3 <=
            res.calc_viral_load(0, timedelta(months=0), 0, 0, max_viral_load)
            <= max_viral_load + res.vl_stdev_on_art * 3)
    # min_vl_on_art + vl_stdev_on_art * rng.normal()
    assert (res.min_vl_on_art - res.vl_stdev_on_art * 3 <=
            res.calc_viral_load(3, timedelta(months=6), 0.8, 0.8, max_viral_load)
            <= res.min_vl_on_art + res.vl_stdev_on_art * 3)

def test_calc_prob_new_mutation():
    pop = Population(size=1, start_date=date(2000, 1, 1))
    res = pop.resistance
    res.mutation_risk_change = 0.5

    # 0.05 * (50 + 20) / 2 * 0.5 = 0.875
    assert isclose(res.calc_prob_new_mutation(0, timedelta(months=0), 0, 0, False, False, 50, 20), 0.875)
    # 0.30 * (50 + 20) / 2 * 0.5 = 5.25 >> min(5.25, 1) = 1
    assert isclose(res.calc_prob_new_mutation(1.25, timedelta(months=2), 1, 0.6, False, True, 50, 20), 1)
    # 0.002 * (50 + 20) / 2 * 0.5 = 0.035
    assert isclose(res.calc_prob_new_mutation(3, timedelta(months=6), 0.8, 0.8, False, False, 50, 20), 0.035)
