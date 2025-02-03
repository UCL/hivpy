from math import isclose, sqrt

import hivpy.column_names as col
from hivpy.common import SexType, date, timedelta
from hivpy.population import Population


def test_matrix_value_retrieval():
    time_step = timedelta(months=1)
    pop = Population(size=1, start_date=date(2000, 1, 1))
    pop.set_present_variable(col.ART_ADHERENCE, 0)
    pop.date += time_step
    pop.step += 1
    pop.set_present_variable(col.NUM_ACTIVE_DRUGS, 0)
    pop.set_present_variable(col.CONT_ON_ART, timedelta(months=0))
    pop.set_present_variable(col.ART_ADHERENCE, 0)
    pop.set_present_variable(col.ON_NEV, False)
    pop.set_present_variable(col.ON_EFA, False)

    res = pop.resistance
    res.active_drug_indices, res.cont_on_art_indices, \
        res.adherence_indices, res.adherence_tm1_indices = res.get_all_matrix_indices(pop, pop.data.index)

    # get vl_matrix[0][0][0] -> (1, 0, 0) -> max_viral_load
    assert res.get_matrix_value(res.viral_load_matrix, 0) == (1, 0, 0)
    # get cd4_matrix[0][0][0] -> -18
    assert res.get_matrix_value(res.cd4_delta_matrix, 0) == -18
    # get nm_matrix[0][0][0] -> 0.05
    assert res.get_matrix_value(res.new_mutation_matrix, 0, on_nev=pop.data.loc[0, col.ON_NEV],
                                on_efa=pop.data.loc[0, col.ON_EFA]) == 0.05

    pop.set_present_variable(col.NUM_ACTIVE_DRUGS, 1.25)
    pop.set_present_variable(col.CONT_ON_ART, timedelta(months=4))
    pop.data[pop.get_correct_column(col.ART_ADHERENCE, dt=0)] = 0.2
    pop.data[pop.get_correct_column(col.ART_ADHERENCE, dt=1)] = 0.6
    pop.set_present_variable(col.ON_NEV, True)
    res.active_drug_indices, res.cont_on_art_indices, \
        res.adherence_indices, res.adherence_tm1_indices = res.get_all_matrix_indices(pop, pop.data.index)

    # get vl_matrix[5][1][0][1] -> (1, -0.05, 0) -> max_viral_load - 0.05
    assert res.get_matrix_value(res.viral_load_matrix, 0) == (1, -0.05, 0)
    # get cd4_matrix[5][1][0][1] -> -17.5
    assert res.get_matrix_value(res.cd4_delta_matrix, 0) == -17.5
    # get nm_matrix[5][1][0->1][1] -> 0.35
    assert res.get_matrix_value(res.new_mutation_matrix, 0, on_nev=pop.data.loc[0, col.ON_NEV],
                                on_efa=pop.data.loc[0, col.ON_EFA]) == 0.35

    pop.set_present_variable(col.NUM_ACTIVE_DRUGS, 1.25)
    pop.set_present_variable(col.CONT_ON_ART, timedelta(months=7))
    pop.data[pop.get_correct_column(col.ART_ADHERENCE, dt=0)] = 1
    res.active_drug_indices, res.cont_on_art_indices, \
        res.adherence_indices, res.adherence_tm1_indices = res.get_all_matrix_indices(pop, pop.data.index)

    # get nm_matrix[5][2][2] -> 0.3
    assert res.get_matrix_value(res.new_mutation_matrix, 0, on_nev=pop.data.loc[0, col.ON_NEV],
                                on_efa=pop.data.loc[0, col.ON_EFA]) == 0.3

    pop.set_present_variable(col.NUM_ACTIVE_DRUGS, 2.25)
    pop.set_present_variable(col.CONT_ON_ART, timedelta(months=5))
    pop.data[pop.get_correct_column(col.ART_ADHERENCE, dt=0)] = 0.8
    pop.data[pop.get_correct_column(col.ART_ADHERENCE, dt=1)] = 1.2
    res.active_drug_indices, res.cont_on_art_indices, \
        res.adherence_indices, res.adherence_tm1_indices = res.get_all_matrix_indices(pop, pop.data.index)

    # get vl_matrix[9][1][2][2] -> (0, 1.4, 0) -> 1.4
    assert res.get_matrix_value(res.viral_load_matrix, 0) == (0, 1.4, 0)
    # get cd4_matrix[9][1][2][2] -> 23
    assert res.get_matrix_value(res.cd4_delta_matrix, 0) == 23
    # get nm_matrix[9][1][2][2] -> 0.05
    assert res.get_matrix_value(res.new_mutation_matrix, 0, on_nev=pop.data.loc[0, col.ON_NEV],
                                on_efa=pop.data.loc[0, col.ON_EFA]) == 0.05

    pop.set_present_variable(col.NUM_ACTIVE_DRUGS, 3)
    pop.set_present_variable(col.CONT_ON_ART, timedelta(months=6))
    pop.data[pop.get_correct_column(col.ART_ADHERENCE, dt=0)] = 0.8
    pop.data[pop.get_correct_column(col.ART_ADHERENCE, dt=1)] = 0.8
    res.active_drug_indices, res.cont_on_art_indices, \
        res.adherence_indices, res.adherence_tm1_indices = res.get_all_matrix_indices(pop, pop.data.index)

    # get vl_matrix[12][2][2] -> (0, 0, 1) -> min_vl_on_art
    assert res.get_matrix_value(res.viral_load_matrix, 0) == (0, 0, 1)
    # get cd4_matrix[12][2][2] -> 30
    assert res.get_matrix_value(res.cd4_delta_matrix, 0) == 30
    # get nm_matrix[12][2][2] -> 0.002
    assert res.get_matrix_value(res.new_mutation_matrix, 0, on_nev=pop.data.loc[0, col.ON_NEV],
                                on_efa=pop.data.loc[0, col.ON_EFA]) == 0.002


def test_calc_viral_load():
    time_step = timedelta(months=1)
    pop = Population(size=1, start_date=date(2000, 1, 1))
    pop.set_present_variable(col.ART_ADHERENCE, 0)
    pop.date += time_step
    pop.step += 1
    pop.set_present_variable(col.NUM_ACTIVE_DRUGS, 0)
    pop.set_present_variable(col.CONT_ON_ART, timedelta(months=0))
    pop.set_present_variable(col.ART_ADHERENCE, 0)
    max_viral_load = 4
    pop.set_present_variable(col.MAX_VIRAL_LOAD, max_viral_load)

    res = pop.resistance
    res.active_drug_indices, res.cont_on_art_indices, \
        res.adherence_indices, res.adherence_tm1_indices = res.get_all_matrix_indices(pop, pop.data.index)

    # max_viral_load + vl_stdev_on_art * rng.normal()
    assert (max_viral_load - res.vl_stdev_on_art * 3 <=
            res.calc_viral_load(pop.data.loc[0])
            <= max_viral_load + res.vl_stdev_on_art * 3)

    pop.set_present_variable(col.NUM_ACTIVE_DRUGS, 3)
    pop.set_present_variable(col.CONT_ON_ART, timedelta(months=6))
    pop.data[pop.get_correct_column(col.ART_ADHERENCE, dt=0)] = 0.8
    pop.data[pop.get_correct_column(col.ART_ADHERENCE, dt=1)] = 0.8
    res.active_drug_indices, res.cont_on_art_indices, \
        res.adherence_indices, res.adherence_tm1_indices = res.get_all_matrix_indices(pop, pop.data.index)

    # min_vl_on_art + vl_stdev_on_art * rng.normal()
    assert (res.min_vl_on_art - res.vl_stdev_on_art * 3 <=
            res.calc_viral_load(pop.data.loc[0])
            <= res.min_vl_on_art + res.vl_stdev_on_art * 3)


def test_calc_cd4_delta():
    pop = Population(size=1, start_date=date(2000, 1, 1))
    res = pop.resistance
    res.hindered_cd4_recovery = -3

    # check basic case at age 20
    # 6 + 0.1 * -18 = 4.2
    # 50 + 4.2 = 54.2
    cd4, delta = res.calc_cd4_delta(20, SexType.Male, 0, timedelta(months=0), 0, 0,
                                    False, False, False, False, False, False, 50, 0.1, 100, False, False)
    assert isclose(cd4, 54.2)
    assert isclose(delta, 4.2)
    # 6 + 0.1 * 30 = 9
    # 50 + 9 = 59
    cd4, delta = res.calc_cd4_delta(20, SexType.Male, 3, timedelta(months=6), 0.8, 0.8,
                                    False, False, False, False, False, False, 50, 0.1, 100, False, False)
    assert isclose(cd4, 59)
    assert isclose(delta, 9)

    # check basic case with female recovery factor
    # 6 + 2 + 0.1 * 30 = 11
    # 50 + 11 = 61
    cd4, delta = res.calc_cd4_delta(20, SexType.Female, 3, timedelta(months=6), 0.8, 0.8,
                                    False, False, False, False, False, False, 50, 0.1, 100, False, False)
    assert isclose(cd4, 61)
    assert isclose(delta, 11)

    # check hindered cd4 recovery
    # 3 + 0.1 * -18 = 1.2
    # 50 + 1.2 = 51.2
    cd4, delta = res.calc_cd4_delta(20, SexType.Male, 0, timedelta(months=0), 0, 0,
                                    True, False, False, False, False, False, 50, 0.1, 100, False, False)
    assert isclose(cd4, 51.2)
    assert isclose(delta, 1.2)

    # check improved cd4 recovery with pi recovery factor
    # 9 + 0.1 * 30 = 12
    # 50 + 12 = 62
    cd4, delta = res.calc_cd4_delta(20, SexType.Male, 3, timedelta(months=6), 0.8, 0.8,
                                    False, False, False, False, False, True, 50, 0.1, 100, False, False)
    assert isclose(cd4, 62)
    assert isclose(delta, 12)

    # check adjustments on ARV
    # 6 + 0.2 * 6 = 12 >> 12 * 0.85 = 10.2
    # 150 + 10.2 = 160.2 >> sqrt(160.2) + cd4_stdev_on_art * rng.normal() ** 2
    cd4, delta = res.calc_cd4_delta(20, SexType.Male, 3, timedelta(months=6), 0.8, 0.8,
                                    False, False, False, False, False, False, 150, 0.2, 200, True, False)
    assert sqrt(160.2) - res.cd4_stdev_on_art * 3 <= cd4 <= sqrt(160.2) + res.cd4_stdev_on_art * 3
    assert isclose(delta, 10.2)

    # check max cd4 cap on ARV
    # 6 + 0.2 * 6 = 12 >> 12 * 0.7 = 8.4
    # 100 + rng.normal() * 50
    cd4, delta = res.calc_cd4_delta(20, SexType.Male, 3, timedelta(months=6), 0.8, 0.8,
                                    False, False, False, False, False, False, 10000, 0.2, 100, False, True)
    assert 100 - 50 * 3 <= cd4 <= 100 + 50 * 3
    assert isclose(delta, 8.4)


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
