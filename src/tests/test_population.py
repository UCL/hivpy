from datetime import date
import numpy as np 

from hivpy import column_names as col
from hivpy.population import Population
from hivpy.common import SexType
from hivpy.common import COND, AND, OR
import operator as op


def test_population_init():
    pop = Population(size=100, start_date=date(1989, 1, 1))
    print(pop)
    assert (len(pop.data) == 100)
    assert ((col.HIV_STATUS) in pop.data.columns)

def test_COND():
    pop = Population(size=1000, start_date=date(1989, 1, 1))
    pop.data[col.SEX] = np.array([SexType.Male, SexType.Female] * 500)
    males = pop.get_sub_pop(COND(col.SEX, op.eq, SexType.Male))
    females = pop.get_sub_pop(COND(col.SEX, op.ne, SexType.Male))  # check different operator
    assert (len(males) == 500)
    assert (len(females) == 500)
    assert (SexType.Female not in pop.get_variable(col.SEX, males))
    assert all(pop.get_variable(col.SEX, females) == SexType.Female)
    
    # test returning empty
    pop.data[col.SEX] = SexType.Male
    females = pop.get_sub_pop(COND(col.SEX, op.ne, SexType.Male))
    assert (len(females) == 0)

def test_AND():
    pop = Population(size=1000, start_date=date(1989, 1, 1))
    pop.data[col.SEX] = np.concatenate((np.array([SexType.Male] * 500), np.array([SexType.Female] * 500)))
    pop.data[col.AGE] = np.array([10,20,30,40] * 250)
    female_over_15 = pop.get_sub_pop(AND(COND(col.SEX, op.eq, SexType.Female),
                                         COND(col.AGE, op.ge, 15)))
    assert(len(female_over_15) == 375)
    assert all(pop.get_variable(col.AGE, female_over_15) >= 15)
    assert all(pop.get_variable(col.SEX, female_over_15) == SexType.Female)

    male_over_15_under_40 = pop.get_sub_pop(AND(COND(col.SEX, op.eq, SexType.Male),
                                                COND(col.AGE, op.ge, 15),
                                                COND(col.AGE, op.lt, 40)))
    assert(len(male_over_15_under_40) == 250)
    assert all(pop.get_variable(col.AGE, male_over_15_under_40) >= 15)
    assert all(pop.get_variable(col.AGE, male_over_15_under_40) < 40)
    assert all(pop.get_variable(col.SEX, male_over_15_under_40) == SexType.Male)

    # test AND with only one argument
    male_pop = pop.get_sub_pop(AND(COND(col.SEX, op.eq, SexType.Male)))
    assert(len(male_pop) == 500)
    assert all(pop.get_variable(col.SEX, male_pop) == SexType.Male)

def test_OR():
    pop = Population(size=1500, start_date=date(1989, 1, 1))
    pop.data[col.SEX] = np.concatenate((np.array([SexType.Male] * 750), np.array([SexType.Female] * 750)))
    pop.data[col.AGE] = np.array([10,20,30,40,60,80] * 250)
    women_or_children = pop.get_sub_pop(OR(COND(col.SEX, op.eq, SexType.Female),
                                           COND(col.AGE, op.lt, 18)))
    assert(len(women_or_children) == 875)
    assert all((pop.get_variable(col.AGE, women_or_children) <= 18) | (pop.get_variable(col.SEX, women_or_children) == SexType.Female))

    women_or_children_or_elderly = pop.get_sub_pop(OR(COND(col.SEX, op.eq, SexType.Female),
                                                      COND(col.AGE, op.lt, 18),
                                                      COND(col.AGE, op.ge, 65)))
    assert(len(women_or_children_or_elderly) == 1000)
    ages = pop.get_variable(col.AGE, women_or_children_or_elderly)
    sexes = pop.get_variable(col.SEX, women_or_children_or_elderly)
    assert all ((sexes == SexType.Female) | (ages < 18) | (ages >= 65) ) 

    # test OR with only one argument
    male_pop = pop.get_sub_pop(OR(COND(col.SEX, op.eq, SexType.Male)))
    assert(len(male_pop) == 750)
    assert all(pop.get_variable(col.SEX, male_pop) == SexType.Male)

def test_compound_expression():
    pop = Population(size=1000, start_date=date(1989, 1, 1))
    pop.data[col.SEX] = np.concatenate((np.array([SexType.Male] * 500), np.array([SexType.Female] * 500)))
    pop.data[col.AGE] = np.array([10,20,30,40] * 250)
    female_over_25_or_male_under_25 = pop.get_sub_pop(
        OR(
            AND(
                COND(col.SEX, op.eq, SexType.Female),
                COND(col.AGE, op.ge, 25)
            ),
            AND(
                COND(col.SEX, op.eq, SexType.Male),
                COND(col.AGE, op.lt, 25)
            )
        )
    )
    assert(len(female_over_25_or_male_under_25) == 500)
    ages = pop.get_variable(col.AGE, female_over_25_or_male_under_25)
    sexes = pop.get_variable(col.SEX, female_over_25_or_male_under_25)
    assert all( (ages < 25) | (sexes == SexType.Female) )
    assert all( (ages > 25) | (sexes == SexType.Male) )