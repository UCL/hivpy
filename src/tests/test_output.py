from datetime import date, timedelta
from math import isclose, log

import hivpy.column_names as col
from hivpy.common import SexType
from hivpy.output import SimulationOutput
from hivpy.population import Population

# age boundaries
age_min = 15
age_max_active = 65
age_step = 10


def test_HIV_prevalence():

    # build population
    N = 1000
    pop = Population(size=N, start_date=date(1990, 1, 1))
    pop.data.loc[:int(N/2)-1, col.SEX] = SexType.Female
    pop.data.loc[int(N/2):, col.SEX] = SexType.Male
    pop.data[col.AGE] = 25
    pop.data[col.SEX_WORKER] = False
    pop.data.loc[:int(N/4)-1, col.SEX_WORKER] = True
    pop.data.loc[:int(N/4)-1, col.HIV_STATUS] = True

    out = SimulationOutput(date(1990, 1, 1), date(1990, 3, 1), timedelta(days=90))
    out._update_HIV_prevalence(pop)

    # a quarter of all people have HIV
    assert isclose(out.output_stats["HIV prevalence (tot)"], 0.25)
    # all are in the 25-35 age bracket
    assert isclose(out.output_stats["HIV prevalence (25-34)"], 0.25)
    # all HIV positive people are women
    assert isclose(out.output_stats["HIV prevalence (female)"], 0.5)
    assert isclose(out.output_stats["HIV prevalence (male)"], 0)
    # all HIV positive people are sex workers
    assert isclose(out.output_stats["HIV prevalence (sex worker)"], 1)


def test_HIV_incidence():

    # build population
    N = 1000
    pop = Population(size=N, start_date=date(1990, 1, 1))
    pop.data[col.SEX] = SexType.Female
    pop.data.loc[:int(N*0.2)-1, col.AGE] = 20
    pop.data.loc[int(N*0.2):int(N*0.4)-1, col.AGE] = 30
    pop.data.loc[int(N*0.4):int(N*0.6)-1, col.AGE] = 40
    pop.data.loc[int(N*0.6):int(N*0.8)-1, col.AGE] = 50
    pop.data.loc[int(N*0.8):, col.AGE] = 60

    pop.data[col.HIV_STATUS] = False
    pop.data[col.IN_PRIMARY_INFECTION] = False
    pop.data.loc[:int(N*0.2)-1, col.IN_PRIMARY_INFECTION] = True
    pop.data.loc[int(N*0.2):int(N*0.35)-1, col.IN_PRIMARY_INFECTION] = True
    pop.data.loc[int(N*0.4):int(N*0.5)-1, col.IN_PRIMARY_INFECTION] = True
    pop.data.loc[int(N*0.6):int(N*0.65)-1, col.IN_PRIMARY_INFECTION] = True

    out = SimulationOutput(date(1990, 1, 1), date(1990, 3, 1), timedelta(days=90))
    out._update_HIV_incidence(pop)

    # get age stats
    for age_bound in range(age_min, age_max_active, age_step):
        age_group = int(age_bound/10)-1
        key = f"HIV incidence ({age_bound}-{age_bound+(age_step-1)}, female)"
        assert isclose(out.output_stats[key], 1-age_group*0.25)


def test_partner_sex_balance():

    # build population
    N = 100
    pop = Population(size=N, start_date=date(1990, 1, 1))
    pop.data.loc[:int(N/2)-1, col.SEX] = SexType.Female
    pop.data.loc[int(N/2):, col.SEX] = SexType.Male
    pop.data[col.AGE] = 25
    pop.data[col.NUM_PARTNERS] = 2
    pop.data[col.STP_AGE_GROUPS] = [[1, 2]]*N

    out = SimulationOutput(date(1990, 1, 1), date(1990, 3, 1), timedelta(days=90))
    out._update_partner_sex_balance(pop)

    # check overall sex balance is equal
    assert isclose(out.output_stats["Partner sex balance (male)"], 0)
    assert isclose(out.output_stats["Partner sex balance (female)"], 0)
    # check age-specific ratio is 2
    assert isclose(out.output_stats["Partner sex balance (25-34, male)"], log(2, 10))
    assert isclose(out.output_stats["Partner sex balance (25-34, female)"], log(2, 10))
