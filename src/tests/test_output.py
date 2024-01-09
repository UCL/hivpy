from datetime import date, timedelta
from math import isclose, log

import hivpy.column_names as col
from hivpy.common import SexType
from hivpy.output import SimulationOutput
from hivpy.population import Population

# age boundaries
age_min = 15
age_max_active = 65
age_max = 100
age_step = 10


def test_partner_sex_balance():

    # build population
    N = 100
    pop = Population(size=N, start_date=date(1990, 1, 1))
    pop.data.loc[:int(N/2), col.SEX] = SexType.Female
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
