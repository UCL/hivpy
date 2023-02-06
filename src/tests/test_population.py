from datetime import date

from hivpy import column_names as col
from hivpy.population import Population


def test_population_init():
    pop = Population(size=100, start_date=date(1989, 1, 1))
    print(pop)
    assert (len(pop.data) == 100)
    assert ((col.HIV_STATUS) in pop.data.columns)
