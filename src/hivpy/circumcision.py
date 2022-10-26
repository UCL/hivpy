import importlib.resources

import hivpy.column_names as col

from .common import SexType, rng
from .circumcision_data import CircumcisionData


class CircumcisionModule:

    def __init__(self, **kwargs):
        # init cirumcision data
        with importlib.resources.path("hivpy.data", "circumcision.yaml") as data_path:
            self.c_data = CircumcisionData(data_path)
        self.prob_birth_circ = self.c_data.prob_birth_circ.sample()

    def initialise_circumcision(self, population):
        male_population = population.index[population[col.SEX] == SexType.Male]
        r = rng.uniform(size=len(male_population))
        circumcision = r < self.prob_birth_circ
        population.loc[male_population, col.CIRCUMCISED] = circumcision
