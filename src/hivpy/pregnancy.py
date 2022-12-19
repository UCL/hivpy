import importlib.resources

from .pregnancy_data import PregnancyData


class PregnancyModule:

    def __init__(self, **kwargs):

        # init pregnancy data
        with importlib.resources.path("hivpy.data", "pregnancy.yaml") as data_path:
            self.c_data = PregnancyData(data_path)

        self.can_be_pregnant = self.c_data.can_be_pregnant
        self.inc_cat = self.c_data.inc_cat.sample()
        self.rate_birth_with_infected_child = self.c_data.rate_birth_with_infected_child.sample()
