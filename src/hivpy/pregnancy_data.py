from hivpy.exceptions import DataLoadException

from .data_reader import DataReader


class PregnancyData(DataReader):
    """
    Class to hold and interpret pregnancy data loaded from a yaml file.
    """

    def __init__(self, filename):
        super().__init__(filename)

        try:
            self.can_be_pregnant = self.data["can_be_pregnant"]
            self.rate_want_no_children = self.data["rate_want_no_children"]
            self.date_pmtct = self.data["date_pmtct"]
            self.pmtct_inc_rate = self.data["pmtct_inc_rate"]
            self.fertility_factor = self.data["fertility_factor"]

            self.inc_cat = self._get_discrete_dist("inc_cat")
            self.rate_test_anc_inc = self._get_discrete_dist("rate_test_anc_inc")
            self.prob_birth_with_infected_child = self._get_discrete_dist("prob_birth_with_infected_child")
            self.max_children = self.data["max_children"]
            self.init_num_children_distributions = self._get_discrete_dist_list("init_num_children_distributions")

        except KeyError as ke:
            print(ke.args)
            raise DataLoadException
