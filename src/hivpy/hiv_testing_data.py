from hivpy.exceptions import DataLoadException

from .data_reader import DataReader


class HIVTestingData(DataReader):
    """
    Class to hold and interpret HIV testing data loaded from a yaml file.
    """

    def __init__(self, filename):
        super().__init__(filename)

        try:
            self.date_start_anc_testing = self.data["date_start_anc_testing"]
            self.date_start_testing = self.data["date_start_testing"]
            self.init_rate_first_test = self.data["init_rate_first_test"]
            self.eff_max_freq_testing = self.data["eff_max_freq_testing"]
            self.test_scenario = self.data["test_scenario"]
            self.no_test_if_np0 = self.data["no_test_if_np0"]

            self.test_targeting = self._get_discrete_dist("test_targeting")
            self.date_test_rate_plateau = self._get_discrete_dist("date_test_rate_plateau")
            self.an_lin_incr_test = self._get_discrete_dist("an_lin_incr_test")

        except KeyError as ke:
            print(ke.args)
            raise DataLoadException
