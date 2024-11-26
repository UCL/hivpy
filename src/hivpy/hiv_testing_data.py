from hivpy.exceptions import DataLoadException

from .data_reader import DataReader


class HIVTestingData(DataReader):
    """
    Class to hold and interpret HIV testing data loaded from a yaml file.
    """

    def __init__(self, filename):
        super().__init__(filename)

        try:
            self.date_start_testing = self.data["date_start_testing"]
            self.date_rate_testing_incr = self.data["date_rate_testing_incr"]
            self.init_rate_first_test = self.data["init_rate_first_test"]
            self.eff_max_freq_testing = self.data["eff_max_freq_testing"]
            self.test_scenario = self.data["test_scenario"]
            self.no_test_if_np0 = self.data["no_test_if_np0"]

            self.prob_anc_test_trim1 = self.data["prob_anc_test_trim1"]
            self.prob_anc_test_trim2 = self._get_discrete_dist("prob_anc_test_trim2")
            self.prob_anc_test_trim3 = self.data["prob_anc_test_trim3"]
            self.prob_test_postdel = self.data["prob_test_postdel"]

            self.prob_test_non_hiv_symptoms = self.data["prob_test_non_hiv_symptoms"]
            self.prob_test_who4 = self.data["prob_test_who4"]
            self.prob_test_tb = self.data["prob_test_tb"]
            self.prob_test_non_tb_who3 = self.data["prob_test_non_tb_who3"]
            self.test_targeting = self._get_discrete_dist("test_targeting")
            self.date_general_testing_plateau = self._get_discrete_dist("date_general_testing_plateau")
            self.date_targeted_testing_plateau = self.data["date_targeted_testing_plateau"]
            self.an_lin_incr_test = self._get_discrete_dist("an_lin_incr_test")
            self.incr_test_rate_sympt = self._get_discrete_dist("incr_test_rate_sympt")

        except KeyError as ke:
            print(ke.args)
            raise DataLoadException
