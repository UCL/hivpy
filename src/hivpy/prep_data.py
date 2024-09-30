from hivpy.exceptions import DataLoadException

from .data_reader import DataReader


class PrEPData(DataReader):
    """
    Class to hold and interpret PrEP data loaded from a yaml file.
    """

    def __init__(self, filename):
        super().__init__(filename)

        try:
            self.prep_strategy = self._get_discrete_dist("prep_strategy")
            self.date_prep_oral_intro = self.data["date_prep_oral_intro"]
            self.date_prep_cab_intro = self.data["date_prep_cab_intro"]
            self.date_prep_len_intro = self.data["date_prep_len_intro"]
            self.date_prep_vr_intro = self.data["date_prep_vr_intro"]
            self.prob_risk_informed_prep = self.data["prob_risk_informed_prep"]
            self.prob_greater_risk_informed_prep = self.data["prob_greater_risk_informed_prep"]
            self.prob_suspect_risk_prep = self.data["prob_suspect_risk_prep"]

            self.rate_test_onprep_any = self.data["rate_test_onprep_any"]
            self.prep_willing_threshold = self.data["prep_willing_threshold"]
            self.prob_test_prep_start = self._get_discrete_dist("prob_test_prep_start")
            self.prob_prep_restart = self._get_discrete_dist("prob_prep_restart")

        except KeyError as ke:
            print(ke.args)
            raise DataLoadException
