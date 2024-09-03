from hivpy.exceptions import DataLoadException

from .data_reader import DataReader


class HIVDiagnosisData(DataReader):
    """
    Class to hold and interpret HIV diagnosis data loaded from a yaml file.
    """

    def __init__(self, filename):
        super().__init__(filename)

        try:
            self.test_sens_general = self.data["test_sens_general"]
            self.test_sens_primary_ab = self._get_discrete_dist("test_sens_primary_ab")
            self.test_sens_prep_inj_primary_ab = self._get_discrete_dist("test_sens_prep_inj_primary_ab")
            self.test_sens_prep_inj_3m_ab = self._get_discrete_dist("test_sens_prep_inj_3m_ab")
            self.test_sens_prep_inj_ge6m_ab = self._get_discrete_dist("test_sens_prep_inj_ge6m_ab")
            self.tests_sens_prep_inj = self._get_discrete_dist("tests_sens_prep_inj")
            self.test_sens_prep_inj_primary_na = self.data["test_sens_prep_inj_primary_na"]
            self.test_sens_prep_inj_3m_na = self.data["test_sens_prep_inj_3m_na"]
            self.test_sens_prep_inj_ge6m_na = self.data["test_sens_prep_inj_ge6m_na"]
            self.prob_loss_at_diag = self._get_discrete_dist("prob_loss_at_diag")

        except KeyError as ke:
            print(ke.args)
            raise DataLoadException
