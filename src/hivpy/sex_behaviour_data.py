import numpy as np

from hivpy.exceptions import DataLoadException

from .common import SexType, rng
from .data_reader import DataReader


class SexualBehaviourData(DataReader):
    """
    Class to hold and interpret sexual behaviour data loaded from a yaml file.
    """

    def __init__(self, filename):
        super().__init__(filename)

        try:
            self.male_stp_dists = self._get_discrete_dist_list(
                "short_term_partner_distributions", "Male")

            self.female_stp_u25_dists = self._get_discrete_dist_list(
                "short_term_partner_distributions", "Female", "Under_25")

            self.female_stp_o25_dists = self._get_discrete_dist_list(
                "short_term_partner_distributions", "Female", "Over_25")

            self.sexworker_stp_dists = self._get_discrete_dist_list(
                "short_term_partner_distributions", "Sex_Worker")

            self.sex_behaviour_transition_options = self.data["sex_behaviour_transition_options"]

            self.sex_mixing_matrix_male_options = self.data["sex_age_mixing_matrices"]["Male"]

            self.sex_mixing_matrix_female_options = self.data["sex_age_mixing_matrices"]["Female"]

            self.sex_behaviour_male_options = self.data[
                "sex_behaviour_transition_options"]["Male"]

            self.sex_behaviour_female_options = self.data[
                "sex_behaviour_transition_options"]["Female"]

            self.sex_behaviour_sex_worker_options = self.data[
                "sex_behaviour_transition_options"]["Sex_Worker"]

            self.init_sex_behaviour_probs = {
                SexType.Male: self._get_discrete_dist(
                    "initial_sex_behaviour_probabilities", "Male"),
                SexType.Female: self._get_discrete_dist(
                    "initial_sex_behaviour_probabilities", "Female")}

            # risk reduction factors
            self.risk_initial = self.data["risk_initial"]

            self.age_based_risk = np.array(
                rng.choice(self.data["age_based_risk_options"]["risk_factor"],
                           p=self.data["age_based_risk_options"]["Probability"]))

            self.new_partner_dist = self._get_discrete_dist("new_partner_factor")

            self.risk_long_term_partnered = self._get_discrete_dist("risk_partnered")

            self.p_risk_p_dist = self._get_discrete_dist("population_risk_personal")

            self.risk_diagnosis = self._get_discrete_dist("risk_diagnosis")
            self.risk_diagnosis_period = self.data["risk_diagnosis"]["Period"]

            self.yearly_risk_change = {"1990s": self._get_discrete_dist("yearly_risk_change_90s"),
                                       "2010s": self._get_discrete_dist("yearly_risk_change_10s")}

            self.risk_art_adherence = self.data["risk_art_adherence"]["Value"]
            self.adherence_threshold = self.data["risk_art_adherence"]["Adherence_Threshold"]
            self.risk_art_adherence_probability = self.data["risk_art_adherence"]["Probability"]

            self.base_start_sex_work = self._get_discrete_dist("base_rate_start_sex_work")
            self.base_stop_sex_work = self._get_discrete_dist("base_rate_stop_sex_work")
            self.risk_sex_worker_age = self.data["risk_sex_worker_age"]
            self.prob_init_sex_work_age = self.data["prob_init_sex_work_age"]
            self.prob_sw_program_effect = self._get_discrete_dist("prob_sw_program_effect")
            self.incr_rate_sw_high_sex_risk = self.data["incr_rate_sw_high_sex_risk"]
            self.probability_high_sexual_risk = self._get_discrete_dist("probability_high_sexual_risk")

        except KeyError as ke:
            print(ke.args)
            raise DataLoadException
