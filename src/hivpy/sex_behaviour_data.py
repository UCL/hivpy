import numpy as np
import yaml

from hivpy.exceptions import DataLoadException

from .common import DiscreteChoice, SexType, rng


class SexualBehaviourData:
    """Class to hold and interpret sexual behaviour data loaded from the yaml file"""

    def _setup_probabilty_dist(self, prob_dict):
        if ("Range" in prob_dict):
            min = prob_dict["Range"][0]
            max = prob_dict["Range"][1]
            N = max - min + 1
            return DiscreteChoice(np.arange(min, max+1, 1), np.array([1./N]*N))
        else:
            values = np.array(prob_dict["Value"])
            probs = np.array(prob_dict["Probability"])
            probs /= sum(probs)
            return DiscreteChoice(values, probs)

    def _get_discrete_dist_list(self, *keys):
        dist_list = self.data
        for k in keys:
            dist_list = dist_list[k]
        return np.array([self._setup_probabilty_dist(x) for x in dist_list])

    def _get_discrete_dist(self, *keys):
        dist_data = self.data
        for k in keys:
            dist_data = dist_data[k]
        vals = np.array(dist_data["Value"])
        probs = np.array(dist_data["Probability"])
        probs /= sum(probs)
        return DiscreteChoice(vals, probs)

    def _get_stepwise_dist(self, keys):
        dist_data = self.data
        for k in keys:
            dist_data = dist_data[k]

    def _norm_probs(self, prob_dict: dict):
        return {
            key: data / sum(data)
            for key, data in prob_dict.items()
        }

    def __init__(self, filename):
        with open(filename, 'r') as file:
            self.data = yaml.safe_load(file)
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

            self.short_term_partners_male_options = self.data[
                "sex_behaviour_transition_options"]["Male"]

            self.short_term_partners_female_options = self.data[
                "sex_behaviour_transition_options"]["Female"]

            self.init_sex_behaviour_probs = {
                SexType.Male: self._get_discrete_dist(
                    "initial_sex_behaviour_probabilities", "Male"),
                SexType.Female: self._get_discrete_dist(
                    "initial_sex_behaviour_probabilities", "Female")}

            # risk reduction factors
            self.rred_initial = self.data["rred_initial"]

            self.age_based_risk = np.array(
                rng.choice(self.data["age_based_risk_options"]["risk_factor"],
                           p=self.data["age_based_risk_options"]["Probability"]))

            self.new_partner_dist = self._get_discrete_dist("new_partner_factor")

            self.rred_long_term_partnered = self._get_discrete_dist("rred_partnered")

            self.p_rred_p_dist = self._get_discrete_dist("population_rred_personal")

            self.rred_diagnosis = self._get_discrete_dist("rred_diagnosis")
            self.rred_diagnosis_period = self.data["rred_diagnosis"]["Period"]

            self.yearly_risk_change = {"1990s": self._get_discrete_dist("yearly_risk_change_90s"),
                                       "2010s": self._get_discrete_dist("yearly_risk_change_10s")}

            self.rred_art_adherence = self.data["rred_art_adherence"]["Value"]
            self.adherence_threshold = self.data["rred_art_adherence"]["Adherence_Threshold"]
            self.rred_art_adherence_probability = self.data["rred_art_adherence"]["Probability"]

        except KeyError as ke:
            print(ke.args)
            raise DataLoadException
