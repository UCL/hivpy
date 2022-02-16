import numpy as np
import scipy.stats as stat
import yaml

from .common import SexType


class SexualBehaviourData:
    """Class to hold and interpret sexual behaviour data loaded from the yaml file"""

    def _get_discrete_dist_list(self, keys):
        dist_list = self.data
        for k in keys:
            dist_list = dist_list[k]
        return np.array([stat.rv_discrete(values=(x["Value"], x["Probs"])) for x in dist_list])

    def _get_discrete_dist(self, keys):
        dist_data = self.data
        for k in keys:
            dist_data = dist_data[k]
        values = np.array(dist_data["Value"])
        probs = np.array(dist_data["Probs"])
        probs /= sum(probs)
        return stat.rv_discrete(values=(values, probs))

    def _get_stepwise_dist(self, keys):
        dist_data = self.data
        for k in keys:
            dist_data = dist_data[k]

    def _norm_probs(self, prob_dict: dict):
        return {
            key: data / sum(data)
            for key, data in prob_dict.items()
        }

    def _select_matrix(self, matrix_list):
        return matrix_list[np.random.choice(len(matrix_list))]

    def __init__(self, filename):
        with open(filename, 'r') as file:
            self.data = yaml.safe_load(file)
        self.male_stp_dists = self._get_discrete_dist_list(
            ["short_term_partner_distributions", "Male"])
        self.female_stp_u25_dists = self._get_discrete_dist_list(
            ["short_term_partner_distributions", "Female", "Under_25"])
        self.female_stp_o25_dists = self._get_discrete_dist_list(
            ["short_term_partner_distributions", "Female", "Over_25"])
        self.baseline_risk = np.array(self._select_matrix(self.data["baseline_risk_options"]))
        self.sex_behaviour_transition_options = self.data["sex_behaviour_transition_options"]
        self.sex_mixing_matrix_male_options = self.data["sex_age_mixing_matrices"]["Male"]
        self.sex_mixing_matrix_female_options = self.data["sex_age_mixing_matrices"]["Female"]
        self.short_term_partners_male_options = self.data["sex_behaviour_transition_options"]["Male"]
        self.short_term_partners_female_options = self.data["sex_behaviour_transition_options"]["Female"]
        self.init_sex_behaviour_probs = {SexType.Male: self._get_discrete_dist(["initial_sex_behaviour_probabilities","Male"]),
                                         SexType.Female: self._get_discrete_dist(["initial_sex_behaviour_probabilities","Female"])}
