import numpy as np
import yaml

from hivpy.exceptions import DataLoadException

from .common import DiscreteChoice


class CircumcisionData:
    """
    Class to hold and interpret circumcision data loaded from the yaml file.
    """

    # TODO: This is ripped directly from sex_behaviour_data.py,
    # we should make a new data reader module to store functions like this.
    def _get_discrete_dist(self, *keys):
        dist_data = self.data
        for k in keys:
            dist_data = dist_data[k]
        vals = np.array(dist_data["Value"])
        probs = np.array(dist_data["Probability"])
        probs /= sum(probs)
        return DiscreteChoice(vals, probs)

    def __init__(self, filename):
        with open(filename, 'r') as file:
            self.data = yaml.safe_load(file)
        try:
            self.vmmc_start_year = self.data["vmmc_start_year"]
            self.circ_rate_change_year = self.data["circ_rate_change_year"]
            self.prob_circ_calc_cutoff_year = self.data["prob_circ_calc_cutoff_year"]
            self.policy_intervention_year = self.data["policy_intervention_year"]
            self.circ_after_test = self.data["circ_after_test"]
            self.prob_circ_after_test = self.data["prob_circ_after_test"]
            self.covid_disrup_affected = self.data["covid_disrup_affected"]
            self.vmmc_disrup_covid = self.data["vmmc_disrup_covid"]
            self.circ_policy_scenario = self.data["circ_policy_scenario"]
            self.circ_increase_rate = self._get_discrete_dist("circ_increase_rate")
            self.circ_rate_change_post_2013 = self._get_discrete_dist("circ_rate_change_post_2013")
            self.circ_rate_change_15_19 = self._get_discrete_dist("circ_rate_change_15_19")
            self.circ_rate_change_20_29 = self._get_discrete_dist("circ_rate_change_20_29")
            self.circ_rate_change_30_49 = self._get_discrete_dist("circ_rate_change_30_49")
            self.prob_birth_circ = self._get_discrete_dist("prob_birth_circ")
        except KeyError as ke:
            print(ke.args)
            raise DataLoadException
