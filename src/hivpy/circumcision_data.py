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
            self.vmmc_start_date = self.data["mc_int"]
            self.year_interv = self.data["year_interv"]
            self.test_link_circ = self.data["test_link_circ"]
            self.test_link_circ_prob = self.data["test_link_circ_prob"]
            self.covid_disrup_affected = self.data["covid_disrup_affected"]
            self.vmmc_disrup_covid = self.data["vmmc_disrup_covid"]
            self.circumcision_increase_scenario = self.data["circ_inc_rate_year_i"]
            self.circ_increase_rate = self._get_discrete_dist("circ_inc_rate")
            self.circ_rate_change_post_2013 = self._get_discrete_dist("rel_incr_circ_post_2013")
            self.circ_rate_change_15_19 = self._get_discrete_dist("circ_inc_15_19")
            self.circ_rate_change_20_30 = self._get_discrete_dist("circ_red_20_30")
            self.circ_rate_change_30_50 = self._get_discrete_dist("circ_red_30_50")
            self.prob_birth_circ = self._get_discrete_dist("prob_birth_circ")
        except KeyError as ke:
            print(ke.args)
            raise DataLoadException
