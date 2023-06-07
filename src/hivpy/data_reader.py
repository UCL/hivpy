from abc import ABC, abstractmethod

import numpy as np
import yaml

from .common import DiscreteChoice


class DataReader(ABC):
    """
    An abstract base data reader class to allow subclasses to
    hold and interpret data loaded from a given yaml file.
    """

    def _setup_probabilty_dist(self, prob_dict):
        if ("Range" in prob_dict):
            min = prob_dict["Range"][0]
            max = prob_dict["Range"][1]
            N = max - min + 1
            return DiscreteChoice(np.arange(min, max+1, 1), np.array([1./N]*N))
        else:
            return self._extract_discrete_dist(prob_dict)

    def _get_discrete_dist_list(self, *keys):
        dist_list = self.data
        for k in keys:
            dist_list = dist_list[k]
        return np.array([self._setup_probabilty_dist(x) for x in dist_list])

    def _get_discrete_dist(self, *keys):
        dist_data = self.data
        for k in keys:
            dist_data = dist_data[k]
        return self._extract_discrete_dist(dist_data)

    def _extract_discrete_dist(self, dist_data):
        vals = np.array(dist_data["Value"])
        if "Probability" in dist_data:
            probs = np.array(dist_data["Probability"])
        else:
            probs = np.ones(vals.size)
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

    @abstractmethod
    def __init__(self, filename):
        with open(filename, 'r') as file:
            self.data = yaml.safe_load(file)
