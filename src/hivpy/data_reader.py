from abc import ABC, abstractmethod

import numpy as np
import yaml

from .common import BetaDistribution, DiscreteChoice


class DataReader(ABC):
    """
    An abstract base data reader class to allow subclasses to
    hold and interpret data loaded from a given yaml file.
    """

    def _setup_probabilty_dist(self, prob_dict):
        if "Range" in prob_dict:
            min = prob_dict["Range"][0]
            max = prob_dict["Range"][1]
            N = max - min + 1
            return DiscreteChoice(np.arange(min, max + 1, 1), np.array([1.0 / N] * N))
        else:
            # count number of params (not including Probability)
            n_params = 0
            for k in prob_dict:
                n_params = n_params + 1 if k != "Probability" else n_params
            if n_params == 1:
                return self._extract_discrete_dist(prob_dict)
            else:
                return self._extract_multi_distribution(prob_dict)

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

    def _extract_multi_distribution(self, dist_data):
        length = -1
        for k in dist_data:
            if length == -1:
                length = len(dist_data[k])
            else:
                assert length == len(dist_data[k])
        vals = []
        for i in range(length):
            valmap = {}
            for k in dist_data:
                if k == "Probability":
                    pass
                else:
                    valmap[k] = dist_data[k][i]
            vals.append(valmap)
        if "Probability" in dist_data:
            probs = np.array(dist_data["Probability"], dtype=float)
        else:
            probs = np.ones(size=length)
        probs /= sum(probs)
        return DiscreteChoice(vals, probs)

    def _extract_discrete_dist(self, dist_data):
        vals = np.array(dist_data["Value"])
        if "Probability" in dist_data:
            probs = np.array(dist_data["Probability"], dtype=float)
        else:
            probs = np.ones(vals.size, dtype=float)
        probs /= sum(probs)
        return DiscreteChoice(vals, probs)

    def _get_stepwise_dist(self, keys):
        dist_data = self.data
        for k in keys:
            dist_data = dist_data[k]

    def _norm_probs(self, prob_dict: dict):
        return {key: data / sum(data) for key, data in prob_dict.items()}

    def _get_beta_distribution(self, *keys):
        dist_data = self.data
        for k in keys:
            dist_data = dist_data[k]
        alpha = dist_data["alpha"]
        beta = dist_data["beta"]
        return BetaDistribution(alpha, beta)

    @abstractmethod
    def __init__(self, filename):
        with open(filename, "r") as file:
            self.data = yaml.safe_load(file)
