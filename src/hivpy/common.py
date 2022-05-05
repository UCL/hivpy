"""Functionality shared between multiple parts of the framework."""

import operator
from enum import IntEnum
from functools import reduce

import numpy as np
import scipy.stats as stat


class DiscreteChoice:
    def __init__(self, vals, probs):
        N = len(vals)
        if(len(probs) != N):
            raise Exception
        index_range = np.arange(0, N, 1)
        self.data = vals
        self.dist = stat.rv_discrete(values=(index_range, probs))

    def sample(self, size=1):
        indices = self.dist.rvs(size=size)
        return self.data[indices]


class SexType(IntEnum):
    Male = 0
    Female = 1


def selector(population, **kwargs):
    """Select the rows of a population data frame matching a set of criteria.

    FIXME Describe usage in more detail.
    """
    index = reduce(operator.and_,
                   (op(population[kw], val) for kw, (op, val) in kwargs.items()))
    return index


def between(values, limits):
    """A helper function for selecting values within a range."""
    min_value, max_value = limits
    # def _is_in_range(values):
    return (min_value <= values) & (values < max_value)
    # return _is_in_range
