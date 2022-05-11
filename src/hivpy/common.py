"""Functionality shared between multiple parts of the framework."""

import operator
from contextlib import contextmanager
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


class ResettableRandomState:
    def __init__(self):
        self.rng = np.random.default_rng()

    def __getattr__(self, name):
        """Delegate method calls and attribute lookups to the Generator."""
        return getattr(self.rng, name)

    def set_seed(self, seed):
        """Set the seed that controls the sequence of values generated.

        Generating samples from the same seed should always return the same
        results.
        """
        self.rng = np.random.default_rng(seed)

    @contextmanager
    def set_temp_seed(self, temp_seed):
        """Allow setting the seed temporarily and restoring it automatically.

        This can be useful if we want to generate a sequence of numbers
        without affecting the underlying random state. This allows us to use
        a particular seed only in a given block of code, and then effectively
        forget about it.

        The temporary seed can be used in a with-statement, like:
        with rng.set_temp_seed(n):
            rng.random(...)

        When the with-block is ended, the old random state is restored, as if
        the temporary seed had never been set.
        """
        old_generator = self.rng
        self.rng = np.random.default_rng(temp_seed)
        yield self.rng  # we don't use this, but yielding a value is required
        self.rng = old_generator


# A shared random number generator to be used from different modules.
# Will be initialised at first import.
rng = ResettableRandomState()
