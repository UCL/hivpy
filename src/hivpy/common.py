"""Functionality shared between multiple parts of the framework."""

import operator
from contextlib import contextmanager
from enum import IntEnum
from functools import reduce

import numpy as np
import scipy.stats as stat

import hivpy.column_names as col


class DiscreteChoice:
    def __init__(self, vals: np.ndarray, probs):
        N = len(vals)
        if(len(probs) != N):
            raise Exception
        index_range = np.arange(0, N, 1)
        self.data = vals
        self.dist = stat.rv_discrete(values=(index_range, probs))

    def sample(self, size=None):
        """Samples from a random discrete distribution.
        Used without an argument it will return a single object.
        Used with the "size" argument it will return an array of that size."""
        if size is None:
            index = self.dist.rvs()
            return self.data[index]
        else:
            indices = np.array(self.dist.rvs(size=size))
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


def transform_group(df, param_list, func, use_size=True):
    """Groups the data by a list of parameters and applies a function to each grouping"""
    # HIV_STATUS is just a dummy column to allow us to use the transform method
    def general_func(g):
        args = [n for n in g.name] # list(g.name)
        if(use_size):
            args.append(g.size)
        return func(*args)
    return df.groupby(param_list)[col.HIV_STATUS].transform(general_func)


def between(values, limits):
    """A helper function for selecting values within a range."""
    min_value, max_value = limits
    # def _is_in_range(values):
    return (min_value <= values) & (values < max_value)
    # return _is_in_range


class ResettableRandomState:
    """A convenience class for using the NumPy random number generator.

    This is meant to be used as: `from hivpy.common import rng`

    For most things, this can be used exactly as NumPy's `Generator`. All the
    methods like `random`, `normal`, `choice` are supported and called in
    exactly the same way.

    The primary purpose of this is to share the `Generator` across multiple files,
    while allowing anyone to set the seed, whether in tests or the main code.
    It also offers the ability to use a temporary seed.
    """
    def __init__(self):
        """Create a new wrapper around a NumPy Generator."""
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
        ```
        with rng.set_temp_seed(17):
            rng.random(...)  # any calls to random methods here
        ```
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
