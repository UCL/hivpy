"""Functionality shared between multiple parts of the framework."""

import operator
from enum import IntEnum
from functools import reduce


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
