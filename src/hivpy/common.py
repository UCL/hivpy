"""
Functionality shared between multiple parts of the framework.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .population import Population

import operator
from abc import ABC, abstractmethod
from contextlib import contextmanager
from enum import IntEnum
from functools import reduce

import numpy as np
import scipy.stats as stat


class SeedManager:
    FixSeed = True  # set this in a config
    UniversalSeed = 50  # set this in a config


seedManager = SeedManager()


class DiscreteChoice:
    def __init__(self, vals: np.ndarray, probs):
        N = len(vals)
        if (len(probs) != N):
            raise Exception
        index_range = np.arange(0, N, 1)
        self.probs = probs
        self.data = vals
        seed = seedManager.UniversalSeed if seedManager.FixSeed else None
        self.dist = stat.rv_discrete(values=(index_range, probs), seed=seed)

    def sample(self, size=None):
        """
        Samples from a random discrete distribution.
        Used without an argument it will return a single object.
        Used with the "size" argument it will return an array of that size.
        """
        if size is None:
            index = self.dist.rvs()
            return self.data[index]
        else:
            indices = np.array(self.dist.rvs(size=size))
            return self.data[indices]


class SexType(IntEnum):
    Male = 0
    Female = 1


def opposite_sex(sex: SexType):
    return (1 - sex)


class date:
    def __init__(self, year, month=1, day=1):
        self.year = year
        self.month = month
        self.day = day

    def __add__(self, delta):
        year = self.year + delta.year + (self.month + delta.month)//12
        month = (self.month + delta.month) % 12
        return date(year, month, self.day)

    def __sub__(self, delta):
        if (type(delta) is timedelta):
            return self.__add__(timedelta(years=-delta.year, months=-delta.month))
        elif (type(delta) is date):
            month = (self.month - delta.month) % 12
            year = (self.year - delta.year) + (self.month - delta.month)//12
            return timedelta(year, month)

    def __repr__(self):
        return f"({self.year}, {self.month}, {self.day})"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, d2):
        return (self.year == d2.year) and (self.month == d2.month) and (self.day == d2.day)

    def __gt__(self, d2):
        if (self.year != d2.year):
            return (self.year > d2.year)
        elif (self.month != d2.month):
            return (self.month > d2.month)
        else:
            return (self.day > d2.day)

    def __lt__(self, d2):
        if (self.year != d2.year):
            return (self.year < d2.year)
        elif (self.month != d2.month):
            return (self.month < d2.month)
        else:
            return (self.day < d2.day)

    def __ge__(self, d2):
        return not self.__lt__(d2)

    def __le__(self, d2):
        return not self.__gt__(d2)

    def __truediv__(self, dt):
        months1 = self.year * 12 + self.month
        months2 = dt.year * 12 + dt.month
        return months1 / months2


def float_to_date(fp_year):
    int_year = int(fp_year)
    int_month = int((fp_year - int_year) * 12)
    return date(int_year, int_month)


class timedelta:
    def __init__(self, years=0, months=0, days=0):
        self.year = years
        self.month = months + (days // 30)

    def __mul__(self, x):
        years = self.year * x
        year_remainder = years % 1
        years = int(years)
        months = (self.month * x) + (year_remainder * 12)
        years = years + int(months//12)
        months = int(months % 12)
        return timedelta(years=years, months=months)

    def __rmul__(self, x):
        return self.__mul__(x)

    def __add__(self, x):
        year = self.year + x.year + (self.month + x.month) // 12
        month = (self.month + x.month) % 12
        return timedelta(years=year, months=month)

    def __repr__(self) -> str:
        return f"dt({self.year, self.month, 0})"

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, dt2: timedelta):
        return (self.year == dt2.year) and (self.month == dt2.month)

    def __gt__(self, dt2: timedelta):
        if (self.year != dt2.year):
            return (self.year > dt2.year)
        else:
            return (self.month > dt2.month)

    def __lt__(self, dt2):
        if (self.year != dt2.year):
            return (self.year < dt2.year)
        else:
            return (self.month < dt2.month)

    def __ge__(self, dt2):
        return not self.__lt__(dt2)

    def __le__(self, dt2):
        return not self.__gt__(dt2)

    def __truediv__(self, x):
        months1 = self.year * 12 + self.month
        months2 = x.year * 12 + x.month
        return months1 / months2

    def years(self):
        return self.year + (self.month / 12)


def diff_years(date_end: date, date_begin: date):
    return (date_end.year - date_begin.year) + (date_end.month - date_begin.month) / 12


def between(values, limits):
    """
    A helper function for selecting values within a range.
    """
    min_value, max_value = limits
    # def _is_in_range(values):
    return (min_value <= values) & (values < max_value)
    # return _is_in_range


class ResettableRandomState:
    """
    A convenience class for using the NumPy random number generator.

    This is meant to be used as: `from hivpy.common import rng`.

    For most things, this can be used exactly as NumPy's `Generator`. All the
    methods like `random`, `normal`, `choice` are supported and called in
    exactly the same way.

    The primary purpose of this is to share the `Generator` across multiple files,
    while allowing anyone to set the seed, whether in tests or the main code.
    It also offers the ability to use a temporary seed.
    """
    def __init__(self):
        """
        Create a new wrapper around a NumPy Generator.
        """
        self.rng = np.random.default_rng(50)

    def __getattr__(self, name):
        """
        Delegate method calls and attribute lookups to the Generator.
        """
        return getattr(self.rng, name)

    def set_seed(self, seed):
        """
        Set the seed that controls the sequence of values generated.

        Generating samples from the same seed should always return the same
        results.
        """
        self.rng = np.random.default_rng(seed)

    @contextmanager
    def set_temp_seed(self, temp_seed):
        """
        Allow setting the seed temporarily and restoring it automatically.

        This can be useful if we want to generate a sequence of numbers
        without affecting the underlying random state. This allows us to use
        a particular seed only in a given block of code, and then effectively
        forget about it.

        The temporary seed can be used in a with-statement, like:

        .. code-block:: python

            with rng.set_temp_seed(17):
                rng.random(...)  # any calls to random methods here

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


class LogicExpr(ABC):
    """
    Abstract class for representing general logical expressions as applied to a population.
    Derived classes are: COND for individual conditions (e.g. age > 15)
                         AND for conjunctions
                         OR for disjunctions
    AND and OR can take a list of LogicExpr which may be of any of the three types to
    give full freedom to create any logical statement without being beholden to e.g.
    conjunctive normal form.
    """
    @abstractmethod
    def eval(self, pop: Population):
        pass


class COND(LogicExpr):
    """
    LogicExpr type for individual conditions. These are initialised in the form:
    COND(var, op, val) where:
    var: The variable you want to check as a dataframe column e.g. col.AGE
    op: The operator you want to apply, such as op.eq, op.le, op.gt etc.
    val: The value you want to compare the variable to
    As an example, expressing "age > 15" would be
    COND(col.AGE, op.gt, 15)
    """
    def __init__(self, var, op, val):
        self.var = var
        self.op = op
        self.val = val

    def eval(self, pop: Population):
        if self.val is None:
            if self.op == operator.eq:
                return pop.data[pop.get_correct_column(self.var)].isnull()
            else:
                return pop.data[pop.get_correct_column(self.var)].notnull()
        else:
            return self.op(pop.data[pop.get_correct_column(self.var)], self.val)


class AND(LogicExpr):
    """
    LogicExpr for expressing conjunctions.
    You can call the constructor with an arbitrary number of arguments, where
    each argument is a LogicExpr type (i.e. COND, AND, or OR).
    For example to express age > 15 and age < 65 we can write:
    `AND(COND(col.AGE, op.gt, 15), COND(col.AGE, op.lt, 65))`
    """
    def __init__(self, *props):
        self.props = props

    def eval(self, pop: Population):
        return reduce(operator.and_,
                      (p.eval(pop) for p in self.props))


class OR(LogicExpr):
    """
    LogicExpr for expressing disjunctions.
    You can call the constructor with an arbitrary number of arguments, where
    each argument is a LogicExpr type (i.e. COND, AND, or OR).
    For example to express age < 15 or age > 65 we can write:
    `OR(COND(col.AGE, op.lt, 15), COND(col.AGE, op.gt, 65))`
    """
    def __init__(self, *props):
        self.props = props

    def eval(self, pop: Population):
        return reduce(operator.or_,
                      (p.eval(pop) for p in self.props))
