import logging
from datetime import date, datetime, timedelta

import numpy as np
import pytest
import scipy.integrate

from hivpy import column_names as col
from hivpy.common import SexType
from hivpy.demographics import (ContinuousAgeDistribution, DemographicsModule,
                                StepwiseAgeDistribution)
from hivpy.demographics_data import DemographicsData
from hivpy.population import Population


def test_population_init():
    pop = Population(size=100, start_date=date(1989, 1, 1))
    print(pop)
    assert(len(pop.data)==100)
    assert((col.HIV_STATUS, 0) in pop.data.columns)