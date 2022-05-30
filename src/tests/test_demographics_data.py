import importlib.resources

import pytest

from hivpy.common import SexType
from hivpy.demographics_data import DemographicsData


@pytest.fixture(scope="module")
def default_data():
    with importlib.resources.path("hivpy", "data") as data_path:
        path = data_path / "demographics.yaml"
    return DemographicsData(path)


def test_death_rate_ages(default_data):
    """Check that the default death rate specification is consistent."""
    male_death_rates = default_data.death_rates[SexType.Male]
    female_death_rates = default_data.death_rates[SexType.Female]
    age_limits = default_data.death_age_limits
    assert len(male_death_rates) == len(female_death_rates)
    # We only record the lower limit and forego the first group (assumed death rate 0)
    assert len(age_limits) == len(male_death_rates) - 1
