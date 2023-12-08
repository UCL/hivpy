from hivpy.common import date

import numpy as np

from hivpy.common import rng
from hivpy.population import Population


def test_seed_reproduce():
    """
    Check that using the same seed value produces the same random values.
    """
    sample_size = 10
    # Get samples with a specific seed
    rng.set_seed(50)
    samples1 = rng.random(size=sample_size)
    # Do the same with a different seed, and repeat with the first
    rng.set_seed(42)
    samples2 = rng.random(size=sample_size)
    rng.set_seed(50)
    samples3 = rng.random(size=sample_size)
    # The first and last set of samples should be the identical...
    assert all(samples1 == samples3)
    # ...but the second should be different
    assert not all(samples1 == samples2)


def test_seed_context():
    """
    Check that setting the temporary seed has the correct effect.
    """
    sample_size = 10
    # Get samples with a particular seed
    rng.set_seed(50)
    samples_base = rng.random(size=sample_size)
    # Get the same from a temporary context; they should be the same
    with rng.set_temp_seed(50):
        samples_context = rng.random(size=sample_size)
    assert all(samples_context == samples_base)


def test_seed_context_switching():
    """
    Check that the right state is restored after using a temporary seed.
    """
    sample_size = 10
    rng.set_seed(50)
    samples_base = rng.random(size=2*sample_size)
    # Reset the seed. Get some samples
    rng.set_seed(50)
    samples_partial1 = rng.random(size=sample_size)
    # Interrupt with a temporary, different seed
    with rng.set_temp_seed(42):
        samples_temp = rng.random(size=sample_size)
    # Get some more samples from the initial seed
    samples_partial2 = rng.random(size=sample_size)
    # Check that the interruption had no effect
    assert all(samples_base == np.concatenate((samples_partial1, samples_partial2)))
    # And also that the temporary seed produced different values
    assert not all(samples_temp == samples_partial2)


def test_rng_in_functions():
    rng.set_seed(50)
    N = 1000
    pop1 = Population(size=N, start_date=date(1989, 1, 1))
    rng.set_seed(50)
    pop2 = Population(size=N, start_date=date(1989, 1, 1))
    assert (pop1.data.equals(pop2.data))
