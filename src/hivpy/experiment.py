from dataclasses import dataclass

from .population import Population
from .simulation import run_simulation, SimulationConfig


@dataclass
class ExperimentConfig:
    """Configuration parameters covering all stages of an experiment."""
    population_size: int
    simulation: SimulationConfig

    def initialize_population(self):
        """Create an initial population for use in simulations."""
        return Population(self.population_size)


def create_experiment_from_file(filename):
    """Create an experiment config from a YAML file."""
    # Dummy values for now
    from datetime import date, timedelta
    sim_config = SimulationConfig(date(1980, 1, 1), date(2040, 12, 31), timedelta(days=30))
    return ExperimentConfig(1000, sim_config)


def run_experiment(experiment_config):
    """Run an entire experiment.

    An experiment can consist of one or more simulation runs,
    as well as processing steps after those are completed.
    """
    simulation_config = experiment_config.simulation
    population = experiment_config.initialize_population()
    result = run_simulation(population, simulation_config)
