from .simulation import run_simulation


class ExperimentConfig:
    pass


def initialize_population(experiment_config):
    pass


def run_experiment(experiment_config):
    """Run an entire experiment.

    An experiment can consist of one or more simulation runs,
    as well as processing steps after those are completed.
    """
    simulation_config = experiment_config.simulation
    population = initialize_population(experiment_config)
    result = run_simulation(population, simulation_config)
