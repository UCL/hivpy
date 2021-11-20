from .config import ExperimentConfig, OutputConfig, SimulationConfig
from .simulation import run_simulation


def create_experiment(all_params):
    simulation_config = SimulationConfig.from_file(all_params['EXPERIMENT'])
    output_config = OutputConfig.from_file(all_params['OUTPUT'])
    return ExperimentConfig(simulation_config, output_config)


def run_experiment(experiment_config):
    """Run an entire experiment.

    An experiment can consist of one or more simulation runs,
    as well as processing steps after those are completed.
    """
    experiment_config.output_config.start_logging()
    result = run_simulation(experiment_config.simulation_config)
    return result
