from .simulation import run_simulation


def run_experiment(experiment_config):
    """Run an entire experiment.

    An experiment can consist of one or more simulation runs,
    as well as processing steps after those are completed.
    """
    experiment_config.output_config.start_logging()
    result = run_simulation(experiment_config.simulation_config)
    return result
