import os
from datetime import date, datetime, timedelta

from .config import ExperimentConfig, LoggingConfig, SimulationConfig
from .simulation import run_simulation


def create_simulation(experiment_param):
    try:
        start_date = date(int(experiment_param['START_YEAR']), 1, 1)
        end_date = date(int(experiment_param['END_YEAR']), 12, 31)
        population_size = int(experiment_param['POPULATION'])
        interval = timedelta(days=int(experiment_param['TIME_INTERVAL_DAYS']))
        return SimulationConfig(population_size, start_date, end_date, interval)
    except ValueError as err:
        print('Error parsing the experiment parameters {}'.format(err))
    except KeyError as kerr:
        print('Error extracting values from the parameter set {}'.format(kerr))
    return None


def create_log(log_param):
    log_dir = log_param['LOG_DIRECTORY']
    logfilename = log_param['LOGFILE_PREFIX']+"."+datetime.now().strftime("%y%m%d-%H%M%S")+".log"
    log_level = log_param['LOG_FILE_LEVEL']
    console_log_level = log_param['CONSOLE_LOG_LEVEL']
    logpath = os.path.join(log_dir, logfilename)
    return LoggingConfig(log_dir, logpath, fileLogLevel=log_level, consoleLogLevel=console_log_level)


def create_experiment(all_params):
    simulation_config = create_simulation(all_params['EXPERIMENT'])
    logging_config = create_log(all_params['LOGGING'])
    return ExperimentConfig(simulation_config, logging_config)


def run_experiment(experiment_config):
    """Run an entire experiment.

    An experiment can consist of one or more simulation runs,
    as well as processing steps after those are completed.
    """
    experiment_config.logging_config.start_logging()
    result = run_simulation(experiment_config.simulation_config)
    return result
