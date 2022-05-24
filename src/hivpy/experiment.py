import os
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentConfig, LoggingConfig, SimulationConfig
from .exceptions import OutputException
from .simulation import run_simulation


class OutputHandler:
    """
    Handles output for the experiment.
    Could be used to just keep track of files and handle writing/reading
    Or could also be used to calculate summary statistics
    Although if that is going to be elaborate, a separate statistics module
    could be useful.
    """
    files: dict
    output_dir: str
    simulation_data: pd.DataFrame
    summary_stats: np.ndarray

    def __init__(self, output_params):
        self.output_dir = os.path.normpath(output_params['OUTPUT_DIR'])
        self.files = {"Raw": "raw_data.out", "Stats": "summary_statistics.out"}

    def set_file_name(self, fileId, filename):
        filepath = os.path.join(self.output_dir, filename)
        self.files[fileId] = filepath

    def _commit_data(self, fileId, data, write_mode):
        if fileId not in self.files:
            raise OutputException(f"File matching {fileId} does not exist.")
        if(isinstance(data, pd.DataFrame)):
            data.to_csv(self.files[fileId], mode=write_mode)
        elif(isinstance(data, np.ndarray)):
            with open(self.files[fileId], write_mode) as outfile:
                np.savetxt(outfile, data, delimiter=",")

    def append(self, fileId, data):
        self._commit_data(self, fileId, data, 'a')

    def overwrite(self, fileId, data):
        self._commit_data(self, fileId, data, 'w')

    def _gen_statistic(self, f_stat, field):
        return f_stat(self.simulation_data[field])

    def gen_summary_stats(self):
        return [self._gen_statistic(f_stat, field) for (f_stat, field) in self.summary_stats]


def create_simulation(experiment_param):
    try:
        start_date = date(int(experiment_param['start_year']), 1, 1)
        end_date = date(int(experiment_param['end_year']), 12, 31)
        population_size = int(experiment_param['population'])
        interval = timedelta(days=int(experiment_param['time_interval_days']))
        output_dir = Path(experiment_param['simulation_output_dir'])
        if not output_dir.exists():
            output_dir.mkdir()
        return SimulationConfig(population_size, start_date, end_date, output_dir, interval)
    except ValueError as err:
        print('Error parsing the experiment parameters {}'.format(err))
    except KeyError as kerr:
        print('Error extracting values from the parameter set {}'.format(kerr))
    return None


def create_log(log_param):
    log_dir = log_param['log_directory']
    log_file = log_param['logfile_prefix']+"."+datetime.now().strftime("%y%m%d-%H%M%S")+".log"
    log_level = log_param['log_file_level']
    console_log_level = log_param['console_log_level']
    return LoggingConfig(log_dir, log_file, fileLogLevel=log_level,
                         consoleLogLevel=console_log_level)


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
