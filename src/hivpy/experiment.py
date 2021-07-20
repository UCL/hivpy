from dataclasses import dataclass
from datetime import date, timedelta
from .simulation import run_simulation, SimulationConfig
import logging
import os

LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

@dataclass
class OutputConfig:
    output_dir: str
    logfile: str
    loglevel: str

    def start_logging(self):
        logging.basicConfig(filename=self.logfile, level=LEVELS[self.loglevel])
        logging.info("starting experiment")



def create_experiment(experiment_param):
    start_date = date(1980,1,1)
    end_date = date(2040, 12, 31)
    interval = timedelta(days=30)
    population_size = 1000
    try:
        start_date = date(int(experiment_param['START_YEAR']),1 ,1 )
        end_date = date(int(experiment_param['END_YEAR']),12, 31)
        population_size = int(experiment_param['POPULATION'])
        interval = timedelta(days = int(experiment_param['TIME_INTERVAL_DAYS']))
    except RuntimeError as err:
        print('Error parsing the experiment parameters {}'.format(err))

    # Dummy values for now
    return SimulationConfig(population_size, start_date, end_date, interval)

def create_output(output_param):
    outputdir = output_param['OUTPUT_DIRECTORY']
    logfilename = output_param['LOGOUTPUT_NAME']
    log_level = output_param['LOG_LEVEL']
    logpath = os.path.join(outputdir, logfilename)
    return OutputConfig(outputdir, logpath, log_level)



def run_experiment(experiment_config, output_config):
    """Run an entire experiment.

    An experiment can consist of one or more simulation runs,
    as well as processing steps after those are completed.
    """
    output_config.start_logging()
    result = run_simulation(experiment_config)
