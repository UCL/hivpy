from datetime import date, timedelta
from .simulation import run_simulation
from .config import SimulationConfig, OutputConfig
import os


def create_experiment(experiment_param):
    try:
        start_date = date(int(experiment_param['START_YEAR']),1 ,1 )
        end_date = date(int(experiment_param['END_YEAR']),12, 31)
        population_size = int(experiment_param['POPULATION'])
        interval = timedelta(days = int(experiment_param['TIME_INTERVAL_DAYS']))
        return SimulationConfig(population_size, start_date, end_date, interval)
    except ValueError as err:
        print('Error parsing the experiment parameters {}'.format(err))
    except KeyError as kerr:
        print('Error extracting values from the parameter set {}'.format(kerr))
    return None

    # Dummy values for now

def create_output(output_param):
    outputdir = output_param['OUTPUT_DIRECTORY']
    logfilename = output_param['LOGOUTPUT_NAME']
    log_level = output_param['LOG_LEVEL']
    logpath = os.path.join(outputdir, logfilename)
    return OutputConfig(outputdir, logpath, log_level)



def run_experiment(experiment_config):
    """Run an entire experiment.

    An experiment can consist of one or more simulation runs,
    as well as processing steps after those are completed.
    """
    experiment_config.output_config.start_logging()
    result = run_simulation(experiment_config.simulation_config)
