import configparser
import argparse
import pathlib
from .experiment import create_experiment, create_output, run_experiment

'''
Maybe we may want to register parameters/properties in future?
'''
def register_parameters():
    parser = argparse.ArgumentParser(description="register model parameters")
    parser.add_argument("parameters", type=pathlib.Path, help="register_parameters parameters.csv")
    args = parser.parse_args()
    parameter_filepath = args.parameters


def run_model():
    parser = argparse.ArgumentParser(description="submit a simulation")
    parser.add_argument("input", type=pathlib.Path, help="run_model -i config.conf")
    args = parser.parse_args()
    conf_filename = args.input
    config = configparser.ConfigParser()
    try:
        config.read(conf_filename)
        experiment_config = create_experiment(config['EXPERIMENT'])
        output_config = create_output(config['OUTPUT'])
        run_experiment(experiment_config, output_config)

    except configparser.Error as err:
        print('error parsing the config file {}'.format(err))


if __name__ == '__main__':
    print('calling main')
    run_model()