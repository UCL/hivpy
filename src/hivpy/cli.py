import argparse
import pathlib

import yaml

from .experiment import create_experiment, run_experiment

"""
Maybe we may want to register parameters/properties in future?
"""
# def register_parameters():
#     parser = argparse.ArgumentParser(description="register model parameters")
#     parser.add_argument("parameters", type=pathlib.Path,
#                         help="register_parameters parameters.csv")
#     args = parser.parse_args()
#     parameter_filepath = args.parameters


def run_model():
    """
    Running a HIV model simulation.
    Assuming there is a hivpy.yaml file, the command is
    $run_model hivpy.yaml.
    """
    parser = argparse.ArgumentParser(description="run a simulation")
    parser.add_argument("input", type=pathlib.Path, help="run_model config.yaml")
    args = parser.parse_args()
    config_filename = args.input
    try:
        with open(config_filename, 'r') as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as err:
        print("Error parsing yaml file {}".format(err))

    try:
        experiment_config = create_experiment(config)
        run_experiment(experiment_config)
    except KeyError as err:
        print('Error finding necessary configuration parameter or section {}'.format(err))


if __name__ == '__main__':
    print('calling main')
    run_model()
