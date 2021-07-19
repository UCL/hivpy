import click
import configparser
from experiment import create_experiment_from_config, create_experiment_setup


@click.command()
@click.argument('conf_filename', type=click.Path(exists=True))
def submit(conf_filename):
    click.echo(click.format_filename(conf_filename))
    config = configparser.ConfigParser()
    try:
        config.read(conf_filename)
        experiment_config = create_experiment_from_config(config['EXPERIMENT'])
        general_config = create_experiment_setup(config['GENERAL'])

    except configparser.Error as err:
        print('error parsing the config file {}'.format(err))


if __name__ == '__main__':
    submit()