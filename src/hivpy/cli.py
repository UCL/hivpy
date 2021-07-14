import click

@click.command()
def simulate():
    click.echo('run a simulation')

@click.command()
def echo():
    click.echo("hivpy command line interface")

if __name__ == '__main__':
    print("Welcome to HIVPy")
    echo()