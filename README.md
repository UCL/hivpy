# hivpy

This repository contains a Python package for running simulations
of the transmission of HIV across a population, and related quantities.

The code is under active development. While you are welcome to use it,
be aware that the structure, behaviour and interface (inputs and outputs)
are all likely to change.

More extensive documentation for this package is under development.

TODO: Link to publications using hiv-synthesis model and other pages.

## Getting started
### Prerequisites
This code requires Python 3.7 or newer.

We recommend that you use a virtual environment to install the package.
See the offical Python guide for [creating](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment)
and [activating](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#activating-a-virtual-environment) virtual environments.

If you are using Python through anaconda, see
[their guide for managing environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

### Installing the package
Once you have activated the relevant virtual environment (if desired),
you can install the package by running the following command
from the top level of this repository (the location of this README file):
```bash
pip install .
```
(If you are planning to make changes to the code, see the
developer instructions below.)

This will install the libraries that the package requires, and make available
the command-line tool for running simulations.

In the future, we plan to make the package available through the
[Python Package Index](https://pypi.org/). Until then, installing from source
is the only way to access the code.

## Documentation
We use [Sphinx](https://www.sphinx-doc.org/en/master/) to auto-generate our documentation. You can build it by running `make <target>` in the `docs/` directory. The following command will generate documentation in HTML format in `docs/_build/html/`:
```bash
make html
```
If you execute `make` without an argument you can find a list of other available format options.

### Accessing HTML documentation
Once you build the documentation as HTML, please visit our landing page `docs/_build/html/index.html` if you would like to search for something or browse our index pages. If you would like to jump straight to the page that contains all the most relevant `hivpy` package information, our documentation is best viewed through `docs/_build/html/hivpy.html`.

## Running simulations
Simulations can be run from the terminal with `run_model`:
```bash
run_model <configuration file>
```
The command takes as its argument the path to a YAML file,
which contains various settings for the simulation, such as
the start and end dates and the size of the population.
For an example, see the [sample file](./hivpy.yaml)
included in this repository.

When the simulation runs, it will produce a CSV file with outputs.
The file contains one row per time step and a column for each quantity
that is being tracked or computed.
The file name reflects the date and time that the simulation was launched,
for example `simulation_output_20220826-163116.csv`.

## Support
If you notice something that appears wrong, please let us know by
[opening an issue](https://github.com/UCL/hivpy/issues/new/choose)
on this repository.

## Developer instructions
### Installation
If you are planning to make changes to the package code,
use the following command to install the package instead:
```bash
pip install -e .[dev]
```
This will install some additional libraries and tools for development, and also create an "editable" installation; this means that any changes you make will be applied automatically, without needing to reinstall the package.
N.B. Depending on your shell interpreter you might need to use `.[dev]` inside quotes for the installation: 
```pip install -e '.[dev]'```
### Testing
The package comes with unit tests. To run them, simply run
`pytest` from the top level (this directory). The `pytest` package and command
are installed during the package installation if the `[dev]` extra option
is supplied, as above.

### Contributions
If you want to make changes to the code, we recommend the following workflow:
- Create a new git branch
- Make the relevant changes
- Ideally, implement unit test(s) to verify the correctness of your changes
- Run the tests (old and new)
- Run a linter (`flake8 src/`) to check for issues related to code style
-  [Open a pull request](https://github.com/UCL/hivpy/compare) to merge your changes

### Extending the model
The exact steps will change depending on the nature of the changes,
but this is a rough outline of the actions needed:

- Add variables to the population to track measures of interest.
- Find which model modules are affected; if adding new behaviour, consider whether it belongs in an existing module or is better placed in a new one.
- If needed, add outputs to be tracked in [`simulation.py`](src/hivpy/simulation.py).
- Implement one or more tests to cover your new changes, and run the whole test suite.
