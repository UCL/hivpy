## General Tutorial

Welcome to the HIVpy package. Below is an overview of the repository contents:

- `src/hivpy/` - Package source code.
- `src/hivpy/data/` - YAML files containing module-relevant data and variables.
- `src/tests/` - Unit tests for each module.
- `docs/` - Auto-generated Sphinx documentation.
- `tutorials/` - More in-depth information about the package and its modules.

### Module-Specific Tutorials

- [Circumcision Tutorial](circumcision.md)
- [Pregnancy Tutorial](pregnancy.md)
- [Sexual Behaviour Tutorial](sexual_behaviour.md)

### Other Tutorials

- [Setup and Package Installation](setup.md)

### Common Procedures

#### Adding a New Module to Documentation

While the documentation is auto-generated (using docstring contents), `docs/hivpy.rst` should be manually updated to let Sphinx know to include any newly created modules. The following example can be used as a template:
```
hivpy.<module_name> module
-------------------------

.. automodule:: hivpy.<module_name>
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: <function_name>

hivpy.<module_name>\_data module
-------------------------------

.. automodule:: hivpy.<module_name>_data
   :members:
   :undoc-members:
   :show-inheritance:
```
The `<module_name>_data` module is optional, as your new module may not necessarily have a YAML file, and thus not need a data module.

Adding `:exclude-members:` is also optional, but it can be used to specify any functions or variables you would like to exclude from the documentation.

#### Adding a New Data Variable to a Module

If you would like to add a new data variable to one of your modules, first add it to the relevant `src/hivpy/data/<module_name>.yaml` file. Then add it to `__init__` in `src/hivpy/<module_name>_data.py`:
```python
self.your_var_name_here = self.data["your_var_name_here"]
```
If your variable is more than just a single value, use an appropriate data reader method instead:
```python
# e.g. reading in a discrete distribution
self.your_var_name_here = self._get_discrete_dist("your_var_name_here")
```
Now add your variable to `__init__` in `src/hivpy/<module_name>.py`:
```python
self.your_var_name_here = self.<mn>_data.your_var_name_here
```
If your variable needs to be picked from a distribution, use the sample method:
```python
self.your_var_name_here = self.<mn>_data.your_var_name_here.sample()
```
Please note that in both of the examples above `<mn>` is a stand-in for an abbreviation of the module's name.
