## General Tutorial

Welcome to the HIVpy package. Below is an overview of the repository contents:

- `src/hivpy/` - Package source code.
- `src/hivpy/data/` - YAML files containing module-relevant data and variables.
- `src/tests/` - Unit tests for each module.
- `docs/` - Auto-generated Sphinx documentation.
- `tutorials/` - More in-depth information about the package and its modules.

### Module-Specific Tutorials

- [Circumcision Tutorial](circumcision.md)

### Common Procedures

#### Adding a New Variable

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
