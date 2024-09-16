import operator
from functools import reduce

import pandas as pd

import hivpy.column_names as col

from .circumcision import CircumcisionModule
from .common import LogicExpr, date, timedelta
from .demographics import DemographicsModule
from .hiv_diagnosis import HIVDiagnosisModule
from .hiv_status import HIVStatusModule
from .hiv_testing import HIVTestingModule
from .pregnancy import PregnancyModule
from .prep import PrEPModule
from .sexual_behaviour import SexualBehaviourModule

HIV_APPEARANCE = date(1989, 1, 1)


class Population:
    """
    A set of individuals with particular characteristics.
    """
    size: int  # how many individuals to create in total
    data: pd.DataFrame  # the underlying data
    params: dict  # population-level parameters
    date: date  # current date
    HIV_introduced: bool  # whether HIV has been introduced yet
    variable_history: dict  # how many steps we need to store for each variable
    step: int

    def __init__(self, size, start_date):
        """
        Initialise a population of the given size.
        """
        self.size = size
        self.date = start_date
        self.step = 0
        self.variable_history = {}
        self.demographics = DemographicsModule()
        self.circumcision = CircumcisionModule()
        self.sexual_behaviour = SexualBehaviourModule()
        self.pregnancy = PregnancyModule()
        self.hiv_status = HIVStatusModule()
        self.hiv_testing = HIVTestingModule()
        self.hiv_diagnosis = HIVDiagnosisModule()
        self.prep = PrEPModule()
        self.HIV_introduced = False
        self._sample_parameters()
        self._create_population_data()
        # Can be useful to switch off death during some tests
        # so that random deaths don't interfere with results
        self.apply_death = True

    def _sample_parameters(self):
        """
        Randomly determine the uncertain population-level parameters.
        """
        # Example: Each person will have a predetermined max age,
        # which will come from a normal distribution. The mean of
        # that distribution is chosen randomly for each population.
        # avg_max_age = random.choices([80, 85, 90], [0.4, 0.4, 0.2])
        # self.params = {
        #     'avg_max_age': avg_max_age,
        # }

    def _create_population_data(self):
        """
        Populate the data frame with initial values.
        """
        # NB This is a prototype. We should use the new numpy random interface:
        # https://numpy.org/doc/stable/reference/random/index.html#random-quick-start
        self.data = pd.DataFrame()
        self.data["Dummy"] = pd.Series([None] * self.size, dtype=object)
        self.init_variable(col.SEX, self.demographics.initialise_sex(self.size))
        self.init_variable(col.AGE, self.demographics.initialise_age(self.size))
        self.init_variable(col.AGE_GROUP, 0)

        self.hiv_status.init_HIV_variables(self)
        self.prep.init_prep_variables(self)
        self.init_variable(col.TEST_MARK, False)
        self.init_variable(col.EVER_TESTED, False)
        self.init_variable(col.LAST_TEST_DATE, None)
        self.init_variable(col.NSTP_LAST_TEST, 0)
        self.init_variable(col.NP_LAST_TEST, 0)
        self.init_variable(col.STI, False)

        self.sexual_behaviour.init_sex_behaviour(self)

        self.init_variable(col.CIRCUMCISED, False)
        self.init_variable(col.CIRCUMCISION_DATE, None)
        self.init_variable(col.VMMC, False)
        self.init_variable(col.HARD_REACH, False)

        self.pregnancy.init_pregnancy(self)

        self.demographics.initialise_hard_reach(self.data)
        if self.circumcision.vmmc_disrup_covid:
            self.circumcision.init_birth_circumcision_born(self.data, self.date)
        else:
            self.circumcision.init_birth_circumcision_all(self.data, self.date)
        self.sexual_behaviour.assign_stp_ages(self)

        # If we are at the start of the epidemic, introduce HIV into the population.
        if self.date >= HIV_APPEARANCE and not self.HIV_introduced:
            self.hiv_status.introduce_HIV(self)
            self.HIV_introduced = True

    def init_variable(self, name: str, init_val, n_prev_steps=0, data_type=None):
        """
           New variable will be initialised as a collection of columns.\\
           Column names (keys) will be (name, 0) ... (name, n_prev_steps).\\
           Updates dictionary to keep track of number of time steps being stored for each
           variable.
           name: string, name of variable
           n_prev_steps: integer, number of previous iterations of this variable which need
           to be stored (default 0).
        """
        self.variable_history[name] = n_prev_steps + 1
        if (n_prev_steps == 0):
            if data_type is not None:
                self.data[name] = pd.Series([init_val]*self.size, dtype=data_type)
            else:
                self.data[name] = init_val
        else:
            for i in range(0, n_prev_steps + 1):
                if data_type is not None:
                    self.data[self.constructParamColumn(name, i)] = pd.Series([init_val]*self.size, dtype=data_type)
                else:
                    self.data[self.constructParamColumn(name, i)] = init_val

    def constructParamColumn(self, name, i):
        return name + "," + str(i)

    def get_sub_pop(self, conditions):
        """
        Get a dataframe representing a sub-population meeting a list of conditions.\\
        Conditions are expressed as a tuple (variable, operator, value)\\
        e.g. `(col.AGE, operator.ge, 15)` gets people who are 15 and over\\
        `conditions` is a list (or other iterable) of such tuples.
        """
        # if / else statements here for backwards compatibility
        # (Python doesn't allow overloaded functions)
        if (isinstance(conditions, LogicExpr)):
            return self.data.index[conditions.eval(self)]
        else:
            index = reduce(operator.and_,
                           (self.disjunction(expr)
                            for expr in conditions))
            return self.data.index[index]

    def disjunction(self, expr):
        """
        Evaluate a disjunction so that is can be used in CNF expressions.
        """
        if isinstance(expr, list):
            return reduce(operator.or_,
                          (self.eval(sub_expr)
                           for sub_expr in expr))
        else:
            return self.eval(expr)

    def eval(self, expr):
        var, op, val = expr
        if val is None:
            if op == operator.eq:
                return self.data[self.get_correct_column(var)].isnull()
            else:
                return self.data[self.get_correct_column(var)].notnull()
        else:
            return op(self.data[self.get_correct_column(var)], val)

    def apply_bool_mask(self, bool_mask, sub_pop=None):
        if sub_pop is None:
            return self.data.index[bool_mask]
        else:
            return sub_pop[bool_mask]

    def get_sub_pop_intersection(self, subpop_1, subpop_2):
        """
        Get the indexing of the intersection of two subpopulations.
        """
        return pd.Index.intersection(subpop_1, subpop_2)

    def get_sub_pop_union(self, *args):
        """
        Get the indexing of the union of two subpopulations.
        """
        return reduce(pd.Index.union, args)

    def get_sub_pop_from_array(self, array, sub_pop=None):
        if sub_pop is None:
            return self.data.index[array]
        else:
            return self.data.loc[sub_pop].index[array]

    def apply_vector_func(self, params, func):
        param_cols = list(map(lambda x: self.get_variable(x), params))
        return func(*param_cols)

    def apply_function(self, function, axis, sub_pop=None):
        if sub_pop is None:
            return self.data.apply(function, axis)
        else:
            return self.data.loc[sub_pop].apply(function, axis)

    def get_variable(self, var, sub_pop=None, dt=0):
        var_col = self.get_correct_column(var, dt)
        if sub_pop is None:
            return self.data[var_col]
        else:
            return self.data.loc[sub_pop, var_col]

    def set_present_variable(self, target: str, value, sub_pop=None):
        present_col = self.get_correct_column(target, 0)
        if sub_pop is None:
            self.data[present_col] = value
        else:
            self.data.loc[sub_pop, present_col] = value

    def get_correct_column(self, param, dt=0):
        """Gets the correct column for a parameter and a given time delay."""
        if (self.variable_history[param] == 1):
            return param
        else:
            col_index = (self.step + dt) % self.variable_history[param]
            return self.constructParamColumn(param, col_index)

    def set_variable_by_group(self, target, groups, func, use_size=True, sub_pop=None):
        """Sets the value of a population variable at the present time step
        by calling transform group."""
        target_col = self.get_correct_column(target, 0)
        if sub_pop is None:
            self.data[target_col] = self.transform_group(groups, func, use_size)
        else:
            self.data.loc[sub_pop, target_col] = self.transform_group(groups, func,
                                                                      use_size, sub_pop)

    def transform_group(self, param_list, func, use_size=True, sub_pop=None, dropna=False):
        """
        Groups the data by a list of parameters and applies a function to each grouping.

        `param_list` is a list of names of columns by which you want to group the data. The order
        must match the order of arguments taken by the function `func`. \n
        `func` is a function which takes the values of those columns for a group (and optionally
        the size of the group, which should be the last argument) and returns a value or array of
        values of the size of the group. \n
        `use_size` is true by default, but should be set to false if `func` does not take the size
        of the group as an argument. \n
        `sub_pop` is `None` by default, in which case the transform acts upon the entire dataframe.
        If `sub_pop` is defined, then it acts only on the part of the dataframe defined
        by `data.loc[sub_pop]`. \n
        `dropna` is false by default to allow for the inclusion of missing values in groups, but
        should be set to true if missing values should instead be dropped during groupby.
        """
        def general_func(g):
            if len(param_list) == 1:
                args = [g.name]
            else:
                args = list(g.name)

            if (use_size):
                args.append(g.size)
            return func(*args)

        if sub_pop is not None:
            df = self.data.loc[sub_pop]
        else:
            df = self.data
        # Use Dummy column to in order to enable transform method and avoid any risks to data
        return df.groupby(param_list, dropna=dropna)["Dummy"].transform(general_func)

    def evolve(self, time_step: timedelta):
        """
        Advance the population by one time step.
        """
        # Does nothing just yet except advance the current date, track ages
        # and set death dates.
        ages = self.get_variable(col.AGE)
        ages += time_step.month / 12
        self.set_present_variable(col.AGE, ages)
        n_deaths = 0

        self.hiv_status.reset_diagnoses(self)

        if self.HIV_introduced:
            self.hiv_status.set_primary_infection(self)
            self.hiv_status.set_viral_load_groups(self)
            self.prep.prep_willingness(self)

        if self.circumcision.vmmc_disrup_covid:
            self.circumcision.update_birth_circumcision(self.data, time_step, self.date)
        self.circumcision.update_vmmc(self, time_step)
        # Get the number of sexual partners this time step
        self.sexual_behaviour.update_sex_behaviour(self)
        self.pregnancy.update_pregnancy(self)

        # If HIV has been introduced, then run HIV relevant code
        if self.HIV_introduced:
            self.hiv_status.update_HIV_status(self)
            HIV_deaths = self.hiv_status.HIV_related_disease_risk(self, time_step)
            self.hiv_testing.update_hiv_testing(self, time_step)
            n_deaths = n_deaths + sum(HIV_deaths)
            if (n_deaths and self.apply_death):
                self.drop_from_population(HIV_deaths)

        # Some population cleanup
        self.pregnancy.reset_anc_at_birth(self)

        # Apply non-hiv deaths
        non_HIV_deaths = self.demographics.determine_deaths(self, time_step)
        n_deaths = n_deaths + sum(non_HIV_deaths)
        if (sum(non_HIV_deaths) and self.apply_death):
            self.drop_from_population(non_HIV_deaths)

        # If we are at the start of the epidemic, introduce HIV into the population.
        if self.date >= HIV_APPEARANCE and not self.HIV_introduced:
            self.hiv_status.introduce_HIV(self)
            self.HIV_introduced = True

        # We should think about whether we want to return a copy or evolve
        # the population in-place. We will likely need a copy at some point.
        self.date += time_step
        self.step += 1
        return self

    def drop_from_population(self, deaths: pd.Series):
        indices = deaths[deaths].index  # indices where deaths==True
        self.data.drop(indices, inplace=True)
