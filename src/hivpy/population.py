import datetime
import operator
from functools import reduce

import numpy as np
import pandas as pd

import hivpy.column_names as col

from .common import rng
from .demographics import DemographicsModule
from .hiv_status import HIVStatusModule
from .sexual_behaviour import SexualBehaviourModule

HIV_APPEARANCE = datetime.date(1989, 1, 1)


class Population:
    """A set of individuals with particular characteristics."""
    size: int  # how many individuals to create in total
    data: pd.DataFrame  # the underlying data
    params: dict  # population-level parameters
    date: datetime.date  # current date
    HIV_introduced: bool  # whether HIV has been introduced yet
    variable_history: dict  # how many steps we need to store for each variable
    step: int

    def __init__(self, size, start_date):
        """Initialise a population of the given size."""
        self.size = size
        self.date = start_date
        self.step = 0
        self.variable_history = {}
        self.demographics = DemographicsModule()
        self.sexual_behaviour = SexualBehaviourModule()
        self.hiv_status = HIVStatusModule()
        self.HIV_introduced = False
        self._sample_parameters()
        self._create_population_data()

    def _sample_parameters(self):
        """Randomly determine the uncertain population-level parameters."""
        # Example: Each person will have a predetermined max age,
        # which will come from a normal distribution. The mean of
        # that distrubition is chosen randomly for each population.
        # avg_max_age = random.choices([80, 85, 90], [0.4, 0.4, 0.2])
        # self.params = {
        #     'avg_max_age': avg_max_age,
        # }

    def _create_population_data(self):
        """Populate the data frame with initial values."""
        # NB This is a prototype. We should use the new numpy random interface:
        # https://numpy.org/doc/stable/reference/random/index.html#random-quick-start
        self.data = pd.DataFrame({
            "Dummy": [None] * self.size,
        })
        self.init_variable(col.SEX, self.demographics.initialize_sex(self.size))
        self.init_variable(col.AGE, self.demographics.initialise_age(self.size))  # when do we use current age and when previous timestep age?
        self.init_variable(col.AGE_GROUP, 0) 
        self.init_variable(col.DATE_OF_DEATH, None)
        self.init_variable(col.HIV_STATUS, False)
        self.init_variable(col.HIV_DIAGNOSIS_DATE, None)
        self.init_variable(col.NUM_PARTNERS, 0)    # I think we need t-3 for pregnancy but might change if we rethink implementation
                                                    # In this case the number of previous steps we need would actually be variable based on timestep!!!
        self.init_variable(col.RRED, 1)
        self.init_variable(col.LONG_TERM_PARTNER, False)
        self.init_variable(col.LTP_AGE_GROUP, 0)
        self.init_variable(col.LTP_LONGEVITY, 0)
        self.init_variable(col.SEX_MIX_AGE_GROUP, 0)
        self.init_variable(col.STP_AGE_GROUPS, np.array([[0]]*self.size))
        self.init_variable(col.RRED_LTP, 1)
        self.sexual_behaviour.init_sex_behaviour_groups(self)
        self.sexual_behaviour.init_risk_factors(self)
        self.sexual_behaviour.num_short_term_partners(self)
        self.sexual_behaviour.assign_stp_ages(self)
        # TEMP
        self.hiv_status.set_dummy_viral_load(self)
        # If we are at the start of the epidemic, introduce HIV into the population.
        if self.date >= HIV_APPEARANCE and not self.HIV_introduced:
            self.set_present_variable(col.HIV_STATUS, self.hiv_status.introduce_HIV(self))
            self.HIV_introduced = True

    def init_variable(self, name: str, init_val, n_prev_steps=0):
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
        if(n_prev_steps == 0):
            self.data[name] = init_val
        else:
            for i in range(0, n_prev_steps + 1):
                self.data[self.constructParamColumn(name, i)] = init_val

    def constructParamColumn(self, name, i):
        return name + "," + str(i)

    def get_sub_pop(self, conditions):
        """
        Get a dataframe representing a sub-population meeting a list conditions.\\
        Conditions are expressed as a tuple (variable, operator, value)\\
        e.g. `(col.AGE, operator.ge, 15)` gets people who are 15 and over\\
        `conditions` is a list (or other iterable) of such tuples.
        """
        index = reduce(operator.and_,
                       (op(self.data[self.get_correct_column(var)], val) for (var, op, val) in conditions))
        return self.data.index[index]

    def get_sub_pop_intersection(self, subpop_1, subpop_2):
        """
        Get the indexing of the intersection of two subpopulations
        """
        return pd.Index.intersection(subpop_1, subpop_2)

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
        #print("cols = ", self.data.columns)
        #print("Column to get = ", var_col)
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
        #(param, dt) = self.make_column_tuple(param_info)
        if(self.variable_history[param] == 1):
            return param
        else:
            col_index = (self.step + dt) % self.variable_history[param]
            return self.constructParamColumn(param, col_index)

    def set_variable_by_group(self, target, groups, func, use_size=True, sub_pop=None):
        """Sets the value of a population variable at the present time step by calling transform group."""
        target_col = self.get_correct_column(target, 0)
        if sub_pop is None:
            self.data[target_col] = self.transform_group(groups, func, use_size)
        else:
            self.data.loc[sub_pop, target_col] = self.transform_group(groups, func, use_size, sub_pop)

    def transform_group(self, param_list, func, use_size=True, sub_pop=None):
        """Groups the data by a list of parameters and applies a function to each grouping. \n
        `param_list` is a list of names of columns by which you want to group the data. The order
        must match the order of arguments taken by the function `func` \n
        `func` is a function which takes the values of those columns for a group (and optionally
        the size of the group, which should be the last argument) and returns a value or array of
        values of the size of the group. \n
        `use_size` is true by default, but should be set to false if `func` does not take the size
        of the group as an argument. \n
        `sub_pop` is `None` by default, in which case the transform acts upon the entire dataframe.
        If `sub_pop` is defined, then it acts only on the part of the dataframe defined
        by `data.loc[sub_pop]`"""
        # Use Dummy column to in order to enable transform method and avoid any risks to data
        param_list = list(map(lambda x: self.get_correct_column(x), param_list))
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
        return df.groupby(param_list)["Dummy"].transform(general_func)

    def evolve(self, time_step: datetime.timedelta):
        """Advance the population by one time step."""
        # Does nothing just yet except advance the current date, track ages
        # and set death dates.
        ages = self.get_variable(col.AGE)
        ages += time_step.days / 365  # Very naive!
        # Record who has reached their max age
        died_this_period = self.demographics.determine_deaths(self)
        # self.data.loc[died_this_period, col.DATE_OF_DEATH] = self.date
        self.set_present_variable(col.DATE_OF_DEATH, self.date, died_this_period)

        # Get the number of sexual partners this time step
        self.sexual_behaviour.update_sex_behaviour(self)
        self.hiv_status.update_HIV_status(self)

        # If we are at the start of the epidemic, introduce HIV into the population.
        if self.date >= HIV_APPEARANCE and not self.HIV_introduced:
            self.set_present_variable(col.HIV_STATUS, self.hiv_status.introduce_HIV(self))
            self.HIV_introduced = True

        # We should think about whether we want to return a copy or evolve
        # the population in-place. We will likely need a copy at some point.
        self.date += time_step
        self.step += 1
        return self
