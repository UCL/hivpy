# Introduction to HIVpy Development

In this tutorial I want to go through the basic steps of introducing new functionality to `HIVpy`. In order to do this we'll need to engage with a few core Python concepts like variables, functions, and classes. 

We'll cover the following topics in this tutorial:

- Quick overview of important structures
    - The population object
    - HIVpy modules
    - The main loop
- Making a new "Module" class
- Adding variables to the population
- Reading \& writing variables 
- Getting sub-populations
- Applying functions by groups
- Adding functionality to the main loop
- Output

There's a lot here so don't worry if it takes a while to go through it all! Once you've gone through this and got used to the syntax and structure, you'll be able to modify and create new functionality for `HIVpy`. 

> **N.B.** `HIVpy` uses the [`pandas`](https://pandas.pydata.org/docs/) library to represent the population data, but modifications to the data can (and generally should) be made using the `HIVpy` wrappers that we will discuss in this section. Nevertheless, you may wish to look into the `pandas` library documentation in the future if you want to understand the data structure better or do things with the data that are not already provided for by wrapper functions. 

## A Quick overview of Important Structures

The `HIVpy` code is made up of a number of classes, which define types of objects. Objects can have data (_variables_) and behaviour (_functions_). 

#### Revision: Classes in Python
Python classes are defined using syntax that looks like this (`#` denotes a comment in Python):
```python=
class MyClass:
    # variables defined here exist for any object of this class
    # We can set them equal to come value or not
    length = 0
    # We can optionally state their type using ": type"
    name: string

    # we can define functions for the class as well
    # "self" is usually used as the default first argument
    # "self" allows us to reference the object itself, for example its internal value "x"
    def length_squared(self):
        return (self.length)**2
    
    # __init__ is a special function which is called when an object is created
    # In this case it sets the internal variables "name" and "x"
    # "name" is set to be whatever string is passed in, and "x" is the length of that string
    def __init__(self, input_name):
        name = input_name
        length = len(input_name)
```

Using the definition above, we can write code such as:

```python
# Create an object of type MyClass, passing the information needed for __init__. We don't need to pass an argument for "self" because it is implicit.
my_object = MyClass("Walter")

# Access the data for our specific object. Every object we create of this type can have different data
print(my_object.name, my_object.length)

# Call one of the object's functions. 
# Again, we don't pass anything for "self" because the self is implicit: 
# because we have used this "." notation to call the function from this specific object, the first argument is always the object itself. 
n = my_object.length_squared()
print(n)
```
which would give the output
```
Walter, 6
36
```

### The Population Object

The most important object in `HIVpy` is the _population_, which is defined in the `Population` class in `population.py`. The population data is stored as a `pandas` dataframe: essentially just a table. Each _row_ represents an individual person, and each _column_ represents a variable that is tracked for the person, such as their sex, age, and number of sexual partners. 

The population has references to the `HIVpy` modules, each of which is responsible for a different aspect of the modelling such as sexual behaviour, pregnancy, HIV transmission and so on. This structure exists to break the code up into more manageable chunks and make functionality easy to find. 

The `__init__` function sets up the population object. The `_create_population_data` function is important as it creates the table which contains the data and initialises the various columns which are required. Many of these will end up being done inside other modules by calling functions like `init_sex_behaviour` which initialises the variables required for the `sexual_behaviour_` module. 

You can access the population data directly, but the population class provides many functions to access and manipulate this data. Using these functions has a few advantages:
- It standardises the way we interact with the data-frame.
- It avoid repeated logic throughout the code.
- If the `pandas` library changes in ways that require changes to the code, we only have to modify this small collection of functions rather than all the code which uses them.
- Changes to the way the data is represented, searched for, or manipulated (for efficiency, flexibility, or even to use entirely different libraries) can be made without changing anything outside the `Population` class.

We'll look into these functions in the following sections. 

### HIVpy Modules

`HIVpy` contains a number of modules, each of which deals with a different aspect of the simulation. Each of these is written in a separate python file, for example `hiv_status.py` or `sexual_behaviour.py`. Some modules will also have an additional python file to handle loading data from the config, such as `sex_behaviour_data.py`. 

> Data loading modules exist in order to convert the configuration file into more useful objects. For example, many parameters need to be sampled over, and the config file provides both values and probabilities. The data module will read the config file and turn this data into a probability distribution object. The main module can then use this distribution to sample a random variable to determine what its value will actually be for this simulation. 

Just like `Population`, the `HIVpy` modules are _classes_, and each contains an `__init__` function which runs when the object is created, as well as defining many internal variables and functions which are relevant to this particular area of the code. 

Variables which are not specific to an individual person should general go into these modules rather than columns in `Population`: for example `SexualBehaviourModule` contains variables like the base rate for starting sex work, the sex behaviour group transition matrices, age mixing matrices, age limits on sexual behaviour, and so on. These variables are the same for everyone and so can just be stored in the module as ordinary variables without the need to create columns for them in the population data.

The population object is usually passed to functions in these modules so that these functions can change the state of the population, e.g. `SexualBehaviourModule` has a function to calculate the number of sexual partners for the sexually active population, and then write these values into the population data. These functions have access both to the population data and the parameters within the module itself. 

### The main loop

In `simulation.py` there is a function `run` which contains the main loop of the program. 

```python
while date <= self.simulation_config.stop_date:
            logging.info("Timestep %s\n", date)
            # Advance the population
            self.population = self.population.evolve(time_step)
            self.output.update_summary_stats(date, self.population.data)
            date = date + time_step
```
This just says:
- While we have yet to reach the end date of the simulation:
    - log some info (in this case the number of the timestep and date)
    - Evolve the population: this is all the updates that happen to the population during this timestep, such as calculating the number of partners, HIV transmission, births, advancing age etc. 
    - Update output statistics: this calculates the outputs that we are interested in for this timestep
    - Update the date by the timestep

The part we are most interested in is the `population.evolve` function, which is where we define all the steps which go into updating the simulation for one timestep. You can find this function inside `Population.py` because it is part of the population class. The following steps are executed by `evolve`:
- Ages are increased
- Deaths are determined
- Update calls are made to modules including:
    - circumcision
    - sexual behaviour
    - HIV status
    - HIV testing
    - Pregnancy
    - Some modules will be dependent on others e.g. we must make the call to the HIV status module _after_ sexual behaviour, because we need up to date information about sexual partners! 
- HIV is introduced if necessary

**Usually changes that you make to the main loop (additionals, removals, or reorderings) will happen inside `evolve`.**

## Making a Module

With that overview out of the way, let's take a look at how we actually _add a new feature_ to the `HIVpy` code! 

For this simple first example, let's make a toy model for PrEP usage. Our model will involve the following:

- People can either be on PrEP or not, so we will need a boolean (true/false) variable similar such as `on_PrEP`. 
- We'll have a base rate of going on PrEP which we'll call `prep_uptake_rate`. This is the probability of going on prep in a given timestep. 
- To introduce some dependence on other variables we'll also have a `prep_post_test_factor` which multiplies the probability of going on PrEP if a person has just had a _negative_ HIV test this timestep. 

This doesn't model people going _off_ PrEP, or missing their dose etc., but all these other features could be implemented using the same techniques that we'll show here. 

To make a module we should **add a new file** for that module. We'll call this `prep.py`. There's a little bit of standard boiler-plate code that each of our modules has, so let's just copy that into our new module:

```python=
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .population import Population

import operator as op

import hivpy.column_names as col

from .common import SexType, rng


class PrepModule:
    def __init__(self, **kwargs):
        return
```
This does the following:
- Lines 1-6 are just there to make our lives easier! This allows us to use the type definitions when we declare a new function and the editor will be able to use that type information to do things like auto-complete function names and display function definitions and what arguments are required as we're typing, which is extremely helpful when using complex objects. 
- Lines 8-12 are just common imports, you may not need all of these but in my experience you generally do! `.common` in line 12 refers to `common.py` which defines very useful things which are used throughout the code such as the random number generator `rng`. 
- Imports which use `import ... as ...` just allow you to give shorter names to your imports so that you can keep your code a bit more compact. 
- We define the `PrepModule` class in line 15, and our class should usually have an `__init__` function. (`__init__` isn't mandatory in python, you only need one if your class needs to execute something when the object is actually created in order to make it usable. For us that is generally the case though, since we generally need to set up a module by sampling variables and so on.)

Let's take a look at our specification again and think about what we should implement inside this module:
> - People can either be on PrEP or not, so we will need a boolean (true/false) variable similar such as `on_PrEP`. 
> - We'll have a base rate of going on PrEP which we'll call `prep_uptake_rate`. This is the probability of going on prep in a given timestep. 
> - To introduce some dependence on other variables we'll also have a `prep_post_test_factor` which multiplies the probability of going on PrEP if a person has just had a _negative_ HIV test this timestep. 

We'll start with the simplest thing we can do, which is to add the variables `prep_uptake_rate` and `prep_post_test_factor` into the PrEP module. 

In Python, code is grouped by _indentation_. Everything which is part of the class needs to be indented relative to the declaration of the class name. The code has left the class's scope when we the indentation goes back. For example, in the following code sample:

```python=
class MyClass:
    x = 10

y = 5
```
`x` is part of `MyClass`, but `y` is not. The same rule applies for things like functions. Your editor will indent your code for you when you declare a class or function, but you'll have to un-indent at the to make sure so that it knows where the scope finishes! 

To add in our two variables we can modify our class:
```python=
class PrepModule:
    def __init__(self, **kwargs):
        self.prep_uptake_rate = 0.05
        self.prep_post_test_factor = 5
        return  # return statement is optional here
```
- In this case we've added the variables in the `__init__` function. You can actually add variables anywhere in the class (inside functions or outside) _but_ we'll normally want to add variables here because most will be initialised from data files, which means they'll be dependent on some other code before they can be initialised. The `__init__` function allows us to ensure that we load the necessary data before creating and assigning our variables, but we'll look at data later!

## Adding Variables to the Population

Next we need to add a variable to the population data table in order to keep track of who is on PrEP. There are a few steps to doing this. 

1. Defining the column name.
    - In `column_names.py` we define all of the column names used in `HIVpy`. The column names are all stored in string variables: this is to help prevent typos, allow for auto-completion and other helpful features, and make name changes straight-forward if necessary. It also allows us to automatically create multiple columns for any variables which need to be stored at more than one time step at a time, as each column will need a different name. Each line of `column_names.py` defines a string variable and contains a comment describing the type and meaning of that column. 
        - For example: `AGE = "age"  # float: age at the current date`. This tells us that the variable `AGE` represents the string `"age"`, which is the name of a column representing the age of a person as a floating point (i.e. fractional rather than whole) number.
    - To add a new variable we start by adding a new line to this file! 

```python=
ON_PREP = "on_PrEP"                             # Bool: true if person is taking PrEP, false otherwise.
```

2. Initialising the variable for the population object. 
    - This step can be undertaken either in the population object itself (inside `_create_population_data`) _or_ inside `PrepModule` in a function which will be called from `_create_population_data`. **The latter option is preferred as it prevents `_create_population_data_` from becoming too long and confusing.**
    - To do this we create a new function in `PrepModule` which takes an object of type `Population` as an argument. 
        - This function should therefore take two variables, `self`, and a `Population` type object. We can denote that one of the arguments should have `Population` type as follows: `def init_prep_variables(self, pop: Population)`. This declaration tells the editor that `pop` has type `Population`. This will allow the editor to show you helpful information when you try to use the object in your code! 
        - Inside this function we then need to call `init_variable` on the population to create a new variable in the table with some initial value(s) for everyone. We'll discuss how to do this in detail below. 
3. Linking things together.
    - Finally inside `Population.py` we need to do two things:
        - Create a `PrepModule` type object in the `__init__` function.
        - Call our function to initialise the prep module variables in `_creat_population_data`.

Let's talk a bit about `init_variable`. This is a member function of the `Population` class, so we call it using the `.` notation that we have seen before. 

```python=
class PrepModule:
    def __init__(self, **kwargs):
        self.prep_uptake_rate = 0.05
        self.prep_post_test_factor = 5
        return
    
    def init_prep_variables(self, pop: Population):
        pop.init_variable(col.ON_PREP, False)
```
`init_variable` takes the following arguments (ignoring `self` which does not need to be provided as discussed above):
1. `name`: The name of the column we want to create. In this case we use `col.ON_PREP`, because `ON_PREP` is the name that we created in the `column_names.py` file, and we imported that file as `col` (see boiler plate code discussion above). 
2. `init_val`: The initial value. This can either be a single value which gets applied to everyone (in this case just `False`) or a list of the the same length as the population, which then becomes the initial values of that column. This is useful when initialising things like random values, such as the "hard to reach" variable which should only be true for a certain fraction of the population. Here we've just done `False` because at the start of the simluation, no one is on PrEP! 
3. **Optional** `n_prev_steps`: Number of previous steps, default is 0. If you set this to be non-zero, more than one column is created and the variables will persist for the number of time steps specified. 
4. **Optional** `data_type`: The type of variable stored in the column. This is typically inferred by the library given the data provided; in this case, because the initial value given is `False`, the library can deduce that the variable must be a boolean. However, in some cases this can be more of an issue, so if you have an **integer** (i.e. whole number) type, you should specify the type as **`data_type=pd.Int32Dtype`**.

If you initialise your variable inside the `PrepModule`, make sure that you make a call to `init_prep_variables` from inside `_create_population_data`. You'll also need to add the `PrepModule` into the `Population` in `__init__` (we'll see more about that later). 

## Reading / Writing Variables in the Population

### Reading

Generally when we read variables we don't do it one at a time, but we request columns from the population. We can then process these columns in a vectorised way, if we wish. Reading columns will be particularly useful for creating outputs. To see, for example, who is on PrEP, we can write 

```python=
is_on_prep = pop.get_variable(col.ON_PREP)
```
This will give us the entire column of `True`/`False` values for people being on PrEP. We can then process this data, for example by summing this column to see how many people are on PrEP. 
```python=
# In a sum, True counts as 1 and False counts as 0
num_on_prep = sum(is_on_prep)
```

Similarly to `init_variable`, `get_variable` has some optional arguments. The full list is:
1. `var`: the column we want to get. This in our case is `col.ON_PREP` for the same reasons as above. 
2. **Optional** `sub_pop`: The sub-population we are looking at, default is `None`. When there is no sub-population, we look at everyone defined in the population table. When there is a sub-population defined (see the next section for how to do that), we only look at the part of the population meeting certain conditions, e.g. betwen 15 and 65, or not being dead. 
3. **Optional** `dt`: time difference, default 0. By default it gets the variable as it is currently defined in this time step. If the variable is stored for more than one time step, then you can specify the number of timesteps you want to look back (e.g. `dt=1` will look at the previous timestep) and it will access the correct column for you.

### Writing

Writing variable works almost identically, except that you can only write to the present value of the variable, not to its value in the past. The function is called `set_present_variable`, and takes the following arguments:
1. `target`: name of column you want to write to. 
2. `value`: A single value will assign everyone that value, or it can be a list of values which needs to be the same length as the population / sub-population under consideration. 
3. **Optional** `sub_pop`, default `None`. As above, `None` means apply to the entire population, otherwise apply only to the sub population given. 

As an example, we might want to assign the value of the `ON_PREP` randomly with the probability `prob_on_prep`. In this case we want to assign `ON_PREP` using a _list_ of values, and generate that list of True and False values randomly. 
```python=
# generate a random list of the size of the population.
# Use the random number generator (rng) we imported.
# Uniform will be uniform between 0 and 1
random_list = rng.uniform(size=pop.size)

new_prep_status = random_list < prob_on_prep

pop.set_variable(col.ON_PREP, new_prep_status)
```

## Sub-Populations 

Getting sub-populations is a key part of managing the data in `HIVpy`. Many things only apply to subsections of the population, based on age, sex, or other factors, and using sub-populations allows us to ensure that only they are affected by these parts of the code _and_ that we don't waste time doing calculations for people that we shouldn't. 

> **N.B. We will ignore checks for if people are dead here because there will be some changes to default behaviour coming soon, where operating on the not-dead will be the default!** 

In order to read or write to a sub-population, we need to first use `get_sub_pop` to find that sub-population. (There are some other methods that we can use to get sub-populations from arrays or intersections of other sub-populations, but we'll stick with the simplest and most common method for now.)

`get_sub_pop` is part of the `Population` class, so we call it on a `Population` type object using `.` just like `init_variable` and `get_variable`. It takes a list of conditions as its argument. In python, a list is written between square brackets `[]`. 

> **N.B. Updated syntax to make conditions clearer is coming soon, but proof of concept has been developed and works!**

The list of conditions is written in [conjunctive normal form](https://en.wikipedia.org/wiki/Conjunctive_normal_form). Each item in the list is either a single condition, or a _disjunction_ (list of _or_ statements). **We'll stick to simple conditions for now, with no "_or_" statements.** The sub-population is the list of people for whom all the conditions are true. 

Conditions are written as a _triple_. A triple is three values, separated by commas, in round brackets `()`. They use the following format: **(column name, operator, value)**. For example, `(col.AGE, op.ge, 15)` is the condition that age is greater than or equal to 15. (`op` here is from the `import operator as op` line at the top of the file.) Common operators are:
- `op.eq`: Equal to
- `op.ne`: Not equal to
- `op.gt`: Greater than
- `op.ge`: Greater than or equal to
- `op.lt`: Less than
- `op.le`: Less than or equal to
You can also define your own functions rather than these operators, as long as they take the appropriate types and return a `Bool`! 

So to express the condition that someone is at least 15, under 65, and not on PrEP we need a list that looks like:
```python=
[(col.AGE, op.ge, 15),
 (col.AGE, op.lt, 65),
 (col.ON_PREP, op.eq, False)]
```
- Formatting like this can make your lists of conditions easier to read!

So in order to get the sub-population for people who might start PrEP, we could write:

```python=
prep_eligible = pop.get_sub_pop([(col.AGE, op.ge, 15),
                                 (col.AGE, op.lt, 65),
                                 (col.ON_PREP, op.eq, False)])
```
The variable `prep_eligible` now contains the indexing for all the people who are between 15 and 65 and not currently on PrEP, which will allow us to only apply our functions to that group of people. 

## A Simple Update Function 

Let's make use of what we've seen so far by just applying the basic rate of prep uptake to the eligible population. 

```python=
    def update_prep_status(self, pop: Population):
        # Get sub population who might start PrEP
        prep_eligible = pop.get_sub_pop([(col.AGE, op.ge, 15),
                                         (col.AGE, op.lt, 65),
                                         (col.ON_PREP, op.eq, False)])
        
        # Length of sub population is number of eligible people
        num_eligible = len(prep_eligible)
        
        # Randomly assign new uptake by generating a list of random numbers 
        # the size of the sub-population and comparing with the probability
        new_users = rng.uniform(size=num_eligible) < self.prep_uptake_rate
        
        # Set the new value of the variable
        pop.set_present_variable(col.ON_PREP, new_users, prep_eligible)
```

## Applying Functions By Groups

There are a few ways that we can introduce dependence on other population variables into our update function. We can go through each person and get their entire row in the table, and apply a function for each person individually: this would be the `apply_function` method in the `Population` object. But this means calling that function once for everyone in the sub-population that we are looking at, which could be a lot of people! The most common is to use `set_variable_by_group`, which groups people in the population together based on some variables they share in common, and then calculates results for each group. This is very useful for things like calculating functions which are different for people of different sexes, or different age groups, since this usually leads to a lot of people being grouped together and so fewer function calls. 

> **N.B.** When we use this group method it doesn't mean that everyone in one group has to be assigned the same result: as we've seen above we can generate a list of random variables and set our population data using lists. Applying functions by group is really about changing the _logic_ that is applied to different groups, for example if men and women have different considerations and probabilities and so can't have their results calculated using the same expression. Generating lists of results in this way is more efficient than going through the population one by one and calling a function which assigns the random variable each time. 

In our toy PrEP model, we want to split the eligible population into two groups: those who have just been tested, and those who haven't. Those who've just been tested will have a boosted likelihood of starting PrEP. 

So now we need a function which also takes this new argument. Because this function is only needed by `update_prep_status`, it can be defined within the scope of `update_prep_status`, which makes this fact clearer. 

```python=
    def update_prep_status(self, pop: Population):
        
        # This is a helper function which is used to generate the
        # new prep users based on whether they have been recently 
        # tested or not. It also takes a variable "size" in order
        # to know how many random numbers to generate.
        def new_prep_uptake(recently_tested, size):
            prob_prep = self.prep_uptake_rate
            if (recently_tested):
                prob_prep *= self.prep_post_test_factor
            new_users = rng.uniform(size=size) < prob_prep
            return new_users
        
        prep_eligible = pop.get_sub_pop([(col.AGE, op.ge, 15),
                                         (col.AGE, op.lt, 65),
                                         (col.ON_PREP, op.eq, False)])
        
        # Now we need to use the set_variable_by_group function
```

> **N.B.** For this example I have added in an additional column `RECENTLY_TESTED`, in the same way we added in the `ON_PREP` column. This is better than grouping by the existing variable `LAST_TEST_DATE` since it produces only two groups instead of many. You could also do this by creating two sub-populations instead of a new variable, and applying different logic to each sub-population. 

The `set_variable_by_group` function, also a member of `Population`, takes the following arguments:
1. `target`: the column that you want to change. In our case, `col.ON_PREP`
2. `groups`: a list of column names. These are the columns that people will be grouped by: anyone with the same values in these columns will be grouped together. For example, if we have `[col.SEX, col.ON_PREP]` then we get four groups: men on PrEP, men not on PrEP, women on PrEP, womeon not on PrEP. It's not a good idea to group by variables where you don't expect people to share values: for example _age_ is a floating point number and almost everyone will have a slightly different age to another person. In order to group people by things like age, we introduce age group variables which convert wider ranges of ages to indices, or use conidtions in sub-populations. In our case, we just group by `[col.RECENTLY_TESTED]` since we don't have any other components in this model. 
3. `func`: the function we want to apply. The arguments for this function **must** match the list of variables we are grouping by, with the optional addition of `size`. So in our case, this function can only depend on whether or not someone is on PrEP, and the size of the group of people who are / are not on PrEP. Like all member functions, the function still has access to any variables that are defined in the `PrepModule` though!
4. `use_size`: whether we should pass the size of the group to the function, default is True. If everyone in the group will receive an identical value, then you don't need to pass the size of the group. But if you need to generate a list of results, for example for randomised functions where people in the same group have different outcomes, then you need `use_size` to be true (which is the default, as this is more common), and your function (`func`) needs an extra argument for the size at the end. 
5. `sub_pop`: subpopulation that this should apply to, default `None` i.e. the whole population. In our case we only want to apply this to people who might take up PrEP i.e. sexually active people not currently on PrEP. 

So using this method we can write the following update function:

```python=
def update_prep_status(self, pop: Population):
        def new_prep_uptake(recently_tested, size):
            prob_prep = self.prep_uptake_rate
            if (recently_tested):
                prob_prep *= self.prep_post_test_factor
            new_users = rng.uniform(size=size) < prob_prep
            return new_users
        
        # Set recently tested to true if last test date is now; this would normally be in the testing module
        pop.set_present_variable(col.RECENTLY_TESTED,
                                 pop.get_variable(col.LAST_TEST_DATE)==pop.date)
        
        prep_eligible = pop.get_sub_pop([(col.AGE, op.ge, 15),
                                         (col.AGE, op.lt, 65),
                                         (col.ON_PREP, op.eq, False)])
        
        pop.set_variable_by_group(col.ON_PREP,
                                  [col.RECENTLY_TESTED],
                                  new_prep_uptake,
                                  sub_pop=prep_eligible)
```

There is a lot of structure going on in here, so don't worry if it takes a little getting used to. The core logic is in `new_prep_uptake`, and the rest (`get_sub_pop` and `set_variable_by_group`) is used to organise to whom the function gets applied. 

## Adding Functionality to the Main Loop

Now that we have a toy model for PrEP working, we need to actually make sure that it executes! This means we need to add a function call for `update_prep_status` in the `evolve` function in the `Population`. 

First, make sure the population has a `PrepModule` component. We need to add the following import to `population.py` to make it available:

```python=
from .prep import PrepModule
```

and then add the following to `__init__`:
```python=
        self.prep_module = PrepModule()
```

and finally add to `evolve`:
```python=
        self.prep_module.update_prep_status(self)
```

It is vital that this line is added _after_ the call to `update_hiv_testing` so that we have up to date testing information! We pass `self` into `update_prep_status` because, remember, `update_prep_status` takes the population as an argument in order to have access to all the population data. 

## Output

We can add an output to the current structure in `simulation.py`. This file contains a class called `SimulationOutput`. In order to update the output we need to modify `update_summary_stats`. 

Summary statistics are kept in a table indexed on names and timesteps, for example 
```python
self.output_stats["Population (over 15)"][self.step]
```

**This section needs to be updated once we have changed the output module to work using the new access functions.** 