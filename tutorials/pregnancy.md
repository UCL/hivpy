## Pregnancy Tutorial

Welcome to the pregnancy module. This code deals with fertility, pregnancy, antenatal care (ANC), prevention of mother to child transmission care (PMTCT), birth, and number of children. The most relevant files are listed below:

- `src/hivpy/pregnancy.py` - The pregnancy module.
- `src/tests/test_pregnancy.py` - Tests for the pregnancy module.
- `src/hivpy/data/pregnancy.yaml` - Pregnancy data and variables.
- `src/hivpy/pregnancy_data.py` - A class for storing data loaded from `pregnancy.yaml`.

If there are any pregnancy-related variables you would like to change before running your simulation, please change them in `pregnancy.yaml`.

### Module Overview

When pregnancy is updated, the eligible population is grouped by age, and pregnancy outcomes are probabilistically determined for the current time step. Pregnancy probabilities are different for each age group due to the influence of an age-based fertility factor, but an individual's number of condomless sex partners and wanting no more children also affects pregnancy probability.

Once pregnancy outcomes are determined, they are assigned to the newly pregnant sub-population and their last pregnancy date is set to the date of the current time step.

Then, newly pregnant women have a chance to enter antenatal care, and thus possibly undergo ANC testing or PMTCT. There is a chance for an ANC test to be carried out at the end of each trimester, as well as one time step after delivery.

Next, the sub-population at the end of their pregnancy gives birth and their number of children are updated. Pregnancies that are not carried to term are not modelled, and newborn children do not become part of the population tracked by the model.

Finally, older women have a chance to stop wanting any more children, which significantly decreases their probability of pregnancy for the rest of their lifetime.

### Pregnancy Data Variables

- *`can_be_pregnant`* - The proportion of women in a population that can become pregnant.
- *`rate_want_no_children`* - The rate at which women stop wanting children over time.
- *`date_pmtct`* - The year during which PMTCT begins (2004 by default).
- *`pmtct_inc_rate`* - The rate at which *`prob_pmtct`* increases per year (multiplicative).
- *`fertility_factor`* - A fertility factor that changes with age and affects pregnancy probability (multiplicative).
- *`inc_cat`* - A demographic variable that affects population growth and pregnancy probability.
- *`prob_pregnancy_base`* - The base probability of pregnancy (calculated when the pregnancy module is initialised).
- *`rate_test_anc_inc`* - The rate at which *`prob_anc`* increases per time step (additive).
- *`prob_birth_with_infected_child`* - The probability for an infected mother to pass HIV on to their child.
- *`max_children`* - The maximum number of children someone can have.
- *`init_num_children_distributions`* - The probability distributions used to initialise some starting number of children for each woman.
- *`prob_anc`* - The probability of ANC attendance (calculated anew at each time step).
- *`prob_pmtct`* - The probability of receiving PMTCT (calculated anew at each time step).
