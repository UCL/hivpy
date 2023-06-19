## Pregnancy Tutorial

Welcome to the pregnancy module. This code deals with fertility, pregnancy, antenatal care (ANC), prevention of mother to child transmission care (PMTCT), birth, and number of children. The most relevant files are listed below:

- `src/hivpy/pregnancy.py` - The pregnancy module.
- `src/tests/test_pregnancy.py` - Tests for the pregnancy module.
- `src/hivpy/data/pregnancy.yaml` - Pregnancy data and variables.
- `src/hivpy/pregnancy_data.py` - A class for storing data loaded from `pregnancy.yaml`.

If there are any pregnancy-related variables you would like to change before running your simulation, please change them in `pregnancy.yaml`.

### Pregnancy Data Variables

- *`can_be_pregnant`* - The proportion of women in a population that can become pregnant.
- *`rate_want_no_children`* - The rate at which women stop wanting children over time.
- *`date_pmtct`* - The year during which PMTCT begins (2004 by default).
- *`pmtct_inc_rate`* - The rate at which the probability of receiving PMTCT increases per year.
- *`fold_preg`* - A fertility factor that changes with age and affects pregnancy probability.
- *`inc_cat`* - A demographic variable that affects population growth and pregnancy probability.
- *`prob_pregnancy_base`* - The base probability of pregnancy (calculated when the pregnancy module is initialised).
- *`rate_testanc_inc`* - An additive modifier to `prob_anc`.
- *`rate_birth_with_infected_child`* - The probability for an infected mother to pass HIV on to their child.
- *`max_children`* - The maximum number of children someone can have.
- *`init_num_children_distributions`* - The probability distributions used to initialise some starting number of children for each woman.
- *`prob_anc`* - The probability of ANC attendance (calculated anew at each time step).
- *`prob_pmtct`* - The probability of receiving PMTCT (calculated anew at each time step).
