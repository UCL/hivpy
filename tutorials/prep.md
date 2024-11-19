## PrEP Tutorial

The PrEP module tracks PrEP preferences, willingness, eligibility, and use, including starting, stopping, and restarting PrEP. The most relevant files are listed below:

- `src/hivpy/prep.py` - The PrEP module.
- `src/tests/test_prep.py` - Tests for the PrEP module.
- `src/hivpy/data/prep.yaml` - PrEP data and variables.
- `src/hivpy/prep_data.py` - A class for storing data loaded from `prep.yaml`.

If there are any testing-related variables you would like to change before running your simulation, please change them in `prep.yaml`.

### Module Overview

When PrEP is updated, the first thing to be calculated is PrEP propensity for the population aged 15 and over. For each individual, this includes: setting preference values for each type of PrEP, determining which types of PrEP they are willing to take, setting preference ranks for each type of PrEP, and calculating their 'favoured PrEP' – the type of PrEP with the highest preference value someone is willing to take that is also currently available.

Next, PrEP eligibility is determined. Based on the current PrEP strategy, a sub-population of people is selected to be eligible for PrEP usage this time step. Eligible people typically either have short-term partners or have a long-term partner who is not on ART.

Finally, PrEP usage is updated for anyone starting, continuing, stopping, or restarting PrEP. (`Note`: PrEP usage relies on the assumption that HIV diagnosis has already taken place in order to identify people that are HIV positive but have falsely not been diagnosed.)

Individuals can be specifically tested to start PrEP for the first time, but tested people in the general population can also decide to start PrEP. When people continue PrEP usage, they can either continue with their current PrEP or switch to a different type if their favoured PrEP has changed. PrEP usage can be stopped for two reasons – an individual can choose to stop, or they can become ineligible for PrEP. People who have chosen to stop taking PrEP but are still eligible can also choose to restart, but anyone who stopped taking PrEP due to a break in eligibility automatically restarts.
