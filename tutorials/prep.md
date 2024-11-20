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

### PrEP Data Variables

- *`prep_strategy`* - An integer that determines which sub-population of people is marked as eligible for PrEP.
- *`date_prep_intro`* - An array containing the introduction dates for each type of PrEP. Intended to be accessed through the use of `PrEPType`s as indices (e.g. `date_prep_intro[PrEPType.Oral]`).
- *`cab_available`* - The boolean flag that determines the availability of `Cabotegravir` PrEP. Cab is only available if the current date has reached the cab introduction date and this flag is True.
- *`prob_risk_informed_prep`* - The probability of an individual with an *uninfected* long-term partner who is not on ART to meet the risk-informed PrEP eligibility criteria.
- *`prob_greater_risk_informed_prep`* - As `prob_risk_informed_prep`, but with a higher probability value. Used with certain `prep_strategy` values.
- *`prob_suspect_risk_prep`* - The probability of an individual with an *infected* long-term partner who is not on ART to meet the risk-informed PrEP eligibility criteria.
- *`prep_oral_pref_beta`* - The alpha value used to draw random oral PrEP preference values for the population from a beta distribution.
- *`prep_cab_pref_beta`* - The alpha value used to draw random cabotegravir PrEP preference values for the population from a beta distribution.
- *`prep_len_pref_beta`* - The alpha value used to draw random lenacapavir PrEP preference values for the population from a beta distribution.
- *`prep_vr_pref_beta`* - The alpha value used to draw random vaginal ring PrEP preference values for the population from a beta distribution.
- *`prep_willing_threshold`* - A threshold value that a PrEP preference value must exceed in order for an individual to be willing to take the respective type of PrEP.
- *`vl_prevalence_affects_prep`* - A boolean flag that dictates whether willingness to take PrEP is affected by low unsuppressed viral load prevalence in the population.
- *`vl_prevalence_prep_threshold`* - The threshold at which low unsuppressed viral load prevalence affects PrEP willingness.
- *`rate_test_onprep_any`* - The rate of being tested for HIV while on PrEP.
- *`prob_test_prep_start`* - The probability of being tested for HIV with the intent to start PrEP.
- *`prob_base_prep_start`* - The base probability of starting any type of PrEP for the first time.
- *`prob_oral_prep_start`* - The probability of starting oral PrEP for the first time.
- *`prob_cab_prep_start`* - The probability of starting cabotegravir PrEP for the first time.
- *`prob_len_prep_start`* - The probability of starting lenacapavir PrEP for the first time.
- *`prob_vr_prep_start`* - The probability of starting vaginal ring PrEP for the first time.
- *`prob_oral_prep_stop`* - The probability of choosing to stop taking oral PrEP despite being eligible.
- *`prob_cab_prep_stop`* - The probability of choosing to stop taking cabotegravir PrEP despite being eligible.
- *`prob_len_prep_stop`* - The probability of choosing to stop taking lenacapavir PrEP despite being eligible.
- *`prob_vr_prep_stop`* - The probability of choosing to stop taking vaginal ring PrEP despite being eligible.
- *`prob_prep_restart`* - The probability of restarting any type of PrEP after choosing to stop taking PrEP.
