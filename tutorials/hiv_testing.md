## HIV Testing Tutorial

The HIV testing module tracks HIV symptomatic, non-HIV symptomatic, voluntary medical male circumcision (VMMC), antenatal care (ANC), and general testing in the population. The most relevant files are listed below:

- `src/hivpy/hiv_testing.py` - The HIV testing module.
- `src/tests/test_hiv_testing.py` - Tests for the HIV testing module.
- `src/hivpy/data/hiv_testing.yaml` - HIV testing data and variables.
- `src/hivpy/hiv_testing_data.py` - A class for storing data loaded from `hiv_testing.yaml`.

If there are any testing-related variables you would like to change before running your simulation, please change them in `hiv_testing.yaml`.

### Module Overview

When HIV testing is updated, people are marked for testing each time step, starting with the symptomatic. First, those exhibiting HIV symptoms have a chance to be scheduled for testing. New testing probabilities are calculated for each class of symptoms (WHO4, TB, non-TB WHO3) each time step until 2015. Note: If the simulation is started after 2015 (or if `date_start_testing` is set to be after 2015) these testing probabilities will not be updated and the code may not work as expected. Second, people with non-HIV symptoms have a chance to be scheduled for testing, borrowing WHO4 and non-TB WHO3 testing probabilities from the previous step, if their symptoms are concerning enough to warrant a test.

Next, men undergoing VMMC this time step are tested and women in ANC this time step have a chance to be tested depending on which trimester they are in or whether they have given birth.

Finally, the general population has a chance to be tested (assuming no COVID disruption). As part of this, sex workers are also scheduled for testing if sex workers regularly test every 6 months. General testing probabilities change based on whether someone is a first time or repeat tester and these probabilities increase each time step.

At the end, everyone scheduled to be tested has their testing information updated, and their test mark is removed.

### HIV Testing Data Variables

- *`date_start_testing`* - The year during which HIV testing begins (2009 by default).
- *`init_rate_first_test`* - The initial testing probability for first time testers.
- *`eff_max_freq_testing`* - An integer between 0-2 used to select the minimum number of days to wait between tests (365, 180, or 90).
- *`test_scenario`* - An integer that affects test targeting. Scenario 0 is the default behaviour, while scenario 1 increases effective test targeting.
- *`no_test_if_np0`* - A boolean that, if True, means people with no partners don't participate in general population testing.
- *`prob_anc_test_trim1`, `prob_anc_test_trim2`, `prob_anc_test_trim3`* - The probabilities of getting tested in antenatal care at the end of each trimester.
- *`prob_test_postdel`* - The probability of being tested after labour and delivery.
- *`prob_test_non_hiv_symptoms`* - The probability of being tested when exhibiting non-HIV symptoms.
- *`prob_test_who4`* - The probability of being tested when exhibiting WHO4 symptoms.
- *`prob_test_tb`* - The probability of being tested when exhibiting tuberculosis symptoms.
- *`prob_test_non_tb_who3`* - The probability of being tested when exhibiting WHO3 symptoms (excluding tuberculosis).
- *`test_targeting`* - A factor that affects an individual's general population testing probability based on their number of partners (multiplicative).
- *`date_test_rate_plateau`* - The year during which general population testing probability plateaus.
- *`an_lin_incr_test`* - The rate at which general population testing probability increases each time step (multiplicative).
- *`incr_test_rate_sympt`* - The rate at which HIV symptomatic testing probability increases each time step before 2015 (multiplicative).
