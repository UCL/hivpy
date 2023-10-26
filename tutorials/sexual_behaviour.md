# Sexual Behaviour Tutorial

The sexual behaviour module tracks the sexual characteristics and interactions of an individual, including:

- Number of (condomless) short term partners in each time step
- Beginning and ending of (condomless) long term partnership 
- Individual propensity for condomless sexual relationships
- Factors associated with risky sexual behaviour
- Engagement in sex work (female only)

As one can see there's quite a bit covered here, so sexual behaviour is one of the largest `hivpy` modules. 

## Module Overview and Initialisation 

- The module initialisation is laid out in the `__init__` function. This is automatically called as soon as the object is created. 

    The module is initialised using the `sex_behaviour.yaml` file in the `data` folder. This contains a wide variety of values that we will discuss in the relevant sub-sections below for the sake of clarity. Parameters in the data file can be assigned fixed values, vectors, matrices, or probability distributions which are sampled. 

    After the data has been loaded and converted into the appropriate forms by the `SexualBehaviourData` module, the `sexual_behaviour` module will get the final values by sampling any relevant probability distributions. This is so that, when running an ensemble of simluations, one can run simulations with varying properties while only reading the data file once. In this case we have just one `SexualBehaviourData` module, and each individual simulation, with its own `sexual_behaviour` module, can sample their own random values. 

- `init_sex_behaviour` is an additional function which needs to be called on the module. This adds the sexual behaviour related variables to the population, and calls a number of other sub functions to initialise parts of the code such as risk factors, initial sex worker status, sexual behaviour classes and groups, and number of short term partners at the beginning of the simulation. This is called after the construction of the class, as the initialisation depends on the class having already sampled all its variables, and some functions may depend on modules having been able to update the population as well. 

    The variables initialised in the population by this module are:
    - `NUM_PARTNERS`: The number of short term partners a person has this time step.
    - `RISK`: Overall risk calculated from all other factors.
    - `LONG_TERM_PARTNER`: Where a person has a long term condomless partner (True/False).
    - `LT_AGE_GROUP`: Age group of their long term partner. 
    - `LTP_LONGEVITY`: Personal variable associated with probability of ending long term partnership.
    - `SEX_MIX_AGE_GROUP`: A person's age group for the purposes of sexual mixing given the age sexual mixing matrices. 
    - `STP_AGE_GROUPS`: Age group for each of the person's short term partners. 
    - `RISK_LTP`: Risk factor associated with long term partnership. 
    - `ADC`: Whether a person has an AIDS Defining Condition 
    - `LIFE_SEX_RISK`: Risk category for engaging in sex work (value betweeen 1 and 3).
    - `SEX_WORKER`: Whether a person is engaged in sex work at this time step (True/False)
    - `SW_TEST_6MONTHLY`: Sex worker engagement in six monthly HIV testing (True/False)
    - `AGE_STOP_SEX_WORK`: Age at which a person last stopped sex work 
    - `SW_AGE_GROUP`: Age grouping for the purpose of sex worker logic 
    - `DATE_STOP_SEX_WORK`: Date at which a person last stopped sex work
    - `SEX_BEHAVIOUR_CLASS`: Classification for the purposes of sexual behaviour. Each class has a number of sexual behaviour groups which determine the probability of having different numbers of short term partners. The classes are:
        - Men
        - Women under 25 (not engaged in sex work)
        - Women over 25 (not engaged in sex work)
        - Sex workers under 30
        - Sex workers over 30
    - `SEX_BEHAVIOUR`: sexual behaviour group within a given class. For men there are 4 groups (0...3), for women who are not sex workers there are 2 (0,1), and for women who are sex workers there are 5 (0...4). 

- `update_sex_behaviour` is run at every time step to update all sexual behaviour related properties of the population e.g. calculating new partners, updating risk, and starting or stopping sex work. To keep the code more modular this calls the following sub-functions:
    - `update_sex_worker_status`
    - `update_sex_behaviour_class` (to take into account changes to sex worker status and age for women)
    - `update_risk`
    - `update_sex_groups` (determined by sexual behaviour transition matrices, see below)
    - `num_short_term_partners`
    - `assign_stp_ages`
    - `update_long_term_partners`

The simulation code run each timestep is organised into subsections for:

1. Sex Work
2. Short term partners 
3. Risk factors 
4. Long term partners

### Sex Work

Women between the ages of 15 and 50 have a chance of engaging in sex work, which will affect their number of partners and their risk factors. Current sex workers may also cease sex work, or engage with sex worker programs which can mitigate risk. 

- `init_sex_worker_status`: Initialises the sex workers present at the start of the population. These are all women with $15 \le \text{age} \lt 50$. The probability of being an initial sex worker varies by age according to `prob_init_sex_work_age` in the data file. The logic is:
    - $P(\text{SW}) = 0$ if `LIFE_SEX_RISK` is 1. 
    - $P(\text{SW}) = P_{sw}(a)$ for age $a$ if `LIFE_SEX_RISK` is 2.
    - $P(\text{SW}) = 3P_{sw}(a)$ for age $a$ is `LIFE_SEX_RISK` is 3. 
    - **N.B. This code implies that no value within `prob_init_sex_work_age` can exceed $\frac{1}{3}$.**
    - We also call `update_sex_worker_program` to calculate any initial enrolment in a sex worker program. 

- `update_sex_worker_status`: Changes sex worker status for those starting / stopping sex work, and records any extra information needed such as `DATE_STOP_SEX_WORK`. 
    - Starting sex work with probability $P$:
        - Depends on age group and `LIFE_SEX_RISK`
        - $P = 0$ if `LIFE_SEX_RISK` is 1
        - $P = P_{SW} = P_\text{base} \times \sqrt{R_\text{pop}} \times R_\text{SW}(a) $ if `LIFE_SEX_RISK` is 2, where:
            - $P_\text{base}$ is base probability of starting sex work, set by `base_rate_start_sex_work` in data file and sampled. 
            - $R_\text{pop}$ is population wide risk factor. (See Risk Factors section below.)
            - $R_\text{SW}$ is age dependent sex work risk factor, set by `risk_sex_worker_age` in data file. 
        - $P = 3P_{SW}$ if `LIFE_SEX_RISK` is 3.
    - Continuing sex work (for existing sex workers) with probability $P$:
        - $P = 0$ if over 50.
        - $P = P_\text{stop} = P_\text{base} \times {R_\text{pop}^{-\frac{1}{2}}}$ if under 40, where 
            - $P_\text{base}$ is set by `base_rate_stop_sex_work` in the data file and sampled. 
            - $R_\text{pop}$ is population level risk factor. 
        - $P = 3P_\text{stop}$ if over 40. 
    - Upon stopping sex work the following takes place:
        - Variables related to sex work programs / testing are set to false
        - `DATE_STOP_SW` and `AGE_STOP_SEX_WORK` are set 
        - `SEX_BEHAVIOUR` group is set to 1, i.e. the higher sex behaviour group for women who are not sex workers. 
    - `update_sex_worker_program` is called to update related variables:
        - For a given simulation, there is a 20% chance of a sex worker program running. 
        - Sex worker program start date is taken to be the 01/01/2015. 
        - If the program exists and it is later than the start date:
            - Probability of newly engaging with the program for sex workers not in a program is 0.1 (10%)
            - Probability of disengaging with a program is 0.025 (2.5%)
        - **Details about the program are currently hard coded but should be moved to the data file for more flexible configuration, especially as these probabilities will be dependent on the time step.**

### Short Term Partners

Short term partners are condomless sex partners for this time step only. They are treated independently, with no continuation, and are not linked to other people in the model population. The population data is only used to calculate the demographic probabilities for short term partners, such as the probability of being HIV positive. 

- `init_sex_behaviour_groups` is used to give everyone in the population their initial sexual behaviour group. This group is used to determine the probability of each person having a particular number of sexual partners in a given timestep. For men, these groups are $[0,1,2,3]$, for women they are $[0,1]$ women not engaged in sex work and $[0,1,2,3,4]$ for sex workers.
    - The probability of beginning the simulation in each of these groups is given by `initial_sex_behaviour_probabilities` in the data file. 
- `prob_transition` gives the probability of transition between sexual behaviour groups, given ones sexual behaviour class and personal risk factors. 
    - This transition probability is given by the `sex_behaviour_transition_options` in the data file. 
    - Are are different options for `Male`, `Female`, and `Sex Worker`. 
    - These options are sampled during `__init__` so that one transition matrix for each category of people is used for a given simulation. 
    - These matrices are normalised by the code to avoid issues with probabilities! 
    - For a given transition matrix $T$, the probability of transitioning from group $i$ to $j$ is given by:
    
        $P(0 | i) = T_{i0} / N$

        $P(j | i) = r T_{ij} / N$
        
        Where:

        - $r$ is the personal risk factor
        - $N = T_{i0} + \sum_{j}{r T_{ij}}$ is the normalisation.
- `num_short_term_partners` is used to calculate the number of short term partners for sexually active people in the population. 
    - People are considered sexually active if $15 \ge \text{age} \lt 65$. 
    - We call `get_partners_for_group` which calculates short term partners on a group by group basis for different sexual behaviour groups. 
    - For a given behaviour group, the probability distributions of sexual partners for a given time step is given in `short_term_partner_distributions` in the data file. 
        - Sex workers over 30 have their number of sexual partners capped at 30. 
- `update_sex_groups` changes sexually active peoples sexual behaviour groups based on the probabilities described above. 
- `update_sex_behaviour_class` only needs to change the sex behaviour class of women, as there is only one category for men. For women, it may change if their sex worker status changes, or they move over an age threshold in this timestep.  

### Risk Factors 

There are a number of risk factors calculated for individuals and for the population. These risk factors may raise or lower risk behaviour, based on whether they are greater or less than 1. An individuals total personal risk is a product of risk factors. Most risk factors have both an `init_risk...` and `update_risk...` function to give initial and then subsequent values. The list of risk factors, and how to set them, is as follows:

- `RISK_PERSONAL`: An individual risk factor that is given to each person. The value `population_risk_personal` in the data gives possible probabilities of a person being designated low risk, and is sampled during the `__init__`. For each person, `RISK_PERSONAL` is set to $10^{-5}$ with this probability, and $1$ otherwise. This is not updated during the simulation. 
- `RISK_AGE`: Age related risk factor. Set this in `age_based_risk_options` in the data file; these options are sampled during `__init__` to give the age based risk factors for a given simulation.  
- `RISK_ADC`: Risk factor associated with reduction of risk behaviour for a person with an AIDS Defining Condition. If a person is HIV positive and has an ADC then this is 0.2; otherwise it is 1. 
- `RISK_BALANCE`: Behavioural changes imposed to balance out the number of male and female short term condomless sex partners if these become uneven. Calculates the number of short term partners of all men $S_m$ and of all women $S_w$. For a population of size $N$, we then calculate the normalised discrepancy $D = \frac{S_m + S_w}{N}$. 
    - The balance factor is set for different thresholds of $D$:
        - $D > 0.1 \implies R_b = 0.1$
        - $D > 0.03 \implies R_b = 0.7$
        - $D > 0.005 \implies R_b = 0.7$
        - $D > 0.004 \implies R_b = 0.75$
        - $D > 0.003 \implies R_b = 0.8$
        - $D > 0.002 \implies R_b = 0.9$
        - $D > 0.001 \implies R_b = 0.97$
        - otherwise $R_b = 1$
    - `RISK_BALANCE` is set to $R_b$ for the sex with the excess, and $R_b^{-1}$ for the other sex. 
- `RISK_DIAGNOSIS`: Change in risk behaviour associated with recent HIV diagnosis. Under `risk_diagnosis` in the data file you can set the values it can take, and the probabilities of that (which are samples at the `__init__` for the simulation), as well as the `Period` which is the length of time after an HIV diagnosis for which this effect lasts. For anyone without a recent HIV diagnosis, this is 1. 
- `RISK_LTP`: Change in risk behaviour associated with being in a long term partnership. Set the possible values for this in `risk_partnered`. Sampled during `__init__`. Set to 1 for anyone without a long term partner. 
- `RISK_ART_ADHERENCE`: Change in risk behaviour associated with low ART adherence. If a person is on ART and their adherence falls below the given threshold, then `RISK_ART_ADHERENCE` takes the value given in the data file, otherwise it is 1. This factor is only considered in a fraction of simulations. In the data file under `risk_art_adherence` you can set the `Value` of the risk factor, the `Adherence_Threshold`, and the `Probability` of this factor being included in a simulation.  
- `risk_population`: This is a population level risk. Unlike the other risk factors, which apply to individuals and are therefore stored as columns in the population data frame, this applies to the everyone and so is only stored as a single value in the sexual behaviour module itself. The value is:
    - $R_\text{pop} = 1$ before 1995
    - $R_\text{pop} = C_{95}^{y - 1995}$ between 1995 and 2000, where $C_{95}$ is the yearly risk change from 1995 and $y$ is the date in years. 
    - $R_\text{pop} = C_{95}^5$ from 2000 to 2010 (i.e. stops changing during this time).
    - $R_\text{pop} = C_{95}^5 \times C_{10}^{y - 2010}$ from 2010 to 2021, where $C_{10}$ is the yearly risk change from 2010. 
    - $R_\text{pop} = C_{95}^5 \times C_{10}^{11}$ from 2021 onwards (i.e. ceases changing thereafter). 
    - You can set the possible values for the yearly risk change in the 90s and 2010s using `yearly_risk_change_90s` and `yearly_risk_change_10s` in the data file. These are sampled during `__init__`. 
- `new_partner_factor`: This factor captures uncertaintly in the likelihood to form short term sexual partnerships. It applies to the whole population and is sampled at the beginning of the simulation. Possible values for this are set in the data file under `new_partner_factor`.
- `RISK` is the total risk factor for a person. It is the product of the other risk factors along with the new partner factor. 

### Long Term Partners

Individuals in the simulation can form longer term partnerships. An individual can only have one long term partner, but they can have short term partners at the same time. Unlike short term partners, long term partners may be maintained for more than one time step. 
- `update_long_term_partnerships` is the key function, run each timestep, to determine if long term partnerships begin or end. It starts off by getting two sub-populations: those who are sexually active and not partnered at the start of this timestep, and those who are sexually active and are partnered at the start of this timestep. It then calculates who will start a new long term partnership, assigns longevity to those partnerships which are new, and finally determines which existing partnerships should end. 
- `start_new_ltp` determines whether a new long term partnership starts depending on the person's sex, age group, and a balance factor for their sex. 
    - The `ltp_balance_factor` is calculated to try to address unbalanced numbers of men and women with long term partners. 
        - If we calculate the ratio $R = \frac{N_m}{N_w}$ where $N_m$ is the number of men in long term partnerships and $N_w$ is the number of women in long term partnerships:
        - $R < 0.8 \implies B_{male} = 4, B_{female} = 1$
        - $0.8 \le R < 0.9 \implies B_{male} = 2, B_{female} = 1$
        - $0.9 \le R < 1.1 \implies B_{male} = 1, B_{female} = 1$
        - $1.1 \le R < 1.2 \implies B_{male} = 1, B_{female} = 2$
        - $R > 1.2 \implies B_{male} = 1, B_{female} = 4$
    - There are four age groups: Under 35, 35 to 45, 45 to 55, over 55. The age factor for each group is:
        - 1 for under 35
        - $\frac{1}{2}$ for 35 to 45
        - $\frac{1}{3}$ for 45 to 55
        - $\frac{1}{5}$ for 55+
    - The base rate of forming an long term partnership is sampled in the `__init__` as $0.1 \times e^{-x/4}$ where $x$ is drawn from $\mathcal{N}(0, 1)$.
    - The final probability is the product of these three factors. 
- `new_ltp_longevity`: The longevity is determined for each new partnership, and determines how likely it is to end in any given timestep. It is also dependent on the agre groups described above. 
    - For those under 45s:
        - $P(L = 1) = 0.3$
        - $P(L = 2) = 0.3$
        - $P(L = 3) = 0.4$
    - For 45 - 55
        - $P(L = 1) = 0.3$
        - $P(L = 2) = 0.5$
        - $P(L = 3) = 0.2$
    - For 55+
        - $P(L = 1) = 0.3$
        - $P(L = 2) = 0.7$
        - $P(L = 3) = 0.0$
- `continue_ltp`: determines which long term partnerships continue or end. 
    - The probability to end a long term partnership depends on the longevity and a population wide rate change. 
    - The longevity factor is 0.25, 0.05, and 0.02 for $L = $ 1, 2, and 3 respectively. 
    - The population wide factor `ltp_rate_change`