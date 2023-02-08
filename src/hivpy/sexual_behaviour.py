from __future__ import annotations

import datetime
import importlib.resources
import operator
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

import hivpy.column_names as col

from .common import SexType, diff_years, rng
from .sex_behaviour_data import SexualBehaviourData

if TYPE_CHECKING:
    from .population import Population


class MaleSexBehaviour(IntEnum):
    ZERO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class FemaleSexBehaviour(IntEnum):
    ZERO = 0
    ANY = 1


SexBehaviours = {SexType.Male: MaleSexBehaviour, SexType.Female: FemaleSexBehaviour}


date1995 = datetime.date(1995, 1, 1)
date2000 = datetime.date(2000, 1, 1)
date2010 = datetime.date(2010, 1, 1)
date2021 = datetime.date(2021, 1, 1)


class SexualBehaviourModule:

    def __init__(self, **kwargs):
        # init sexual behaviour data
        with importlib.resources.path("hivpy.data", "sex_behaviour.yaml") as data_path:
            self.sb_data = SexualBehaviourData(data_path)

        # Randomly initialise sexual behaviour group transitions
        self.sex_behaviour_trans = {
            SexType.Male: np.array(
                rng.choice(self.sb_data.sex_behaviour_transition_options["Male"])),
            SexType.Female: np.array(
                rng.choice(self.sb_data.sex_behaviour_transition_options["Female"]))
        }
        self.init_sex_behaviour_probs = self.sb_data.init_sex_behaviour_probs
        self.age_based_risk = self.sb_data.age_based_risk
        self.risk_categories = len(self.age_based_risk)-1
        self.risk_min_age = 15  # This should come out of config somewhere
        self.risk_age_grouping = 5  # ditto
        self.risk_max_age = 65
        self.sex_mix_age_grouping = 10
        self.sex_mix_age_groups = np.arange(self.risk_min_age,
                                            self.risk_max_age,
                                            self.sex_mix_age_grouping)
        self.num_sex_mix_groups = len(self.sex_mix_age_groups)
        self.age_limits = [self.risk_min_age + n*self.risk_age_grouping
                           for n in range((self.risk_max_age - self.risk_min_age)
                                          // self.risk_age_grouping)]
        self.sex_mixing_matrix = {
            SexType.Male: rng.choice(self.sb_data.sex_mixing_matrix_male_options),
            SexType.Female: rng.choice(self.sb_data.sex_mixing_matrix_female_options)
        }
        # FIXME we don't have the over25 distribution here!
        self.short_term_partners = {SexType.Male: self.sb_data.male_stp_dists,
                                    SexType.Female: self.sb_data.female_stp_u25_dists}
        self.ltp_risk_factor = self.sb_data.risk_long_term_partnered.sample()
        self.risk_diagnosis = self.sb_data.risk_diagnosis.sample()
        self.risk_diagnosis_period = datetime.timedelta(days=365)*self.sb_data.risk_diagnosis_period
        self.yearly_risk_change = {"1990s": self.sb_data.yearly_risk_change["1990s"].sample(),
                                   "2010s": self.sb_data.yearly_risk_change["2010s"].sample()}

        # risk_art_adherence is only incorporated into the model in a fraction of the runs
        # usage of risk_art_adherence is randomly selected to be on or off for this simulation here
        self.use_risk_art_adherence = (
            rng.random() < self.sb_data.risk_art_adherence_probability)
        self.risk_art_adherence = self.sb_data.risk_art_adherence
        self.adherence_threshold = self.sb_data.adherence_threshold
        self.new_partner_factor = self.sb_data.new_partner_dist.sample()
        self.balance_thresholds = [0.1, 0.03, 0.005, 0.004, 0.003, 0.002, 0.001]
        self.balance_factors = [0.1, 0.7, 0.7, 0.75, 0.8, 0.9, 0.97]
        self.p_risk_p = self.sb_data.p_risk_p_dist.sample()

        # long term partnerships
        self.new_ltp_rate = 0.1 * np.exp(rng.normal() * 0.25)  # three month ep-rate
        self.annual_ltp_rate_change = rng.choice([0.8, 0.9, 0.95, 1.0])
        self.ltp_rate_change = 1.0
        self.ltp_end_rate_by_longevity = np.array([0, 0.25, 0.05, 0.02])
        self.ltp_balance_factor = {SexType.Male: 1, SexType.Female: 1}
        self.new_ltp_age_bins = [35, 45, 55]

        # sex workers
        self.base_start_sw = self.sb_data.base_start_sex_work.sample()
        self.base_stop_sw = self.sb_data.base_stop_sex_work.sample()
        self.risk_sex_worker_age = self.sb_data.risk_sex_worker_age
        self.sw_age_bins = [20, 25, 35]

    def age_index(self, age):
        return np.minimum((age.astype(int)-self.risk_min_age) //
                          self.risk_age_grouping, self.risk_categories)

    def update_sex_worker_status(self, population: Population):
        # Only consider female sex workers
        # is there a max age for sex workers?
        female_15to49_not_sw = population.get_sub_pop([(col.SEX, operator.eq, SexType.Female),
                                                 (col.AGE, operator.ge, 15),
                                                 (col.AGE, operator.lt, 50),
                                                 (col.SEX_WORKER, operator.eq, False)])
        population.set_present_variable(col.SW_AGE_GROUP, np.digitize(population.get_variable(
            col.AGE, female_15to49_not_sw), self.sw_age_bins), female_15to49_not_sw)

        sex_workers = population.get_sub_pop([(col.SEX_WORKER, operator.eq, True)])

        def start_sex_work(age_group, size):
            sw_status = rng.random(size) < (self.base_start_sw * np.sqrt(self.risk_population)
                                            * self.risk_sex_worker_age[age_group])
            return sw_status

        def stop_sex_work(age_group, size):
            # group 0: under 40, group 1 over 40, group 2 over 50
            if (age_group == 2):
                # Over 50s stop sex work
                return False
            prob_stop_sw = self.base_stop_sw / np.sqrt(self.risk_population)
            if (age_group == 1):
                prob_stop_sw *= 3
            sw_status = rng.random(size) > prob_stop_sw
            return sw_status

        # Women starting sex work 
        population.set_variable_by_group(col.SEX_WORKER,
                                         [col.SW_AGE_GROUP],
                                         start_sex_work,
                                         sub_pop=female_15to49_not_sw)

        # Women stopping sex work
        new_sex_work_status = population.transform_group([col.SW_AGE_GROUP],
                                                         stop_sex_work,
                                                         sub_pop=sex_workers)
        population.set_present_variable(col.SEX_WORKER, new_sex_work_status, sex_workers)
        # We need a way of getting the indexing for a population of people 
        # returned from a function like this
        women_stopping_sex_work = population.get_sub_pop_from_array(new_sex_work_status, sex_workers)
        population.set_present_variable(col.DATE_STOP_SW, population.date, women_stopping_sex_work)

    def update_sex_behaviour(self, population: Population):
        self.update_sex_worker_status(population)
        self.num_short_term_partners(population)
        self.assign_stp_ages(population)
        self.update_sex_groups(population)
        self.update_risk(population)
        self.update_long_term_partners(population)

    # Haven't been able to locate the probabilities for this yet
    # Doing them uniform for now
    def init_sex_behaviour_groups(self, population: Population):
        population.init_variable(col.SEX_BEHAVIOUR, None)
        population.set_variable_by_group(col.SEX_BEHAVIOUR,
                                         [col.SEX],
                                         lambda sex, n:
                                         self.init_sex_behaviour_probs[sex].sample(size=n))

    # Here we need to figure out how to vectorise this which is currently blocked
    # by the sex if statement
    def prob_transition(self, sex, risk, i, j):
        """
        Calculates the probability of transitioning from sexual behaviour
        group i to group j, based on sex and age.
        """
        transition_matrix = self.sex_behaviour_trans[sex]

        denominator = transition_matrix[i][0] + risk*sum(transition_matrix[i][1:])

        if (j == 0):
            Probability = transition_matrix[i][0] / denominator
        else:
            Probability = risk*transition_matrix[i][j] / denominator

        return Probability

    def get_partners_for_group(self, sex, group, size):
        """
        Calculates the number of short term partners for people in a given intersection of
        sex and sexual behaviour group. Args are: [sex, sexual behaviour group, size of group].
        """
        group = int(group)
        stp = self.short_term_partners[sex][group].sample(size)
        return stp

    def num_short_term_partners(self, population: Population):
        """
        Calculate the number of short term partners for the whole population
        """
        active_pop = population.get_sub_pop([(col.AGE, operator.ge, 15),
                                             (col.AGE, operator.le, 65)])
        population.set_variable_by_group(col.NUM_PARTNERS,
                                         [col.SEX, col.SEX_BEHAVIOUR],
                                         self.get_partners_for_group,
                                         sub_pop=active_pop)

    def _assign_new_sex_group(self, sex, group, risk, size):
        group = int(group)
        rands = rng.uniform(0.0, 1.0, size)
        dim = self.sex_behaviour_trans[sex].shape[0]
        Pmin = np.zeros(size)
        Pmax = np.zeros(size)
        new_groups = np.zeros(size)
        for new_group in range(dim):
            Pmin = Pmax.copy()
            Pmax += self.prob_transition(sex, risk, group, new_group)
            new_groups[(rands >= Pmin) & (rands < Pmax)] = new_group
        return new_groups

    def update_sex_groups(self, population: Population):
        """
        Determine changes to sexual behaviour groups.
           Loops over sex, and behaviour groups within each sex.
           Within each group it then loops over groups again to check all transition pairs (i,j).
        """
        active_pop = population.get_sub_pop([(col.AGE, operator.gt, 15),
                                             (col.AGE, operator.le, 65)])
        population.set_variable_by_group(col.SEX_BEHAVIOUR,
                                         [col.SEX, col.SEX_BEHAVIOUR, col.RISK],
                                         self._assign_new_sex_group,
                                         sub_pop=active_pop)

    # risk reduction factors
    def init_risk_factors(self, pop: Population):
        self.init_risk_personal(pop)
        pop.init_variable(col.RISK_AGE, 1)  # Placeholder to be changed each time step
        self.update_risk_age(pop)
        pop.set_present_variable(col.RISK,
                                 pop.apply_vector_func([col.RISK_PERSONAL, col.RISK_AGE],
                                                       self.calc_risk_base))
        self.init_risk_adc(pop)
        self.init_risk_diagnosis(pop)
        self.init_risk_population()
        self.init_risk_art_adherence(pop)
        self.init_risk_balance(pop)

    def calc_risk_base(self, risk_personal, risk_age):
        return (self.new_partner_factor *
                risk_personal *
                risk_age)

    def update_risk(self, population: Population):
        self.update_risk_adc(population)
        self.update_risk_balance(population)
        self.update_risk_diagnosis(population)
        self.update_risk_population(population.date)
        self.update_risk_age(population)
        self.update_risk_long_term_partnered(population)
        if (self.use_risk_art_adherence):
            self.update_risk_art_adherence(population)

        def comined_risk(age, adc, balance, diagnosis, personal, ltp, art):
            return (self.new_partner_factor * age * adc * balance *
                    diagnosis * personal * self.risk_population * ltp * art)
        population.set_present_variable(col.RISK,
                                        population.apply_vector_func([col.RISK_AGE,
                                                                      col.RISK_ADC,
                                                                      col.RISK_BALANCE,
                                                                      col.RISK_DIAGNOSIS,
                                                                      col.RISK_PERSONAL,
                                                                      col.RISK_LTP,
                                                                      col.RISK_ART_ADHERENCE],
                                                                     comined_risk))

    def init_risk_art_adherence(self, pop: Population):
        pop.init_variable(col.RISK_ART_ADHERENCE, 1.0)

    def update_risk_art_adherence(self, pop: Population):
        # need to make this column check more intuitive
        if (col.ART_ADHERENCE in pop.data.columns):
            low_adherence_pop = pop.get_sub_pop(
                [(col.ART_ADHERENCE, operator.lt, self.adherence_threshold)])
            pop.set_present_variable(col.RISK_ART_ADHERENCE,
                                     self.risk_art_adherence, low_adherence_pop)

    def update_risk_age(self, pop: Population):
        over_15s = pop.get_sub_pop([(col.AGE, operator.gt, 15)])
        age = pop.get_variable(col.AGE, over_15s)
        sex = pop.get_variable(col.SEX, over_15s)
        age_index = self.age_index(age)
        pop.set_present_variable(col.RISK_AGE, self.age_based_risk[age_index, sex], over_15s)

    def update_risk_long_term_partnered(self, pop: Population):
        pop.set_present_variable(col.RISK_LTP, 1)  # Unpartnered people
        partnered_pop = pop.get_sub_pop([(col.LONG_TERM_PARTNER, operator.eq, True)])
        pop.set_present_variable(col.RISK_LTP, self.ltp_risk_factor, partnered_pop)

    def init_risk_personal(self, population: Population):
        population.init_variable(col.RISK_PERSONAL, 1)  # personal risk doesn't update(?)
        r = rng.uniform(size=population.size)
        mask = r < self.p_risk_p
        population.set_present_variable(col.RISK_PERSONAL, 1e-5, mask)

    def init_risk_adc(self, population: Population):
        population.init_variable(col.RISK_ADC, 1.0)
        self.update_risk_adc(population)

    def update_risk_adc(self, population: Population):
        """
        Updates risk reduction for AIDS defining condition
        """
        # We don't need the risk_adc==1 condition (since they are all set to risk_adc = 1 to start)
        # It prevents needless assignments but requires checking more conditions
        # Not sure which is more efficient or if it matters.
        indices = population.get_sub_pop([(col.ADC, operator.eq, True),
                                          (col.HIV_STATUS, operator.eq, True)])
        population.set_present_variable(col.RISK_ADC, 0.2, indices)

    def init_risk_population(self):
        """
        Initialise general population risk reduction w.r.t. condomless sex with new partners
        """
        self.risk_population = 1.0

    def update_risk_population(self, date):
        yearly_change_90s = self.yearly_risk_change["1990s"]
        yearly_change_10s = self.yearly_risk_change["2010s"]
        if (date1995 < date <= date2000):
            dt = diff_years(date1995, date)
            self.risk_population = yearly_change_90s**dt
        elif (date2000 < date < date2010):
            self.risk_population = yearly_change_90s**5
        elif (date2010 < date < date2021):
            dt = diff_years(date2010, date)
            self.risk_population = yearly_change_90s**5 * yearly_change_10s**dt
        elif (date2021 < date):
            self.risk_population = yearly_change_90s**5 * yearly_change_10s**11

    def init_risk_diagnosis(self, population: Population):
        population.init_variable(col.RISK_DIAGNOSIS, 1)  # do we want previous timesteps?

    def update_risk_diagnosis(self, population: Population):
        new_HIV_pop = population.get_sub_pop(
            [(col.HIV_STATUS, operator.eq, True),
             (col.HIV_DIAGNOSIS_DATE, operator.ge, population.date-self.risk_diagnosis_period)])
        population.set_present_variable(col.RISK_DIAGNOSIS, self.risk_diagnosis, new_HIV_pop)
        # TODO: Could make this update more efficient by only
        # getting the people whose value need to change
        # i.e. just crossed the threshold this time step
        old_HIV_pop = population.get_sub_pop(
            [(col.HIV_STATUS, operator.eq, True),
             (col.HIV_DIAGNOSIS_DATE, operator.lt, population.date-self.risk_diagnosis_period)])
        population.set_present_variable(
            col.RISK_DIAGNOSIS, np.sqrt(self.risk_diagnosis), old_HIV_pop)

    def init_risk_balance(self, population: Population):
        """
        Initialise risk reduction factor for balancing sex ratios
        """
        population.init_variable(col.RISK_BALANCE, 1.0)

    def update_risk_balance(self, population: Population):
        """
        Update balance of new partners for consistency between sexes.
           Integral discrepancies have been replaced with fractional discrepancy.
        """
        # We first need the difference of new partners between men and women
        men = population.get_sub_pop([(col.SEX, operator.eq, SexType.Male)])
        women = population.get_sub_pop([(col.SEX, operator.eq, SexType.Female)])
        mens_partners = sum(population.get_variable(col.NUM_PARTNERS, men))
        womens_partners = sum(population.get_variable(col.NUM_PARTNERS, women))
        partner_discrepancy = abs(mens_partners - womens_partners) / population.size

        risk_balance = 1
        for (t, b) in zip(self.balance_thresholds, self.balance_factors):
            if partner_discrepancy >= t:
                risk_balance = b
                break

        if (mens_partners > womens_partners):
            population.set_present_variable(col.RISK_BALANCE, risk_balance, men)
            population.set_present_variable(col.RISK_BALANCE, 1/risk_balance, women)
        else:
            population.set_present_variable(col.RISK_BALANCE, risk_balance, women)
            population.set_present_variable(col.RISK_BALANCE, 1/risk_balance, men)

    def gen_stp_ages(self, sex, age_group, num_partners, size):
        # TODO: Check if this needs additional balancing factors for age
        stp_age_probs = self.sex_mixing_matrix[sex][age_group]
        stp_age_groups = rng.choice(self.num_sex_mix_groups, [size, num_partners], p=stp_age_probs)
        return list(stp_age_groups)  # dataframe won't accept a 2D numpy array

    def assign_stp_ages(self, population: Population):
        """Calculate the ages of a persons short term partners
        from the mixing matrices."""
        population.set_present_variable(col.SEX_MIX_AGE_GROUP,
                                        (np.digitize(population.get_variable(col.AGE),
                                                     self.sex_mix_age_groups) - 1))
        active_pop = population.get_sub_pop([(col.NUM_PARTNERS, operator.gt, 0)])
        population.set_variable_by_group(col.STP_AGE_GROUPS,
                                         [col.SEX, col.SEX_MIX_AGE_GROUP, col.NUM_PARTNERS],
                                         self.gen_stp_ages,
                                         sub_pop=active_pop)

    def update_ltp_rate_change(self, date):
        if date1995 < date < date2000:
            dt = diff_years(date1995, date)
            self.ltp_rate_change = self.annual_ltp_rate_change**(dt)
        elif date >= date2000:
            self.ltp_rate_change = self.annual_ltp_rate_change**5
            # don't really need to keep doing this update after 2000

    def balance_ltp_factor(self, population: Population):
        num_ltp_women = sum(population.get_variable(col.LONG_TERM_PARTNER,
                            population.get_sub_pop([(col.SEX, operator.eq, SexType.Female)])))
        num_ltp_men = sum(population.get_variable(col.LONG_TERM_PARTNER,
                          population.get_sub_pop([(col.SEX, operator.eq, SexType.Male)])))
        if num_ltp_women > 0:
            ratio_ltp = num_ltp_men / num_ltp_women
            if ratio_ltp < 0.8:
                return (4, 1)
            elif ratio_ltp < 0.9:
                return (2, 1)
            elif ratio_ltp < 1.1:
                return (1, 1)
            elif ratio_ltp < 1.2:
                return (1, 2)
            else:
                return (1, 4)
        else:
            return (1, 1)

    def start_new_ltp(self, sex, age_group, size):
        # TODO add alterations for new diagnosis
        ltp_rands = rng.random(size=size)
        rate_modifier = 1
        if age_group == 1:
            rate_modifier = 0.5
        elif age_group == 2:
            rate_modifier = 1/3
        elif age_group == 3:
            rate_modifier = 0.2
        new_relationship = ltp_rands < (self.new_ltp_rate
                                        * rate_modifier
                                        * self.ltp_balance_factor[sex])
        return new_relationship

    def new_ltp_longevity(self, age_group, size):
        longevity = np.zeros(size)  # each new relationship needs a longevity
        longevity_rands = rng.random(size)
        thresholds = [0, 0]
        if age_group < 2:
            thresholds = [0.3, 0.6]
        elif age_group == 2:
            thresholds = [0.3, 0.8]
        else:
            thresholds = [0.3, 1.0]

        longevity[longevity_rands < thresholds[0]] = 1
        longevity[(thresholds[0] <= longevity_rands) & (longevity_rands < thresholds[1])] = 2
        longevity[thresholds[1] <= longevity_rands] = 3

        return longevity

    def continue_ltp(self, longevity, size):
        """
        Function to decide which long term partners cease condomless sex based on
        relationship longevity, age, and sex.
        """
        # TODO: Add balancing factors for age and sex demographics.
        end_probability = self.ltp_end_rate_by_longevity[int(longevity)] / self.ltp_rate_change
        r = rng.random(size=size)
        return (r > end_probability)

    def update_long_term_partners(self, population: Population):
        start_partnerless = population.get_sub_pop([
            (col.LONG_TERM_PARTNER, operator.eq, False),
            (col.AGE, operator.ge, 15),
            (col.AGE, operator.lt, 65)
        ])
        start_partnered = population.get_sub_pop([
            (col.LONG_TERM_PARTNER, operator.eq, True),
            (col.AGE, operator.ge, 15),
            (col.AGE, operator.lt, 65)
        ])

        (self.ltp_balance_factor[SexType.Male],
         self.ltp_balance_factor[SexType.Female]) = self.balance_ltp_factor(population)

        # new relationships
        ltp_age_groups = np.digitize(population.get_variable(col.AGE), self.new_ltp_age_bins)
        population.set_present_variable(col.LTP_AGE_GROUP, ltp_age_groups)
        population.set_variable_by_group(
            col.LONG_TERM_PARTNER,
            [col.SEX, col.LTP_AGE_GROUP],
            self.start_new_ltp,
            sub_pop=start_partnerless
        )

        # calculate longevity for new relationships this time step
        # (but not for existing relationships)
        new_ltp_subpop = population.get_sub_pop_intersection(
            start_partnerless,
            population.get_sub_pop([(col.LONG_TERM_PARTNER, operator.eq, True)])
        )

        population.set_variable_by_group(
            col.LTP_LONGEVITY,
            [col.LTP_AGE_GROUP],
            self.new_ltp_longevity,
            sub_pop=new_ltp_subpop
        )

        # ending relationships
        # (doesn't apply to relationships formed just now)
        population.set_variable_by_group(
            col.LONG_TERM_PARTNER,
            [col.LTP_LONGEVITY],
            self.continue_ltp,
            sub_pop=start_partnered
        )
