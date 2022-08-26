import datetime
import importlib.resources
import operator
from enum import IntEnum

import numpy as np

import hivpy.column_names as col

from .common import SexType, diff_years, rng, selector
from .sex_behaviour_data import SexualBehaviourData


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
        self.sex_mixing_matrix = {
            SexType.Male: rng.choice(self.sb_data.sex_mixing_matrix_male_options),
            SexType.Female: rng.choice(self.sb_data.sex_mixing_matrix_female_options)
        }
        self.short_term_partners = {SexType.Male: self.sb_data.male_stp_dists,
                                    SexType.Female: self.sb_data.female_stp_u25_dists}
        self.ltp_risk_factor = self.sb_data.rred_long_term_partnered.sample()
        self.rred_diagnosis = self.sb_data.rred_diagnosis.sample()
        self.rred_diagnosis_period = datetime.timedelta(days=365)*self.sb_data.rred_diagnosis_period
        self.yearly_risk_change = {"1990s": self.sb_data.yearly_risk_change["1990s"].sample(),
                                   "2010s": self.sb_data.yearly_risk_change["2010s"].sample()}

        # rred_art_adherence is only incorporated into the model in a fraction of the runs
        # usage of rred_art_adherence is randomly selected to be on or off for this simulation here
        self.use_rred_art_adherence = (
            rng.random() < self.sb_data.rred_art_adherence_probability)
        self.rred_art_adherence = self.sb_data.rred_art_adherence
        self.adherence_threshold = self.sb_data.adherence_threshold
        self.new_partner_factor = self.sb_data.new_partner_dist.sample()
        self.balance_thresholds = [0.1, 0.03, 0.005, 0.004, 0.003, 0.002, 0.001]
        self.balance_factors = [0.1, 0.7, 0.7, 0.75, 0.8, 0.9, 0.97]
        self.p_rred_p = self.sb_data.p_rred_p_dist.sample()

        # long term partnerships
        self.new_ltp_rate = 0.1 * np.exp(rng.normal() * 0.25)  # three month ep-rate
        self.annual_ltp_rate_change = rng.choice([0.8, 0.9, 0.95, 1.0])
        self.ltp_rate_change = 1.0
        self.ltp_end_rate_by_longevity = np.array([0, 0.25, 0.05, 0.02])
        self.ltp_balance_factor = {SexType.Male: 1, SexType.Female: 1}

    def age_index(self, age):
        return np.minimum((age.astype(int)-self.risk_min_age) //
                          self.risk_age_grouping, self.risk_categories)

    def update_sex_behaviour(self, population):
        self.num_short_term_partners(population)
        self.update_sex_groups(population)
        self.update_rred(population)
        self.update_long_term_partners(population)

    # Haven't been able to locate the probabilities for this yet
    # Doing them uniform for now
    def init_sex_behaviour_groups(self, population):
        population[col.SEX_BEHAVIOUR] = None  # to avoid type problems
        for sex in SexType:
            index = selector(population, sex=(operator.eq, sex))
            n_sex = sum(index)
            population.loc[index, col.SEX_BEHAVIOUR] = self.init_sex_behaviour_probs[sex].sample(
                size=n_sex)

    # Here we need to figure out how to vectorise this which is currently blocked
    # by the sex if statement
    def prob_transition(self, sex, rred, i, j):
        """Calculates the probability of transitioning from sexual behaviour
        group i to group j, based on sex and age."""
        transition_matrix = self.sex_behaviour_trans[sex]

        denominator = transition_matrix[i][0] + rred*sum(transition_matrix[i][1:])

        if (j == 0):
            Probability = transition_matrix[i][0] / denominator
        else:
            Probability = rred*transition_matrix[i][j] / denominator

        return Probability

    def get_partners_for_group(self, sex, group, size):
        """Calculates the number of short term partners for people in a given intersection of
        sex and sexual behaviour group. Args are: [sex, sexual behaviour group, size of group]"""
        group = int(group)
        return self.short_term_partners[sex][group].sample(size)

    def num_short_term_partners(self, population):
        """Calculate the number of short term partners for the whole population"""
        # can we avoid doing population.loc[active_pop] twice? Does it waste time?
        active_pop = population.data.index[(15 <= population.data.age) & (population.data.age < 65)]
        num_partners = population.transform_group(
            [col.SEX, col.SEX_BEHAVIOUR], self.get_partners_for_group, sub_pop=active_pop)
        population.data.loc[active_pop, col.NUM_PARTNERS] = num_partners

    def _assign_new_sex_group(self, sex, group, rred, size):
        group = int(group)
        rands = rng.uniform(0.0, 1.0, size)
        dim = self.sex_behaviour_trans[sex].shape[0]
        Pmin = np.zeros(size)
        Pmax = np.zeros(size)
        new_groups = np.zeros(size)
        for new_group in range(dim):
            Pmin = Pmax.copy()
            Pmax += self.prob_transition(sex, rred, group, new_group)
            new_groups[(rands >= Pmin) & (rands < Pmax)] = new_group
        return new_groups

    def update_sex_groups(self, population):
        """Determine changes to sexual behaviour groups.
           Loops over sex, and behaviour groups within each sex.
           Within each group it then loops over groups again to check all transition pairs (i,j)."""
        active_pop = population.data.index[(15 <= population.data.age) & (population.data.age < 65)]
        new_groups = population.transform_group(
            [col.SEX, col.SEX_BEHAVIOUR, col.RRED], self._assign_new_sex_group, sub_pop=active_pop)
        population.data.loc[active_pop, col.SEX_BEHAVIOUR] = new_groups

    # risk reduction factors
    def init_risk_factors(self, pop_data):
        n_pop = len(pop_data)
        self.init_rred_personal(pop_data, n_pop)
        pop_data[col.RRED_AGE] = np.ones(n_pop)  # Placeholder to be changed each time step
        self.update_rred_age(pop_data)
        pop_data[col.RRED] = (self.new_partner_factor *
                              pop_data[col.RRED_PERSONAL] *
                              pop_data[col.RRED_AGE])
        self.init_rred_adc(pop_data)
        self.init_rred_diagnosis(pop_data)
        self.init_rred_population()
        self.init_rred_art_adherence(pop_data)
        self.init_rred_balance(pop_data)

    def update_rred(self, population):
        self.update_rred_adc(population.data)
        self.update_rred_balance(population.data)
        self.update_rred_diagnosis(population.data, population.date)
        self.update_rred_population(population.date)
        self.update_rred_age(population.data)
        self.update_rred_long_term_partnered(population.data)
        if (self.use_rred_art_adherence):
            self.update_rred_art_adherence(population.data)
        population.data[col.RRED] = (self.new_partner_factor *
                                     population.data[col.RRED_AGE] *
                                     population.data[col.RRED_ADC] *
                                     population.data[col.RRED_BALANCE] *
                                     population.data[col.RRED_DIAGNOSIS] *
                                     population.data[col.RRED_PERSONAL] *
                                     self.rred_population *
                                     population.data[col.RRED_LTP] *
                                     population.data[col.RRED_ART_ADHERENCE])

    def init_rred_art_adherence(self, pop_data):
        pop_data[col.RRED_ART_ADHERENCE] = 1

    def update_rred_art_adherence(self, pop_data):
        if ("art_adherence" in pop_data.columns):
            indices = selector(pop_data, art_adherence=(operator.lt, self.adherence_threshold))
            pop_data.loc[indices, col.RRED_ART_ADHERENCE] = self.rred_art_adherence

    def update_rred_age(self, pop_data):
        over_15s = selector(pop_data, age=(operator.ge, 15))
        age = pop_data.loc[over_15s, col.AGE]
        sex = pop_data.loc[over_15s, col.SEX]
        age_index = self.age_index(age)
        pop_data.loc[over_15s, col.RRED_AGE] = self.age_based_risk[age_index, sex]

    def update_rred_long_term_partnered(self, pop_data):
        pop_data[col.RRED_LTP] = 1  # Unpartnered people
        if ("partnered" in pop_data.columns):
            partnered_idx = selector(pop_data, partnered=(operator.eq, True))
            pop_data.loc[partnered_idx, col.RRED_LTP] = self.ltp_risk_factor
            # This might be more efficient, but is also a bit obscure
            # = ltp_risk_factor if ltp is true, and 1 if ltp is false.
            # pop_data["rred_ltp"] = pop_data["ltp"]*self.ltp_risk_factor+(1-pop_data["ltp"])

    def init_rred_personal(self, population, n_pop):
        population[col.RRED_PERSONAL] = np.ones(n_pop)
        r = rng.uniform(size=n_pop)
        mask = r < self.p_rred_p
        population.loc[mask, col.RRED_PERSONAL] = 1e-5

    def init_rred_adc(self, population):
        population[col.RRED_ADC] = 1.0
        self.update_rred_adc(population)

    def update_rred_adc(self, population):
        """Updates risk reduction for AIDS defining condition"""
        # We don't need the rred_adc==1 condition (since they are all set to rred_adc = 1 to start)
        # It prevents needless assignments but requires checking more conditions
        # Not sure which is more efficient or if it matters.
        indices = selector(population, rred_adc=(operator.eq, 1), HIV_status=(operator.eq, True))
        population.loc[indices, col.RRED_ADC] = 0.2

    def init_rred_population(self):
        """Initialise general population risk reduction w.r.t. condomless sex with new partners"""
        self.rred_population = 1

    def update_rred_population(self, date):
        yearly_change_90s = self.yearly_risk_change["1990s"]
        yearly_change_10s = self.yearly_risk_change["2010s"]
        if (date1995 < date <= date2000):
            dt = diff_years(date1995, date)
            self.rred_population = yearly_change_90s**dt
        elif (date2000 < date < date2010):
            self.rred_population = yearly_change_90s**5
        elif (date2010 < date < date2021):
            dt = diff_years(date2010, date)
            self.rred_population = yearly_change_90s**5 * yearly_change_10s**dt
        elif (date2021 < date):
            self.rred_population = yearly_change_90s**5 * yearly_change_10s**11

    def init_rred_diagnosis(self, population):
        population[col.RRED_DIAGNOSIS] = 1

    def update_rred_diagnosis(self, population, date):
        HIV_idx_new = selector(population, HIV_status=(operator.eq, True),
                               HIV_Diagnosis_Date=(operator.ge, date-self.rred_diagnosis_period))
        population.loc[HIV_idx_new, col.RRED_DIAGNOSIS] = self.rred_diagnosis
        HIV_idx_old = selector(population, HIV_status=(operator.eq, True),
                               HIV_Diagnosis_Date=(operator.lt, date-self.rred_diagnosis_period))
        population.loc[HIV_idx_old, col.RRED_DIAGNOSIS] = np.sqrt(self.rred_diagnosis)

    def init_rred_balance(self, population):
        """Initialise risk reduction factor for balancing sex ratios"""
        population[col.RRED_BALANCE] = 1.0

    def update_rred_balance(self, population):
        """Update balance of new partners for consistency between sexes.
           Integral discrepancies have been replaced with fractional discrepancy."""
        # We first need the difference of new partners between men and women
        men = population[col.SEX] == SexType.Male
        women = population[col.SEX] == SexType.Female
        mens_partners = sum(population.loc[men, col.NUM_PARTNERS])
        womens_partners = sum(population.loc[women, col.NUM_PARTNERS])
        partner_discrepancy = abs(mens_partners - womens_partners) / len(population)

        rred_balance = 1
        for (t, b) in zip(self.balance_thresholds, self.balance_factors):
            if partner_discrepancy >= t:
                rred_balance = b
                break

        if (mens_partners > womens_partners):
            population.loc[men, col.RRED_BALANCE] = rred_balance
            population.loc[women, col.RRED_BALANCE] = 1/rred_balance
        else:
            population.loc[men, col.RRED_BALANCE] = 1/rred_balance
            population.loc[women, col.RRED_BALANCE] = rred_balance

    def update_ltp_rate_change(self, date):
        if date1995 < date < date2000:
            dt = diff_years(date1995, date)
            self.ltp_rate_change = self.annual_ltp_rate_change**(dt)
        elif date >= date2000:
            self.ltp_rate_change = self.annual_ltp_rate_change**5
            # don't really need to keep doing this update after 2000

    def balance_ltp_factor(self, population):
        num_ltp_women = sum(
            population.data.loc[population.data[col.SEX] == SexType.Female, col.LONG_TERM_PARTNER])
        num_ltp_men = sum(
            population.data.loc[population.data[col.SEX] == SexType.Male, col.LONG_TERM_PARTNER])
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

    def start_new_ltp(self, sex, age, size):
        # TODO add alterations for new diagnosis
        ltp_rands = rng.random(size=size)
        rate_modifier = 1
        if (35 <= age < 45):
            rate_modifier = 0.5
        elif (45 <= age < 55):
            rate_modifier = 1/3
        elif (55 <= age):
            rate_modifier = 0.2
        new_relationship = ltp_rands < (self.new_ltp_rate
                                        * rate_modifier
                                        * self.ltp_balance_factor[sex])
        return new_relationship

    def new_ltp_longevity(self, age, sex, size):
        longevity = np.zeros(size)  # each new relationship needs a longevity
        longevity_rands = rng.random(size)
        if (age < 45):
            longevity = ((longevity_rands < 0.3) * 1 +
                         ((0.3 <= longevity_rands) & (longevity_rands < 0.6)) * 2 +
                         (0.6 <= longevity_rands) * 3)
        elif (age < 55):
            longevity = ((longevity_rands < 0.3) * 1 +
                         ((0.3 <= longevity_rands) & (longevity_rands < 0.8)) * 2 +
                         (0.8 <= longevity_rands) * 3)
        else:
            longevity = ((longevity_rands < 0.3) * 1 +
                         (0.3 <= longevity_rands) * 2)
        return longevity

    def continue_ltp(self, longevity, size):
        """Function to decide which long term partners cease condomless sex based on
        relationship longevity, age, and sex."""
        # TODO: Add balancing factors for age and sex demographics.
        end_probability = self.ltp_end_rate_by_longevity[int(longevity)] / self.ltp_rate_change
        r = rng.random(size=size)
        return (r > end_probability)

    def update_long_term_partners(self, population):
        start_partnerless = ((~population.data[col.LONG_TERM_PARTNER]) &
                             (population.data[col.AGE] >= 15) &
                             (population.data[col.AGE] < 65))
        partnerless_idx = population.data.index[start_partnerless]

        partnered_idx = population.data.index[(population.data[col.LONG_TERM_PARTNER]) &
                                              (population.data[col.AGE] >= 15) &
                                              (population.data[col.AGE] < 65)]

        (self.ltp_balance_factor[SexType.Male],
         self.ltp_balance_factor[SexType.Female]) = self.balance_ltp_factor(population)
        # new relationships
        new_relationships = population.transform_group(
            [col.SEX, col.AGE],  self.start_new_ltp, sub_pop=partnerless_idx)
        population.data.loc[partnerless_idx, col.LONG_TERM_PARTNER] = new_relationships
        # not sure about indexing on this next line
        # We only want to update LTP_LONGEVITY where LONG_TERM_PARTNER is true within
        # the sub-population defined by partnerless_idx
        new_ltp_subpop = population.data.index[start_partnerless &
                                               population.data[col.LONG_TERM_PARTNER]]
        longevity = population.transform_group(
            [col.AGE, col.SEX], self.new_ltp_longevity, sub_pop=new_ltp_subpop)
        population.data.loc[new_ltp_subpop, col.LTP_LONGEVITY] = longevity.astype(int)
        # ending relationships
        population.data.loc[partnered_idx, col.LONG_TERM_PARTNER] = population.transform_group(
            [col.LTP_LONGEVITY], self.continue_ltp, sub_pop=partnered_idx)
