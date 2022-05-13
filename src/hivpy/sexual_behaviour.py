import datetime
import operator
from enum import IntEnum

import numpy as np
import pandas as pd

from .common import SexType, selector
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


class SexualBehaviourModule:

    def select_matrix(self, matrix_list):
        return matrix_list[np.random.choice(len(matrix_list))]

    def __init__(self, **kwargs):
        # init sexual behaviour data
        self.sb_data = SexualBehaviourData("data/sex_behaviour.yaml")

        # Randomly initialise sexual behaviour group transitions
        self.sex_behaviour_trans = {
            SexType.Male: np.array(
                self.select_matrix(self.sb_data.sex_behaviour_transition_options["Male"])),
            SexType.Female: np.array(
                self.select_matrix(self.sb_data.sex_behaviour_transition_options["Female"]))
        }
        self.init_sex_behaviour_probs = self.sb_data.init_sex_behaviour_probs
        self.age_based_risk = self.sb_data.age_based_risk
        self.risk_categories = len(self.age_based_risk)-1
        self.risk_min_age = 15  # This should come out of config somewhere
        self.risk_age_grouping = 5  # ditto
        self.sex_mixing_matrix = {
            SexType.Male: self.select_matrix(self.sb_data.sex_mixing_matrix_male_options),
            SexType.Female: self.select_matrix(self.sb_data.sex_mixing_matrix_female_options)
        }
        self.short_term_partners = {SexType.Male: self.sb_data.male_stp_dists,
                                    SexType.Female: self.sb_data.female_stp_u25_dists}
        self.ltp_risk_factor = self.sb_data.rred_long_term_partnered.sample()
        self.rred_diagnosis = self.sb_data.rred_diagnosis.sample()
        self.rred_diagnosis_period = datetime.timedelta(days=365)*self.sb_data.rred_diagnosis_period

    def age_index(self, age):
        return np.minimum((age.astype(int)-self.risk_min_age) //
                          self.risk_age_grouping, self.risk_categories)

    def update_sex_behaviour(self, population):
        self.num_short_term_partners(population.data)
        self.update_sex_groups(population.data)
        self.update_rred_adc(population.data)
        self.update_rred_balance(population.data)
        self.update_rred_diagnosis(population.data, population.date)

    # Haven't been able to locate the probabilities for this yet
    # Doing them uniform for now
    def init_sex_behaviour_groups(self, population):
        for sex in SexType:
            index = selector(population, sex=(operator.eq, sex))
            n_sex = sum(index)
            population.loc[index, "sex_behaviour"] = self.init_sex_behaviour_probs[sex].sample(
                size=n_sex)

    def init_risk_factors(self, pop_data):
        n_pop = len(pop_data)
        self.init_rred_personal(pop_data, n_pop)
        self.init_new_partner_factor(pop_data, n_pop)
        pop_data["rred_age"] = np.ones(n_pop)  # Placeholder to be changed each time step
        pop_data["rred"] = (pop_data["new_partner_factor"] *
                            pop_data["rred_personal"])  # needs * rred_age at each step
        self.init_rred_adc(pop_data)
        self.init_rred_diagnosis(pop_data)

    def calc_rred_age(self, population, index):
        age = population.loc[index, "age"]
        sex = population.loc[index, "sex"]
        age_index = self.age_index(age)
        population.loc[index, "rred_age"] = self.age_based_risk[age_index, sex]

    def init_new_partner_factor(self, population, n_pop):
        population["new_partner_factor"] = self.sb_data.new_partner_dist.sample(size=n_pop)

    def calc_rred_long_term_partnered(self, population):
        population["rred_long_term_partnered"] = 1  # Unpartnered people
        partnered_idx = selector(population, partnered=(operator.eq, True))
        population.loc[partnered_idx, "rred_long_term_partnered"] = self.ltp_risk_factor
        # This might be more efficient, but is also a bit obscure
        # = ltp_risk_factor if ltp is true, and 1 if ltp is false.
        # population["rred_ltp"] = population["ltp"]*self.ltp_risk_factor + (1-population["ltp"])

    def init_rred_personal(self, population, n_pop):
        p_rred_p = self.sb_data.p_rred_p_dist.sample(size=len(population))
        population["rred_personal"] = np.ones(n_pop)
        r = np.random.uniform(size=n_pop)
        mask = r < p_rred_p
        population.loc[mask, "rred_personal"] = 1e-5

    def init_rred_adc(self, population):
        population["rred_adc"] = 1.0
        self.update_rred_adc(population)

    def update_rred_adc(self, population):
        """Updates risk reduction for AIDS defining condition"""
        # We don't need the rred_adc==1 condition (since they are all set to rred_adc = 1 to start)
        # It prevents needless assignments but requires checking more conditions
        # Not sure which is more efficient or if it matters.
        indices = selector(population, rred_adc=(operator.eq, 1), HIV_status=(operator.eq, True))
        population.loc[indices, "rred_adc"] = 0.2

    def init_rred_diagnosis(self, population):
        population["rred_diagnosis"] = 1

    def update_rred_diagnosis(self, population, date):
        HIV_idx_new = selector(population, HIV_status=(operator.eq, True),
                               HIV_Diagnosis_Date=(operator.ge, date-self.rred_diagnosis_period))
        population.loc[HIV_idx_new, "rred_diagnosis"] = self.rred_diagnosis
        HIV_idx_old = selector(population, HIV_status=(operator.eq, True),
                               HIV_Diagnosis_Date=(operator.lt, date-self.rred_diagnosis_period))
        population.loc[HIV_idx_old, "rred_diagnosis"] = np.sqrt(self.rred_diagnosis)

    def init_rred_balance(self, population):
        """Initialise risk reduction factor for balancing sex ratios"""
        population["rred_balance"] = 1.0

    def update_rred_balance(self, population):
        """Update balance of new partners for consistency between sexes.
           Integral discrepancies have been replaced with fractional discrepancy."""
        # We first need the difference of new partners between men and women
        men = population["sex"] == SexType.Male
        mens_partners = sum(population.loc[men, "num_partners"])
        womens_partners = sum(population.loc[(1 - men), "num_partners"])
        partner_discrepancy = (mens_partners - womens_partners) / len(population)
        if partner_discrepancy >= 0.1:
            rred_balance = 0.1
        elif partner_discrepancy >= 0.03:
            rred_balance = 0.7
        elif partner_discrepancy >= 0.005:
            rred_balance = 0.7
        elif partner_discrepancy >= 0.004:
            rred_balance = 0.75
        elif partner_discrepancy >= 0.003:
            rred_balance = 0.8
        elif partner_discrepancy >= 0.002:
            rred_balance = 0.9
        elif partner_discrepancy >= 0.001:
            rred_balance = 0.97
        elif partner_discrepancy >= -0.001:
            rred_balance = 1
        elif partner_discrepancy >= -0.002:
            rred_balance = 1/0.97
        elif partner_discrepancy >= -0.003:
            rred_balance = 1/0.9
        elif partner_discrepancy >= -0.004:
            rred_balance = 1/0.8
        elif partner_discrepancy >= -0.005:
            rred_balance = 1/0.75
        elif partner_discrepancy >= -0.03:
            rred_balance = 1/0.7
        elif partner_discrepancy >= -0.1:
            rred_balance = 1/0.7
        else:
            rred_balance = 10
        self.rred_balance = {SexType.Male: rred_balance,
                             SexType.Female: 1/rred_balance}

    def calc_newp_by_age_group(self, population):
        ages = np.linspace(self.risk_min_age, self.risk_min_age + self.risk_age_grouping*self.risk_categories, self.risk_age_grouping)

    # Here we need to figure out how to vectorise this which is currently blocked
    # by the sex if statement
    def prob_transition(self, sex, rred, i, j):
        """Calculates the probability of transitioning from sexual behaviour
        group i to group j, based on sex and age."""
        transition_matrix = self.sex_behaviour_trans[sex]

        denominator = transition_matrix[i][0] + rred*sum(transition_matrix[i][1:])

        if(j == 0):
            Probability = transition_matrix[i][0] / denominator
        else:
            Probability = rred*transition_matrix[i][j] / denominator

        return Probability

    def num_short_term_partners(self, population: pd.DataFrame):
        for sex in SexType:
            for g in SexBehaviours[sex]:
                index = selector(population, sex=(operator.eq, sex),
                                 sex_behaviour=(operator.eq, g),
                                 age=(operator.gt, 15))
                population.loc[index, "num_partners"] = (
                    self.short_term_partners[sex][g].sample(size=sum(index)))

    def update_sex_groups(self, population: pd.DataFrame):
        """Determine changes to sexual behaviour groups.
           Loops over sex, and behaviour groups within each sex.
           Within each group it then loops over groups again to check all transition pairs (i,j)."""
        for sex in SexType:
            for prev_group in SexBehaviours[sex]:
                index = selector(population, sex=(operator.eq, sex), sex_behaviour=(
                    operator.eq, prev_group), age=(operator.ge, 15))
                if any(index):
                    subpop_size = sum(index)
                    rands = np.random.uniform(0.0, 1.0, subpop_size)
                    self.calc_rred_age(population, index)
                    rred = population.loc[index, "rred"] * population.loc[index, "rred_age"]
                    dim = self.sex_behaviour_trans[sex].shape[0]
                    Pmin = np.zeros(subpop_size)
                    Pmax = np.zeros(subpop_size)
                    for new_group in range(dim):
                        Pmin = Pmax.copy()
                        Pmax += self.prob_transition(sex, rred, prev_group, new_group)
                        # This has to be a Series so it can be combined with index correctly
                        jump_to_new_group = pd.Series(
                            (rands >= Pmin) & (rands < Pmax), index=rred.index)
                        population.loc[index & jump_to_new_group, "sex_behaviour"] = new_group
