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
        self.baseline_risk = self.sb_data.baseline_risk
        self.risk_categories = len(self.baseline_risk)-1
        self.risk_min_age = 15  # This should come out of config somewhere
        self.risk_age_grouping = 5  # ditto
        self.sex_mixing_matrix = {
            SexType.Male: self.select_matrix(self.sb_data.sex_mixing_matrix_male_options),
            SexType.Female: self.select_matrix(self.sb_data.sex_mixing_matrix_female_options)
        }
        self.short_term_partners = {SexType.Male: self.sb_data.male_stp_dists,
                                    SexType.Female: self.sb_data.female_stp_u25_dists}

    # Haven't been able to locate the probabilities for this yet
    # Doing them uniform for now
    def init_sex_behaviour_groups(self, population):
        for sex in SexType:
            index = selector(population, sex=(operator.eq, sex))
            n_sex = sum(index)
            population.loc[index, "sex_behaviour"] = self.init_sex_behaviour_probs[sex].rvs(
                size=n_sex)

    def init_risk_factors(self, population):
        n_pop = len(population)
        self.init_rred_personal(population, n_pop)
        self.init_new_partner_factor(population, n_pop)
        self.init_rred_age(population)
        population["rred"] = (population["new_partner_factor"] *
                              population["rred_age"] *
                              population["rred_personal"])

    def init_rred_age(self, population):
        age = population["age"]
        sex = population["sex"]
        age_index = np.minimum((age.astype(int)-self.risk_min_age)//self.risk_age_grouping,
                               self.risk_categories)
        population["rred_age"] = self.baseline_risk[age_index, sex]

    def init_new_partner_factor(self, population, n_pop):
        population["new_partner_factor"] = self.sb_data.new_partner_factor.rvs(size=n_pop)

    def init_rred_personal(self, population, n_pop):
        p_rred_p = self.sb_data.p_rred_p_dist.rvs(size=len(population))
        population["rred_personal"] = np.ones(n_pop)
        r = np.random.uniform(size=n_pop)
        mask = r < p_rred_p
        population["rred_personal"][mask] = 1e-5

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
                    self.short_term_partners[sex][g].rvs(size=sum(index)))

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
                    rred = population.loc[index, "rred"]
                    dim = self.sex_behaviour_trans[sex].shape[0]
                    Pmin = np.zeros(subpop_size)
                    Pmax = np.zeros(subpop_size)
                    for new_group in range(dim):
                        Pmin = Pmax.copy()
                        Pmax += self.prob_transition(sex, rred, prev_group, new_group)
                        # This has to be a Series so it can be combined with index correctly
                        jump_to_new_group = pd.Series((rands >= Pmin) & (rands < Pmax),
                                                      index)
                        population.loc[index & jump_to_new_group, "sex_behaviour"] = new_group
