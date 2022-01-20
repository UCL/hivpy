import operator
from enum import Enum
from functools import reduce

import numpy as np
import pandas as pd

from . import sex_behaviour_data as sb
from .demographics import SexType


class MaleSexBehaviour(Enum):
    ZERO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class FemaleSexBehaviour(Enum):
    ZERO = 0
    ANY = 1


SexBehaviours = {SexType.Male: MaleSexBehaviour, SexType.Female: FemaleSexBehaviour}


class SexualBehaviourModule:

    def select_matrix(self, matrix_list):
        return matrix_list[np.random.choice(matrix_list.shape[0])]

    def __init__(self, **kwargs):
        # Randomly initialise sexual behaviour group transitions
        self.sex_behaviour_trans_male = self.select_matrix(sb.sex_behaviour_trans_male_options)
        self.sex_behaviour_trans_female = self.select_matrix(sb.sex_behaviour_trans_female_options)
        self.baseline_risk = sb.baseline_risk  # Baseline risk appears to only have one option
        self.sex_mixing_matrix_female = self.select_matrix(sb.sex_mixing_matrix_female_options)
        self.sex_mixing_matrix_male = self.select_matrix(sb.sex_mixing_matrix_male_options)
        self.short_term_partners = {SexType.Male:
                                    self.select_matrix(sb.short_term_partners_male_options),
                                    SexType.Female:
                                    self.select_matrix(sb.short_term_partners_female_options)}

    # Haven't been able to locate the probabilities for this yet
    # Doing them uniform for now
    def init_sex_behaviour_groups(self, population):
        population["sex_behaviour"] = np.where(population['sex'] == SexType.Male,
                                               np.random.choice(MaleSexBehaviour).value,
                                               np.random.choice(FemaleSexBehaviour).value)

    # Here we need to figure out how to vectorise this which is currently blocked
    # by the sex if statement
    def prob_transition(self, sex, age, i, j):
        """Calculates the probability of transitioning from sexual behaviour
        group i to group j, based on sex and age."""
        if(sex == SexType.Female):
            transition_matrix = self.sex_behaviour_trans_female
            sex_index = 1
        else:
            transition_matrix = self.sex_behaviour_trans_male
            sex_index = 0

        age_index = min((int(age)-15)//5, 9)

        risk_factor = self.baseline_risk[age_index][sex_index]

        denominator = transition_matrix[i][0] + risk_factor*sum(transition_matrix[i][1:])

        if(j == 0):
            Probability = transition_matrix[i][0] / denominator
        else:
            Probability = risk_factor*transition_matrix[i][j] / denominator

        return Probability

    def num_short_term_partners(self, population: pd.DataFrame):
        for sex in SexType:
            for g in SexBehaviours[sex]:
                index = selector(population, sex=sex, sex_behaviour=g.value)
                population.loc[index, "num_partners"] = (
                    self.short_term_partners[sex][g.value].rvs(size=sum(index)))


def selector(population, **kwargs):
    index = reduce(operator.and_,
                   (population[kw] == val for kw, val in kwargs.items()))
    return index
