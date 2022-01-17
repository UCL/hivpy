import numpy as np
import scipy.stats as stat

from . import sex_behaviour_data as sb

class SexualBehaviourModule:

    def __init__(self, **kwargs):
        # Randomly initialise sexual behaviour group transitions
        self.sex_behaviour_trans_male = np.random.choice(sb.sex_behaviour_trans_male_options)
        self.sex_behaviour_trans_female = np.random.choice(sb.sex_behaviour_trans_female_options)
        self.baseline_risk = sb.baseline_risk # Baseline risk appears to only have one option
        self.sex_mixing_matrix_female = np.random.choice(sb.sex_mixing_matrix_female_options)
        self.sex_mixing_matrix_male = np.random.choice(sb.sex_mixing_matrix_male_options)
        self.short_term_partners_male = np.random.choice(sb.short_term_partners_male_options)
        self.short_term_partners_female = np.random.choice(sb.short_term_partners_female_options)
    
    # Here we need to figure out how to vectorise this which is currently blocked
    # by the gender if statement
    def prob_transition(self, gender, age, i, j):
        """Calculates the probability of transitioning from sexual behaviour
        group i to group j, based on gender and age."""
        if(gender == 1):
            transition_matrix = self.sex_behaviour_trans_female
            gender_index = 1
        else:
            transition_matrix = self.sex_behaviour_trans_male
            gender_index = 0

        age_index = min((int(age)-15)//5, 9)

        risk_factor = self.baseline_risk[age_index][gender_index]

        denominator = transition_matrix[i][0] + risk_factor*sum(transition_matrix[i][1:])

        if(j == 0):
            Probability = transition_matrix[i][0] / denominator
        else:
            Probability = risk_factor*transition_matrix[i][j] / denominator

        return Probability

    def num_short_term_partners(self, population):
        population["short_term_partners"] = np.where(population['gender'] == 'male',
                                                    self.short_term_partners_male
                                                    [population['sex_behaviour']].rvs(),
                                                    self.short_term_partners_female
                                                    [population['sex_behaviour']].rvs())
