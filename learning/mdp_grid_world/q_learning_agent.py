import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from learning.mdp_grid_world.actions import Action


class QLearningAgent:
    def __init__(self, state_space_size, action_space_size, learning_rate, discount_ratio):
        self.__state_space_size = state_space_size
        self.__action_space_size = action_space_size
        self.__alpha = learning_rate
        self.__gamma = discount_ratio
        self.__actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        self.__q_table = self.__initialize_q_table(self.__state_space_size, self.__action_space_size)
    
    @staticmethod
    def __initialize_q_table(state_space_size, action_space_size):
        return np.random.randn(state_space_size, state_space_size, action_space_size)
    
    @property
    def learning_rate(self):
        return self.__alpha
    
    @property
    def discount_ratio(self):
        return self.__gamma
    
    @property
    def actions(self):
        return self.__actions
    
    @property
    def q_table(self):
        return self.__q_table
    
    def choose_action(self, current_state):
        return np.argmax(self.__q_table[current_state])
    
    def update_q_values(self, current_state, action, reward, next_state, done):
        """ Update Q-values using the Q-learning update rule. """
        current_q = self.__q_table[current_state[0], current_state[1]][action]  # Get current Q-value
        
        if done:
            target_q = reward  # No future rewards if terminal state
        else:
            max_next_q = max(self.__q_table[next_state[0], next_state[1]])  # Best Q-value of next state
            target_q = reward + self.__gamma * max_next_q  # Compute target

        # Q-learning update rule
        self.__q_table[current_state][action] += self.__alpha * (target_q - current_q)

    