from typing import List
import time

import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from learning.mdp_grid_world.grid_world_environment.environment import GridWorldEnvironment, RewardValues
from learning.mdp_grid_world.grid_world_environment.actions import Action


class PolicyIteration:
    """
    Implements the Policy Iteration algorithm for solving GridWorld MDPs.
    
    Attributes:
        environment (GridWorldEnvironment): The GridWorld environment.
        action_state_size (int): Number of possible actions per state.
        gamma (float): Discount factor for future rewards.
        theta (float): Convergence threshold for policy evaluation.
    """

    def __init__(
        self, 
        environment: GridWorldEnvironment,
        action_state_size: int,
        gamma: float,
        theta: float
    ) -> None:
        """
        Initializes the PolicyIteration class with environment and hyperparameters.

        Args:
            environment (GridWorldEnvironment): The grid world environment.
            action_state_size (int): The number of possible actions.
            gamma (float): Discount factor (0 < gamma <= 1).
            theta (float): Convergence threshold for policy evaluation.
        """
        self.__env = environment
        self.__action_state_size = action_state_size
        self.__gamma = gamma
        self.__theta = theta
        self.__actions = [action for action in range(action_state_size)]
        self.__state_values = self.__init_state_values()
        self.__policy_table = self.__init_random_policy()
        self.__policy_stable = False
        
    def __init_state_values(self) -> np.ndarray:
        """
        Initializes state values to zero.

        Returns:
            np.ndarray: A grid of zeros with the same shape as the environment.
        """
        return np.zeros(self.__env.grid.shape)
    
    def __init_random_policy(self) -> np.ndarray:
        """ 
        Initializes a random stochastic policy where all actions have equal probability. 

        Returns:
            np.ndarray: Policy table with uniform probability for each action.
        """
        grid_height, grid_width = self.__env.grid.shape
        policy = np.ones((grid_height, grid_width, self.__action_state_size)) / self.__action_state_size
        return policy

    def __calculate_state_action_subvalue(
        self, 
        action_prob: float, 
        transition_prob: float, 
        reward: float, 
        next_state_value: float, 
        done: bool
    ) -> float:
        """
        Calculates the subvalue for a state-action pair using the Bellman equation.

        Args:
            action_prob (float): Probability of taking the action under the current policy.
            transition_prob (float): Transition probability to the next state.
            reward (float): Reward received after the transition.
            next_state_value (float): Value of the next state.
            done (bool): Whether the next state is terminal.

        Returns:
            float: The calculated subvalue.
        """
        return action_prob * transition_prob * (reward + self.__gamma * next_state_value * (not done))
    
    @property
    def action_state_size(self) -> int:
        return self.__action_state_size
    
    @property
    def gamma(self) -> float:
        return self.__gamma
    
    @property
    def theta(self) -> float:
        return self.__theta
    
    @property
    def actions(self) -> List[int]:
        return self.__actions
    
    @property
    def state_values(self) -> np.ndarray:
        return self.__state_values
    
    @property
    def policy_table(self) -> np.ndarray:
        return self.__policy_table
    
    @property
    def policy_stable(self) -> bool:
        return self.__policy_stable
    
    def __evaluate_policy(self) -> None:
        """
        Evaluates the current policy by updating state values until convergence.
        Uses the Bellman expectation equation for policy evaluation.
        """
        rows, cols = self.__env.grid.shape
        while True:
            new_state_values = np.zeros_like(self.__state_values)
            
            for i in range(rows):
                for j in range(cols):
                    if self.__env.grid[i, j] == RewardValues.HOLE_STATE_VALUE:
                        new_state_values[i, j] = RewardValues.HOLE_STATE_VALUE
                        continue
                    if self.__env.grid[i, j] == RewardValues.GOAL_STATE_VALUE:
                        new_state_values[i, j] = RewardValues.GOAL_STATE_VALUE
                        continue
                    
                    current_state_value = 0
                    
                    for action in self.__actions:
                        action_prob = self.__policy_table[i, j, action]
                        
                        for transition_prob, next_state, reward, done in self.__env.grid_transition_dynamics[(i, j)][Action(action)]:
                            current_state_value += self.__calculate_state_action_subvalue(
                                action_prob,  
                                transition_prob, 
                                reward,  
                                self.__state_values[next_state],
                                done
                            )
                        
                    new_state_values[i, j] = current_state_value  
            
            # Compute max change in state values across all states
            delta = np.max(np.abs(new_state_values - self.__state_values))
                    
            self.__state_values = new_state_values  # Update state values
            # Stop if changes are smaller than the threshold
            if delta < self.__theta:
                break
            
    def __improve_policy(self) -> None:
        """
        Improves the policy greedily based on the current state values.
        Updates the policy to be greedy with respect to the current value function.
        """
        policy_stable = True
        
        for i in range(self.__env.grid.shape[0]):
            for j in range(self.__env.grid.shape[1]):
                if self.__env.grid[i, j] in [RewardValues.HOLE_STATE_VALUE, RewardValues.GOAL_STATE_VALUE]:
                    continue  # No policy improvement needed for terminal states
                old_action = np.argmax(self.__policy_table[i, j])
                
                action_values = np.zeros(self.__action_state_size)
                for action in self.__actions:
                    for transition_prob, next_state, reward, done in self.__env.grid_transition_dynamics[(i, j)][Action(action)]:
                        action_values[action] += transition_prob * (reward + self.__gamma * self.__state_values[next_state] * (not done))
                
                best_action = np.argmax(action_values)
                new_policy = np.zeros(self.__action_state_size)
                new_policy[best_action] = 1.0
                self.__policy_table[i, j] = new_policy
                
                if old_action != best_action:
                    policy_stable = False
        
        self.__policy_stable = policy_stable
        
    def run_policy_iteration(self) -> None:
        """
        Runs the Policy Iteration algorithm:
        1. Evaluate the current policy.
        2. Improve the policy based on the updated values.
        3. Repeat until the policy is stable.
        """
        iteration = 0
        while not self.__policy_stable:
            self.__evaluate_policy()
            self.__improve_policy()
            iteration += 1
        print(f"Optimal policy found on iteration: {iteration}")
        

    def test(self) -> None:
        """
        Tests the optimal policy using the given environment.
        Stops when the goal is reached or the agent falls off the grid.
        """
        current_state = (0, 0)
        best_action = np.argmax(self.__policy_table[current_state])
        
        while True:
            print(f"Current state: {current_state}")
            print(f"Best action: {Action(best_action)}")
            
            next_state, reward, done = self.__env.step(current_state, Action(best_action))
            print(f"Next state: {next_state}, Reward: {reward}, Done: {done}")
            print()
            
            if reward == RewardValues.GOAL_STATE_VALUE:
                print("Goal reached!")
                break
            elif reward == RewardValues.HOLE_STATE_VALUE:
                print("Agent fell off the grid!")
                break
            
            current_state = next_state
            best_action = np.argmax(self.__policy_table[current_state])
            
    def display_policy(self) -> None:
        """
        Displays the best policy for each state in the grid using arrows to represent actions.
        """
        # Mapping action indices to arrows (assuming actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        arrow_mapping = {
            0: '↑',  # Action.UP
            1: '↓',  # Action.DOWN
            2: '←',  # Action.LEFT
            3: '→'   # Action.RIGHT
        }

        grid_height, grid_width = self.__env.grid.shape

        for i in range(grid_height):
            row_policy = ''
            for j in range(grid_width):
                # Check for terminal states
                if self.__env.grid[i, j] == RewardValues.HOLE_STATE_VALUE:
                    row_policy += ' ❌ '  # Hole
                elif self.__env.grid[i, j] == RewardValues.GOAL_STATE_VALUE:
                    row_policy += ' ⭐ '   # Goal
                else:
                    # Select the best action (highest probability)
                    best_action = np.argmax(self.__policy_table[i, j])
                    row_policy += f' {arrow_mapping[best_action]} '
            print(row_policy)

        print(self.__state_values)

    
    
if __name__ == '__main__':
    env = GridWorldEnvironment(transition_prob=0.9)
    policy_iteration = PolicyIteration(env, action_state_size=4, gamma=0.99, theta=1e-8)
    
    start_time = time.time()  # Start the timer
    policy_iteration.run_policy_iteration()
    end_time = time.time()    # End the timer

    policy_iteration.test()
    policy_iteration.display_policy()
    print()
    
    execution_time = end_time - start_time
    print(f"Time taken to find the optimal policy: {execution_time:.6f} seconds")
    

