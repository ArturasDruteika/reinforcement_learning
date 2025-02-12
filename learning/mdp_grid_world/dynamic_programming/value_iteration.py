from typing import List, Tuple, Union
import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from learning.mdp_grid_world.grid_world_environment.environment import GridWorldEnvironment, RewardValues
from learning.mdp_grid_world.grid_world_environment.actions import Action


class ValueIteration:
    """
    A class that implements the Value Iteration algorithm for solving Markov Decision Processes (MDPs)
    in a grid world environment.
    """
    
    def __init__(
        self,
        environment: GridWorldEnvironment,
        action_state_size: int,
        gamma: float,
        theta: float
    ) -> None:
        """
        Initializes the ValueIteration class.

        Args:
            environment (GridWorldEnvironment): The grid world environment.
            action_state_size (int): Number of possible actions per state.
            gamma (float): Discount factor.
            theta (float): Convergence threshold.
        """
        self.__env = environment
        self.__action_state_size = action_state_size
        self.__gamma = gamma
        self.__theta = theta
        self.__state_values = self.__init_state_values()
        self.__policy_table = self.__init_random_policy()

    def __init_state_values(self) -> np.ndarray:
        """
        Initializes state values to zero.

        Returns:
            np.ndarray: A grid of zeros with the same shape as the environment.
        """
        return np.zeros(self.__env.grid.shape)

    def __init_random_policy(self) -> np.ndarray:
        """
        Initializes a random deterministic policy where each state has a random action.

        Returns:
            np.ndarray: Random deterministic policy for each state.
        """
        return np.random.choice(list(Action), size=self.__env.grid.shape)

    def __calculate_state_value(
        self, 
        reward: float, 
        next_state_value: float,
        done: bool
    ) -> float:
        """
        Calculates the state value using the Bellman equation.

        Args:
            reward (float): Reward received after the transition.
            next_state_value (float): Value of the next state.
            done (bool): Whether the next state is terminal.

        Returns:
            float: The calculated state value.
        """
        return reward + self.__gamma * next_state_value * (not done)

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
    def state_values(self) -> np.ndarray:
        return self.__state_values

    @property
    def policy_table(self) -> np.ndarray:
        return self.__policy_table

    def run_value_iteration(self) -> None:
        """
        Executes the Value Iteration algorithm to compute optimal state values and policy.
        """
        rows, cols = self.__env.grid.shape
        delta = float('inf')  # Initialize to track the maximum change

        while delta > self.__theta:
            delta = 0  # Reset delta for this iteration

            for row in range(rows):
                for col in range(cols):
                    if self.__env.grid[row, col] == RewardValues.HOLE_STATE_VALUE:
                        self.__state_values[row, col] = RewardValues.HOLE_STATE_VALUE
                        self.__policy_table[row, col] = '---- -100 ----'
                        continue
                    if self.__env.grid[row, col] == RewardValues.GOAL_STATE_VALUE:
                        self.__state_values[row, col] = RewardValues.GOAL_STATE_VALUE
                        self.__policy_table[row, col] = '---- +100 ----'
                        continue

                    old_value = self.__state_values[row, col]
                    state_value = float('-inf')  # To find the max over actions

                    for action in Action:
                        action_state_value = 0

                        for transition_prob, next_state, reward, done in self.__env.grid_transition_dynamics[(row, col)][action]:
                            action_state_value += transition_prob * self.__calculate_state_value(
                                reward,
                                self.__state_values[next_state],
                                done
                            )

                        if action_state_value > state_value:
                            state_value = action_state_value
                            self.__policy_table[row, col] = action

                    # Update the state value
                    self.__state_values[row, col] = state_value

                    # Track the maximum change (for convergence check)
                    delta = max(delta, np.abs(old_value - state_value))

    def test(self) -> None:
        """
        Tests the optimal policy using the environment.
        The simulation stops when the goal is reached or the agent falls into a hole.
        """
        current_state = (0, 0)
        best_action = self.__policy_table[current_state]

        while True:
            print(f"Current state: {current_state}")
            print(f"Best action: {Action(best_action)}")

            next_state, reward, done = self.__env.step(current_state, best_action)
            print(f"Next state: {next_state}, Reward: {reward}, Done: {done}")
            print()

            if reward == RewardValues.GOAL_STATE_VALUE:
                print("Goal reached!")
                break
            elif reward == RewardValues.HOLE_STATE_VALUE:
                print("Agent fell into a hole!")
                break

            current_state = next_state
            best_action = self.__policy_table[current_state]


if __name__ == '__main__':
    env = GridWorldEnvironment(0.9)
    value_iteration = ValueIteration(
        environment=env, 
        action_state_size=len(Action), 
        gamma=0.99, 
        theta=1e-4
    )
    value_iteration.run_value_iteration()
    value_iteration.test()

    print()
    print(value_iteration.state_values)
    print()
    print(value_iteration.policy_table)
