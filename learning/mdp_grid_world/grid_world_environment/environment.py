from typing import Tuple, Dict
import random

import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from learning.mdp_grid_world.grid_world_environment.actions import Action


class RewardValues:
    """
    Defines the reward values for different states in the GridWorld environment.

    Attributes:
        NORMAL_STATE_VALUE (int): The reward for moving into a normal (non-terminal) state.
            Typically a small negative value to encourage reaching the goal efficiently.
        HOLE_STATE_VALUE (int): The penalty for falling into a hole state, representing a failure.
            This is a large negative value to strongly discourage falling into holes.
        GOAL_STATE_VALUE (int): The reward for reaching the goal state, representing success.
            This is a large positive value to encourage reaching the goal.
        SAME_STATE_VALUE (int): The penalty for attempting to move into an invalid position,
            resulting in staying in the same state. This discourages invalid moves.
    """
    NORMAL_STATE_VALUE = -1
    HOLE_STATE_VALUE = -100
    GOAL_STATE_VALUE = 100
    SAME_STATE_VALUE = -5


class GridWorldEnvironment:
    """
    A 4x4 GridWorld environment for reinforcement learning agents.
    The environment consists of a grid where the agent navigates by taking actions.
    
    Attributes:
        __grid (np.ndarray): A 4x4 numpy array representing state rewards in the environment.
    """

    def __init__(self, transition_prob = 1.0) -> None:
        """
        Initializes the GridWorld environment by creating a grid with rewards.
        """
        self.__transition_prob = transition_prob
        self.__grid: np.ndarray = self.__create_grid()
        self.__grid_transition_dynamics = self.__create_grid_transition_dynamics()

    def __create_grid(self) -> np.ndarray:
        """
        Creates a 4x4 grid representing the environment.
        Positive values indicate rewards, negative values indicate penalties.

        Returns:
            np.ndarray: A 4x4 numpy array representing the grid with rewards.
        """
        grid = np.zeros((4, 4), dtype=np.int16)

        # Define rewards and penalties
        grid[0, 0] = RewardValues.NORMAL_STATE_VALUE
        grid[0, 1] = RewardValues.NORMAL_STATE_VALUE
        grid[0, 2] = RewardValues.NORMAL_STATE_VALUE
        grid[0, 3] = RewardValues.HOLE_STATE_VALUE

        grid[1, 0] = RewardValues.NORMAL_STATE_VALUE
        grid[1, 1] = RewardValues.HOLE_STATE_VALUE
        grid[1, 2] = RewardValues.NORMAL_STATE_VALUE
        grid[1, 3] = RewardValues.NORMAL_STATE_VALUE

        grid[2, 0] = RewardValues.NORMAL_STATE_VALUE
        grid[2, 1] = RewardValues.NORMAL_STATE_VALUE
        grid[2, 2] = RewardValues.HOLE_STATE_VALUE
        grid[2, 3] = RewardValues.NORMAL_STATE_VALUE

        grid[3, 0] = RewardValues.HOLE_STATE_VALUE
        grid[3, 1] = RewardValues.NORMAL_STATE_VALUE
        grid[3, 2] = RewardValues.NORMAL_STATE_VALUE
        grid[3, 3] = RewardValues.GOAL_STATE_VALUE  # Goal state

        return grid
    
    def __create_grid_transition_dynamics(self) -> Dict:
        """
        Creates a transition dynamics dictionary for the grid environment.
        Handles any transition probability dynamically.
        """
        transition_dynamics = {}

        for row in range(self.__grid.shape[0]):
            for col in range(self.__grid.shape[1]):
                state = (row, col)
                transition_dynamics[state] = {}

                for action in Action:
                    intended_next_state = self.__calculate_next_position(state, action, stochastic=False)
                    
                    unintended_actions = [a for a in Action if a != action]  # Other actions
                    unintended_next_states = {a: self.__calculate_next_position(state, a, stochastic=False) for a in unintended_actions}

                    # Initialize probabilities
                    transitions = []
                    intended_prob = self.__transition_prob  # User-defined probability
                    unintended_prob = (1.0 - self.__transition_prob) / len(unintended_actions)  # Spread over other actions

                    # Add intended move
                    transitions.append((
                        intended_prob,
                        intended_next_state,
                        self.__grid[intended_next_state],
                        self.__is_terminal_state(intended_next_state)
                    ))

                    # Add unintended moves
                    for unintended_action, next_state in unintended_next_states.items():
                        transitions.append((
                            unintended_prob,
                            next_state,
                            self.__grid[next_state],
                            self.__is_terminal_state(next_state)
                        ))

                    # Store transitions for (state, action)
                    transition_dynamics[state][action] = transitions

        return transition_dynamics
    
    def __is_terminal_state(self, state: Tuple[int, int]) -> bool:
        """
        Checks if a given state is terminal (goal or hole).
        """
        return self.__grid[state] == RewardValues.HOLE_STATE_VALUE or self.__grid[state] == RewardValues.GOAL_STATE_VALUE

    def __calculate_next_position(self, current_position: Tuple[int, int], action: Action, stochastic: bool = True) -> Tuple[int, int]:
        """
        Calculates the next position based on the action.
        
        Args:
            current_position: The current (row, column) position.
            action: The action to be performed.
            stochastic: If False, removes randomness for deterministic transitions.
            
        Returns:
            Tuple[int, int]: The next position of the agent.
        """
        x, y = current_position
        
        if stochastic and self.__transition_prob < 1.0:
            if random.random() > self.__transition_prob:
                action = random.choice(list(Action))
        
        if action == Action.UP:
            x -= 1
        elif action == Action.DOWN:
            x += 1
        elif action == Action.LEFT:
            y -= 1
        elif action == Action.RIGHT:
            y += 1

        # Ensure the agent stays within the grid boundaries
        if not (0 <= x < 4 and 0 <= y < 4):
            return current_position  # Stay in the same position if out of bounds

        return (x, y)
    
    @property
    def transition_prob(self) -> float:
        """
        Returns the transition probability for the grid environment.

        Returns:
            float: The transition probability.
        """
        return self.__transition_prob

    @property
    def grid(self) -> np.ndarray:
        """
        Returns the grid representation of the environment.

        Returns:
            np.ndarray: A 4x4 numpy array containing the environment's rewards.
        """
        return self.__grid
    
    @property
    def grid_transition_dynamics(self) -> Dict:
        """
        Returns the transition dynamics of the grid environment.

        Returns:
            dict: A dictionary representing the transition dynamics.
        """
        return self.__grid_transition_dynamics

    def step(self, current_position: Tuple[int, int], action: Action) -> Tuple[Tuple[int, int], int, bool]:
        """
        Executes an action in the environment and returns the next state, reward, and done flag.

        Args:
            current_position (Tuple[int, int]): The agent's current position.
            action (Action): The action chosen by the agent.

        Returns:
            Tuple[Tuple[int, int], int, bool]: 
                - The next state (row, column).
                - The reward received after taking the action.
                - A boolean indicating whether the episode is done.
        """
        next_state = self.__calculate_next_position(current_position, action)

        if next_state == current_position:
            reward = RewardValues.SAME_STATE_VALUE
        else:
            # Get the reward from the grid
            reward = self.__grid[next_state]
            
        # Check if game over
        if reward == RewardValues.HOLE_STATE_VALUE:
            done = True  
            return next_state, reward, done

        # Check if the episode should end
        done = reward == RewardValues.GOAL_STATE_VALUE  # Goal reached

        return next_state, reward, done
    
    
if __name__ == '__main__':
    env = GridWorldEnvironment(0.9)
    
    print("\n=== Transition Dynamics ===")
    for state, actions in env.grid_transition_dynamics.items():  # Access private attribute safely
        print(f"\nState {state}:")
        for action, transitions in actions.items():
            print(f"  Action {action.name}:")
            for prob, next_state, reward, done in transitions:
                print(f"    → Prob: {prob:.2f}, Next: {next_state}, Reward: {reward}, Terminal: {done}")
    print("\n===========================")
    
    actions = [Action.DOWN, Action.DOWN, Action.RIGHT, Action.DOWN, Action.RIGHT, Action.RIGHT]
    current_position = (0, 0)
    for action in actions:
        current_position, reward, done = env.step(current_position, action)
        print(f"current position: {current_position}, reward: {reward}, done: {done}")