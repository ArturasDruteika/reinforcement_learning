from tqdm import tqdm
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from learning.mdp_grid_world.environment import GridWorldEnvironment
from learning.mdp_grid_world.q_learning_agent import QLearningAgent


def train_q_learning(env, agent, num_episodes):
    for _ in tqdm(range(num_episodes)):
        current_state = (0, 0)
        done = False
        
        while not done:
            action = agent.choose_action(current_state)
            next_state, reward, done = env.step(current_state, action)
            agent.update_q_values(current_state, action, reward, next_state, done)
            current_state = next_state
            
def test_trained_agent(env, agent):
    """Test the trained Q-learning agent by running an episode."""
    
    print("\n🔹 **Testing Trained Agent**")

    state = (0, 0)
    total_reward = 0
    steps = 0
    max_steps = 50  # Prevent infinite loops in case of bad policies

    while True:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(state, action)
        total_reward += reward
        steps += 1

        print(f"Step {steps}: State {state} → Action {action} → Next State {next_state}, Reward {reward}")

        if done or steps >= max_steps:
            break
        state = next_state  # Move to the next state

    print(f"\n✅ **Test Completed: Total Reward: {total_reward}, Steps Taken: {steps}**")
    assert steps < max_steps, "❌ Test Failed: Agent did not reach goal efficiently."
    assert total_reward > 0, "❌ Test Failed: Agent did not collect positive rewards."



if __name__ == '__main__':
    env = GridWorldEnvironment()
    agent = QLearningAgent(state_space_size=4, action_space_size=4, learning_rate=0.1, discount_ratio=0.9)
    num_episodes = 1000
    
    train_q_learning(env, agent, num_episodes)
    test_trained_agent(env, agent)
