import numpy as np
import matplotlib.pyplot as plt
import time
from src.environment import AdvancedGridWorld
from src.agents import DQLAgent, RandomAgent

def test_agent(episodes=5, use_random=False):
    env = AdvancedGridWorld(grid_size=(10, 10))
    state_size = 4
    action_size = 4
    agent = RandomAgent(env) if use_random else DQLAgent(state_size, action_size)
    if not use_random:
        agent.load()

    steps_survived = []
    rewards_collected = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        total_reward = 0

        print(f"\n=== Episode {episode + 1} ===\n")
        while not done:
            print(env.render())
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            env.move_enemy()
            state = next_state
            steps += 1
            total_reward += reward
            time.sleep(0.5)

        print(f"Survived {steps} steps, Total Reward: {total_reward:.2f}")
        steps_survived.append(steps)
        rewards_collected.append(total_reward)

    # Visualization
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(steps_survived, label="Steps Survived")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Steps Survived per Episode")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(rewards_collected, label="Total Reward", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rewards Collected per Episode")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test with trained DQL agent
    test_agent(episodes=5, use_random=False)
    # Optionally test with random agent for comparison
    # test_agent(episodes=5, use_random=True)