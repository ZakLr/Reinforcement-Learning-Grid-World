import numpy as np
from src.environment import AdvancedGridWorld
from src.agents import DQLAgent

def train_agent(episodes=1000, batch_size=32):
    env = AdvancedGridWorld(grid_size=(10, 10))
    state_size = 4  # (agent_x, agent_y, enemy_x, enemy_y)
    action_size = 4
    agent = DQLAgent(state_size, action_size)

    for episode in range(episodes):
        state = env.reset()  # Already returns (agent_x, agent_y, enemy_x, enemy_y)
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            env.move_enemy()
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        if episode % 100 == 0:
            print(f"Episode {episode}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    agent.save()
    print("Training complete. Model saved to 'examples/trained_model/dql_agent.pth'.")

if __name__ == "__main__":
    train_agent()