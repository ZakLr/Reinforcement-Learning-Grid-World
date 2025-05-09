import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

class RandomAgent:
    def __init__(self, env):
        self.env = env
        self.action_size = 4  # Up, Down, Left, Right

    def act(self, state):
        agent_pos = (state[0], state[1])
        enemy_pos = (state[2], state[3])
        best_move = None
        best_action = None
        best_distance = float('inf')

        # Define movement deltas
        action_map = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1),   # Right
        }

        for action, (dx, dy) in action_map.items():
            new_pos = (enemy_pos[0] + dx, enemy_pos[1] + dy)

            # Check boundaries
            if not (0 <= new_pos[0] < self.env.grid_size[0] and 0 <= new_pos[1] < self.env.grid_size[1]):
                continue

            # Avoid obstacles and penalty zones
            if new_pos in self.env.dynamic_obstacles or new_pos in self.env.penalty_zones:
                continue

            # Calculate Manhattan distance
            distance = abs(new_pos[0] - agent_pos[0]) + abs(new_pos[1] - agent_pos[1])
            if distance < best_distance:
                best_distance = distance
                best_move = new_pos
                best_action = action

        # Fall back to a random valid move if no good one found
        if best_action is not None:
            return best_action
        else:
            return random.randint(0, self.action_size - 1)

class DQLAgent:
    def __init__(self, state_size, action_size, model_path="examples/trained_model/dql_agent.pth"):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def _build_model(self):
        """Build a simple neural network with PyTorch."""
        class QNetwork(nn.Module):
            def __init__(self, state_size, action_size):
                super(QNetwork, self).__init__()
                self.fc1 = nn.Linear(state_size, 24)
                self.fc2 = nn.Linear(24, 24)
                self.fc3 = nn.Linear(24, action_size)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                return self.fc3(x)

        return QNetwork(self.state_size, self.action_size)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Shape: [1, state_size]
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states).to(self.device)  # Shape: [batch_size, state_size]
        actions = torch.LongTensor(actions).to(self.device)  # Shape: [batch_size]
        rewards = torch.FloatTensor(rewards).to(self.device)  # Shape: [batch_size]
        next_states = torch.FloatTensor(next_states).to(self.device)  # Shape: [batch_size, state_size]
        dones = torch.FloatTensor(dones).to(self.device)  # Shape: [batch_size]

        # Compute Q-values
        q_values = self.model(states)  # Shape: [batch_size, action_size]
        next_q_values = self.model(next_states).detach()  # Shape: [batch_size, action_size]
        
        # Compute targets
        targets = rewards + (1 - dones) * self.gamma * next_q_values.max(dim=1)[0]  # Shape: [batch_size]
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # Shape: [batch_size]

        # Compute loss and optimize
        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load(self):
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
