import numpy as np
import random

class AdvancedGridWorld:
    def __init__(self, grid_size=(10, 10), dynamic_obstacles=2, penalty_zones=3, rewards=5):
        self.grid_size = grid_size
        self.start = (0, 0)
        self.agent_pos = list(self.start)
        self.enemy_pos = [grid_size[0] - 1, grid_size[1] - 1]
        self.dynamic_obstacles_count = dynamic_obstacles
        self.penalty_zones_count = penalty_zones
        self.rewards_count = rewards
        self.reset()

    def reset(self):
        """Reset the game state and randomly place obstacles, penalty zones, and rewards."""
        self.agent_pos = list(self.start)
        self.enemy_pos = [self.grid_size[0] - 1, self.grid_size[1] - 1]

        def generate_random_positions(count):
            positions = set()
            while len(positions) < count:
                pos = (random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1))
                if pos != self.start and pos != tuple(self.enemy_pos):
                    positions.add(pos)
            return list(positions)

        self.dynamic_obstacles = generate_random_positions(self.dynamic_obstacles_count)
        self.penalty_zones = generate_random_positions(self.penalty_zones_count)
        self.rewards = generate_random_positions(self.rewards_count)
        return tuple(self.agent_pos) + tuple(self.enemy_pos)  # State: (agent_x, agent_y, enemy_x, enemy_y)

    def step(self, action):
        """Move the agent based on the action."""
        next_pos = list(self.agent_pos)
        if action == 0 and self.agent_pos[0] > 0:  # Up
            next_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.grid_size[0] - 1:  # Down
            next_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:  # Left
            next_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.grid_size[1] - 1:  # Right
            next_pos[1] += 1

        reward = -0.1  # Base step penalty
        if tuple(next_pos) in self.dynamic_obstacles:
            reward -= 1  # Penalty for hitting obstacle
            next_pos = self.agent_pos  # Stay in place
        elif tuple(next_pos) in self.penalty_zones:
            reward -= 2  # Higher penalty for penalty zone
        elif tuple(next_pos) in self.rewards:
            reward += 5  # Reward collected
            self.rewards.remove(tuple(next_pos))

        self.agent_pos = next_pos
        done = self.agent_pos == self.enemy_pos
        return tuple(self.agent_pos) + tuple(self.enemy_pos), reward, done

    def move_enemy(self):
        """Enemy moves toward the agent, avoiding obstacles and penalty zones."""
        best_move = None
        best_distance = float('inf')

        for action in range(4):  # Up, Down, Left, Right
            next_pos = list(self.enemy_pos)
            if action == 0 and self.enemy_pos[0] > 0: next_pos[0] -= 1
            elif action == 1 and self.enemy_pos[0] < self.grid_size[0] - 1: next_pos[0] += 1
            elif action == 2 and self.enemy_pos[1] > 0: next_pos[1] -= 1
            elif action == 3 and self.enemy_pos[1] < self.grid_size[1] - 1: next_pos[1] += 1

            if (tuple(next_pos) not in self.dynamic_obstacles and 
                tuple(next_pos) not in self.penalty_zones):
                distance = abs(next_pos[0] - self.agent_pos[0]) + abs(next_pos[1] - self.agent_pos[1])
                if distance < best_distance:
                    best_distance = distance
                    best_move = next_pos

        if best_move:
            self.enemy_pos = best_move

    def render(self):
        """Render the grid with the current game state."""
        grid = [["." for _ in range(self.grid_size[1])] for _ in range(self.grid_size[0])]
        for obs in self.dynamic_obstacles: grid[obs[0]][obs[1]] = "#"
        for penalty in self.penalty_zones: grid[penalty[0]][penalty[1]] = "X"
        for reward in self.rewards: grid[reward[0]][reward[1]] = "$"
        grid[self.agent_pos[0]][self.agent_pos[1]] = "A"
        grid[self.enemy_pos[0]][self.enemy_pos[1]] = "E"
        return "\n".join([" ".join(row) for row in grid]) + "\n"