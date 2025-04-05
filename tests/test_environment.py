import pytest
from src.environment import AdvancedGridWorld

def test_reset():
    env = AdvancedGridWorld(grid_size=(5, 5))
    state = env.reset()
    assert state == (0, 0, 4, 4)  # Agent at (0,0), Enemy at (4,4)

def test_step():
    env = AdvancedGridWorld(grid_size=(5, 5))
    env.reset()
    next_state, reward, done = env.step(1)  # Move down
    assert next_state[0] == 1  # Agent y should increase
    assert reward == -0.1  # Base penalty
    assert not done