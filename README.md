# Grid World Agent

A reinforcement learning project where an agent navigates a grid world, collecting rewards while avoiding an intelligent enemy, obstacles, and penalty zones.

## Features
- Dynamic grid-world environment with obstacles, penalty zones, and rewards.
- Random agent (baseline) and Deep Q-Learning (DQL) agent.
- Training and visualization of agent performance.

## Installation

### Using Python
1. Clone the repository:
   ```
   git clone https://github.com/ZakLr/Reinforcement-Learning-Grid-World.git
   ```

   ```
   cd grid-world-agent
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```


### Using Docker
1. Build and run:
   ```
   docker build -t grid-world-agent .
   ```

   ```
   docker run grid-world-agent
   ```


## Usage

### Train the DQL Agent
```
python src/train.py
```


### Test and Visualize
```
python src/test_visualize.py
```


View the agent's performance with Matplotlib plots.

## License
MIT License (see 'LICENSE' file).