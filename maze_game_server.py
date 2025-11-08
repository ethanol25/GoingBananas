from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import asyncio
from typing import List, Tuple
import random

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Custom Maze Environment
class MazeEnv(gym.Env):
    """Custom Maze Environment compatible with gym interface"""
    def __init__(self, size=8):
        super(MazeEnv, self).__init__()
        self.size = size
        self.action_space = spaces.Discrete(4)  # Up, Right, Down, Left
        self.observation_space = spaces.Discrete(size * size)
        b = random.choice([0,4]) # whether or not to spawn a banana instead of floor
        
        # Define maze (0=floor, 1=wall, 2=start, 3=goal, 4=banana)
        self.maze = np.array([
            [2, b, b, 1, b, b, b, b],
            [b, 1, b, 1, b, 1, 1, b],
            [b, 1, b, b, b, b, b, b],
            [b, b, b, 1, 1, 1, 1, b],
            [1, 1, b, b, b, b, b, b],
            [b, b, b, 1, b, 1, 1, b],
            [b, 1, b, b, b, b, 1, b],
            [b, 1, 1, 1, b, b, b, 3],
        ], dtype=object) # Enables functions to run
        
        self.original_maze = self.maze.copy()  # Store original for reset
        self.visited_cells = set()
        self.checkpoints = set()
        self.total_steps = 0

        self.start_pos = (0, 0)
        self.goal_pos = (7, 7)
        self.agent_pos = list(self.start_pos)
        self.player_pos = list(self.start_pos)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = list(self.start_pos)
        self.player_pos = list(self.start_pos)
        self.maze = self.original_maze.copy()  # Reset maze to restore power-ups
        self.visited_cells = set()
        self.checkpoints = set()
        self.total_steps = 0
        return self._get_state(), {}
    
    def _get_state(self):
        return self.agent_pos[0] * self.size + self.agent_pos[1]
    
    def step(self, action, is_player=False):
        # Actions: 0=Up, 1=Right, 2=Down, 3=Left
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        if is_player:
            pos = self.player_pos
        else:
            pos = self.agent_pos
            
        new_pos = [
            pos[0] + moves[action][0],
            pos[1] + moves[action][1]
        ]
        
# Initialize reward
        reward = -0.1  # Small penalty for each step
        self.total_steps += 1
        terminated = False
        info = {}
        
        # Calculate Manhattan distance to goal (for reward shaping)
        old_distance = abs(pos[0] - self.goal_pos[0]) + abs(pos[1] - self.goal_pos[1])
        
        # Check boundaries and walls
        if (0 <= new_pos[0] < self.size and 
            0 <= new_pos[1] < self.size):
            
            cell_type = self.maze[new_pos[0], new_pos[1]]
            
            # Hit a wall - penalty and don't move
            if cell_type == 1:
                reward = -5
                info['message'] = "Hit a wall! -5"
                new_distance = old_distance  # Distance unchanged
            else:
                # Valid move
                if is_player:
                    self.player_pos = new_pos
                else:
                    self.agent_pos = new_pos
                
                # Calculate new distance
                new_distance = abs(new_pos[0] - self.goal_pos[0]) + abs(new_pos[1] - self.goal_pos[1])
                
                # Distance-based reward shaping
                if new_distance < old_distance:
                    reward += 0.5  # Getting closer
                elif new_distance > old_distance:
                    reward -= 0.5  # Moving away
                
                # Check for special cells
                if cell_type == 4:  # Banana
                    reward = 5
                    info['message'] = "Picked up a banana! +5"
                
                # Exploration bonus
                cell_tuple = tuple(new_pos)
                if cell_tuple not in self.visited_cells:
                    self.visited_cells.add(cell_tuple)
                    reward += 1
                    info['message'] = info.get('message', '') + " [+1 exploration]"
                
                # Check for goal
                if tuple(new_pos) == self.goal_pos:
                    reward = 100
                    # Bonus for efficiency (fewer steps)
                    if self.total_steps < 50:
                        reward += 20
                        info['message'] = "🏆 Goal! +100 (+20 speed bonus!)"
                    else:
                        info['message'] = "🏆 Goal reached! +100"
                    terminated = True
        else:
            # Out of bounds penalty
            reward = -10
            info['message'] = "Out of bounds! -10"
            new_distance = old_distance
        
        # Time penalty (encourages faster solutions)
        if self.total_steps > 100:
            reward -= 0.5
        
        return self._get_state(), reward, terminated, False, info
    
    def get_maze_state(self):
        """Return current maze state for visualization"""
        return {
            "maze": self.maze.tolist(),
            "agent_pos": self.agent_pos,
            "player_pos": self.player_pos,
            "goal_pos": list(self.goal_pos)
        }

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))
        self.learning_rate = 0.8 # How much Q-values are adjusted
        self.discount_factor = 0.95 # Determines the importance of future awards
        self.epsilon = 1.0 # Epsilon Greedy Policy
        self.epsilon_decay = 0.995 # Decay towards greedy actions
        self.min_epsilon = 0.01
        
    def get_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        
        new_value = old_value + self.learning_rate * (
            reward + self.discount_factor * next_max - old_value
        )
        self.q_table[state, action] = new_value
        
    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


# Global instances
env = MazeEnv()
agent = QLearningAgent(env.observation_space.n, env.action_space.n)


@app.get("/")
async def get():
    return FileResponse("static/index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    episode = 0
    total_reward = 0
    steps = 0
    training = False
    racing = False
    
    try:
        # Send initial state
        maze_state = env.get_maze_state()
        await websocket.send_json({
            "type": "state",
            "maze": maze_state["maze"],
            "agent_pos": maze_state["agent_pos"],
            "player_pos": maze_state["player_pos"],
            "stats": {
                "episode": episode,
                "steps": steps,
                "epsilon": agent.epsilon,
                "total_reward": total_reward
            }
        })
        
        while True:
            # Receive commands from client
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
                
                if data["action"] == "train":
                    training = True
                    racing = False
                    await websocket.send_json({
                        "type": "log",
                        "message": "Training started"
                    })
                elif data["action"] == "stop":
                    training = False
                    racing = False
                elif data["action"] == "race":
                    # Start race mode
                    racing = True
                    training = False
                    state, _ = env.reset()
                    steps = 0
                    
                    maze_state = env.get_maze_state()
                    await websocket.send_json({
                        "type": "state",
                        "maze": maze_state["maze"],
                        "agent_pos": maze_state["agent_pos"],
                        "player_pos": maze_state["player_pos"],
                        "stats": {
                            "episode": episode,
                            "steps": steps,
                            "epsilon": agent.epsilon,
                            "total_reward": 0
                        }
                    })
                    
                elif data["action"] == "player_move":
                    if racing:
                        # Move player
                        direction = data["direction"]
                        _, reward, terminated, _, info = env.step(direction, is_player=True)
                        
                        # Send message if there's feedback
                        if 'message' in info and info['message']:
                            await websocket.send_json({
                                "type": "log",
                                "message": info['message'],
                                "player": True
                            })
                        
                        maze_state = env.get_maze_state()
                        await websocket.send_json({
                            "type": "state",
                            "maze": maze_state["maze"],
                            "agent_pos": maze_state["agent_pos"],
                            "player_pos": maze_state["player_pos"],
                            "stats": {
                                "episode": episode,
                                "steps": steps,
                                "epsilon": agent.epsilon,
                                "total_reward": 0
                            }
                        })
                        
                        if terminated:
                            racing = False
                            await websocket.send_json({
                                "type": "win",
                                "winner": "player"
                            })
                        
                elif data["action"] == "reset":
                    state, _ = env.reset()
                    episode = 0
                    total_reward = 0
                    steps = 0
                    racing = False
                    training = False
                    
                    maze_state = env.get_maze_state()
                    await websocket.send_json({
                        "type": "state",
                        "maze": maze_state["maze"],
                        "agent_pos": maze_state["agent_pos"],
                        "player_pos": maze_state["player_pos"],
                        "stats": {
                            "episode": episode,
                            "steps": steps,
                            "epsilon": agent.epsilon,
                            "total_reward": 0
                        }
                    })
                    
            except asyncio.TimeoutError:
                pass
            
            # Training loop
            if training:
                state, _ = env.reset()
                done = False
                episode += 1
                total_reward = 0
                steps = 0
                
                while not done and steps < 200:
                    action = agent.get_action(state, training=True)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    
                    agent.update(state, action, reward, next_state)
                    
                    total_reward += reward
                    steps += 1
                    state = next_state
                    
                    # Send update every few steps
                    if steps % 5 == 0:
                        maze_state = env.get_maze_state()
                        await websocket.send_json({
                            "type": "state",
                            "maze": maze_state["maze"],
                            "agent_pos": maze_state["agent_pos"],
                            "player_pos": maze_state["player_pos"],
                            "stats": {
                                "episode": episode,
                                "steps": steps,
                                "epsilon": agent.epsilon,
                                "total_reward": total_reward
                            }
                        })
                        await asyncio.sleep(0.05)
                
                agent.decay_epsilon()
                
                if terminated:
                    await websocket.send_json({
                        "type": "log",
                        "message": f"Episode {episode}: SUCCESS in {steps} steps! Reward: {total_reward:.1f}"
                    })
                
                await asyncio.sleep(0.1)
                
            # Race mode
            elif racing:
                # AI makes a move
                action = agent.get_action(env._get_state(), training=False)
                next_state, reward, terminated, truncated, info = env.step(action)
                steps += 1
                
                # Send AI move feedback
                if 'message' in info and info['message']:
                    await websocket.send_json({
                        "type": "log",
                        "message": info['message'],
                        "player": False
                    })
                
                maze_state = env.get_maze_state()
                await websocket.send_json({
                    "type": "state",
                    "maze": maze_state["maze"],
                    "agent_pos": maze_state["agent_pos"],
                    "player_pos": maze_state["player_pos"],
                    "stats": {
                        "episode": episode,
                        "steps": steps,
                        "epsilon": agent.epsilon,
                        "total_reward": 0
                    }
                })
                
                if terminated:
                    racing = False
                    await websocket.send_json({
                        "type": "win",
                        "winner": "ai"
                    })
                
                await asyncio.sleep(0.2)  # AI moves slower for fair competition
            else:
                await asyncio.sleep(0.1)
                
    except WebSocketDisconnect:
        print("Client disconnected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)