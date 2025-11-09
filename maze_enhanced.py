import asyncio
import json
import numpy as np
from typing import Optional
import uvicorn
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import gymnasium as gym
from gymnasium import spaces

# --- Maze Environment (Unchanged) ---
class MazeEnv(gym.Env):
    def __init__(self, size=8):
        super(MazeEnv, self).__init__()
        self.size = size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(size * size)
        self.maze = np.array([
            [2, 0, 0, 1, 0, 0, 0, 4],
            [0, 1, 0, 1, 0, 1, 1, 0],
            [0, 1, 4, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 4],
            [1, 1, 0, 4, 0, 4, 0, 0],
            [0, 4, 0, 1, 0, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [4, 1, 1, 1, 0, 0, 0, 3],
        ])
        self.original_maze = self.maze.copy()
        self.visited_cells = set()
        self.total_steps = 0
        self.start_pos = (0, 0)
        self.goal_pos = (7, 7)
        self.agent_pos = list(self.start_pos)
        self.player_pos = list(self.start_pos)
        self.player_score = 0
        self.agent_score = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = list(self.start_pos)
        self.player_pos = list(self.start_pos)
        self.maze = self.original_maze.copy()
        self.visited_cells = set()
        self.total_steps = 0
        self.player_score = 0
        self.agent_score = 0
        return self._get_state(), {}

    def _get_state(self):
        return self.agent_pos[0] * self.size + self.agent_pos[1]

    def step(self, action, is_player=False):
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        pos = self.player_pos if is_player else self.agent_pos
        new_pos = [pos[0] + moves[action][0], pos[1] + moves[action][1]]
        
        reward = -0.1
        self.total_steps += 1
        terminated = False
        info = {}
        old_distance = abs(pos[0] - self.goal_pos[0]) + abs(pos[1] - self.goal_pos[1])

        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            reward = -10
            info['message'] = "Out of bounds! -10"
        else:
            cell_type = self.maze[new_pos[0], new_pos[1]]
            if cell_type == 1:
                reward = -5
                info['message'] = "Hit a wall! -5"
            else:
                if is_player:
                    self.player_pos = new_pos
                else:
                    self.agent_pos = new_pos
                
                new_distance = abs(new_pos[0] - self.goal_pos[0]) + abs(new_pos[1] - self.goal_pos[1])
                if new_distance < old_distance: reward += 0.5
                elif new_distance > old_distance: reward -= 0.5
                
                if cell_type == 4:
                    reward = 25
                    self.maze[new_pos[0], new_pos[1]] = 0
                    info['message'] = "🍌 Collected banana! +25"
                
                cell_tuple = tuple(new_pos)
                if cell_tuple not in self.visited_cells:
                    self.visited_cells.add(cell_tuple)
                    reward += 1
                    info['message'] = info.get('message', '') + " [+1 exploration]"
                
                if tuple(new_pos) == self.goal_pos:
                    reward = 100
                    if self.total_steps < 50:
                        reward += 20
                        info['message'] = "🏆 Goal! +100 (+20 speed bonus!)"
                    else:
                        info['message'] = "🏆 Goal reached! +100"
                    terminated = True
        
        if self.total_steps > 100:
            reward -= 0.5

        if is_player:
            self.player_score += reward
        else:
            self.agent_score += reward
        
        return self._get_state(), reward, terminated, False, info

    def get_maze_state(self):
        return {
            "maze": self.maze.tolist(), "agent_pos": self.agent_pos,
            "player_pos": self.player_pos, "goal_pos": list(self.goal_pos)
        }

# --- Q-Learning Agent ---
class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))
        self.learning_rate = 0.8
        self.discount_factor = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        self.is_pretrained = False
        self.total_training_episodes = 0

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

    def save_model(self, path="model.npy"):
        np.save(path, self.q_table)
    
    def load_model(self, path="model.npy"):
        try:
            self.q_table = np.load(path)
            self.is_pretrained = True
            self.epsilon = self.min_epsilon
            return True
        except:
            return False

# --- Level Configuration ---
LEVEL_CONFIG = {
    1: {"episodes": 50, "name": "Beginner", "description": "Agent trains for 50 episodes"},
    2: {"episodes": 100, "name": "Intermediate", "description": "Agent trains for 100 episodes"},
    3: {"episodes": 200, "name": "Advanced", "description": "Agent trains for 200 episodes"},
    4: {"episodes": 400, "name": "Expert", "description": "Agent trains for 400 episodes"},
    5: {"episodes": 800, "name": "Master", "description": "Agent trains for 800 episodes"},
}

global_training_lock = asyncio.Lock()
global_agent = QLearningAgent(64, 4)
training_env = MazeEnv()

# Simple in-memory user stats (use database for production)
user_stats = {}

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

async def train_global_agent(episodes=10, websocket=None):
    """Train the global agent and send progress updates"""
    global global_agent
    
    async with global_training_lock:
        print(f"Starting training for {episodes} episodes...")
        
        for episode in range(episodes):
            state, _ = training_env.reset()
            done = False
            steps = 0
            
            while not done and steps < 200:
                action = global_agent.get_action(state, training=True)
                next_state, reward, terminated, truncated, _ = training_env.step(action)
                done = terminated or truncated
                global_agent.update(state, action, reward, next_state)
                state = next_state
                steps += 1
            
            global_agent.decay_epsilon()
            global_agent.total_training_episodes += 1
            
            # Send progress updates every 10%
            if websocket and (episode + 1) % max(1, episodes // 10) == 0:
                progress = int((episode + 1) / episodes * 100)
                await websocket.send_json({
                    "type": "training_progress",
                    "progress": progress,
                    "episode": episode + 1,
                    "total": episodes
                })
            
            await asyncio.sleep(0)
        
        global_agent.is_pretrained = True
        global_agent.save_model()
        print(f"Training complete for {episodes} episodes!")

@app.on_event("startup")
async def startup_event():
    """Load saved model on startup"""
    if global_agent.load_model():
        print("Loaded pre-trained model from disk.")
    else:
        print("No saved model found. Agent will be trained by users.")

@app.get("/")
async def get_root():
    return FileResponse("static/index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("INFO:     connection open")
    
    user_env = MazeEnv()
    
    # Generate unique user ID (use auth in production)
    user_id = str(uuid.uuid4())
    
    # Initialize or load user stats
    if user_id not in user_stats:
        user_stats[user_id] = {
            "total_points": 0,
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "current_level": 1
        }
    
    stats = {
        "level": user_stats[user_id]["current_level"],
        "steps": 0,
        "round_in_progress": False
    }
    
    try:
        # Send initial state with user stats
        maze_state = user_env.get_maze_state()
        await websocket.send_json({
            "type": "state",
            "maze": maze_state["maze"],
            "agent_pos": maze_state["agent_pos"],
            "player_pos": maze_state["player_pos"],
            "user_stats": user_stats[user_id],
            "current_level": stats["level"],
            "level_info": LEVEL_CONFIG[stats["level"]],
            "stats": {
                "steps": stats["steps"],
                "epsilon": global_agent.epsilon,
                "player_score": user_env.player_score,
                "agent_score": user_env.agent_score,
                "total_training_episodes": global_agent.total_training_episodes
            }
        })
        
        while True:
            data = await websocket.receive_json()

            if data["action"] == "start_level":
                # Start a new level - train the agent first
                level = data.get("level", stats["level"])
                stats["level"] = level
                user_stats[user_id]["current_level"] = level
                
                level_config = LEVEL_CONFIG[level]
                episodes = level_config["episodes"]
                
                await websocket.send_json({
                    "type": "log",
                    "message": f"🎯 Starting {level_config['name']} Level! Agent training for {episodes} episodes..."
                })
                
                await websocket.send_json({
                    "type": "training_start",
                    "level": level,
                    "episodes": episodes
                })
                
                # Train the agent with progress updates
                await train_global_agent(episodes=episodes, websocket=websocket)
                
                # Reset the board for the race
                state, _ = user_env.reset()
                stats["steps"] = 0
                stats["round_in_progress"] = True
                
                maze_state = user_env.get_maze_state()
                await websocket.send_json({
                    "type": "training_complete",
                    "level": level
                })
                
                await websocket.send_json({
                    "type": "state",
                    "maze": maze_state["maze"],
                    "agent_pos": maze_state["agent_pos"],
                    "player_pos": maze_state["player_pos"],
                    "user_stats": user_stats[user_id],
                    "current_level": stats["level"],
                    "level_info": level_config,
                    "stats": {
                        "steps": stats["steps"],
                        "epsilon": global_agent.epsilon,
                        "player_score": user_env.player_score,
                        "agent_score": user_env.agent_score,
                        "total_training_episodes": global_agent.total_training_episodes
                    }
                })
                
            elif data["action"] == "player_move":
                if not stats["round_in_progress"]:
                    continue
                
                # Player moves
                direction = data["direction"]
                _, reward_p, terminated_p, _, info_p = user_env.step(direction, is_player=True)
                stats["steps"] += 1
                
                if 'message' in info_p and info_p['message']:
                    await websocket.send_json({
                        "type": "log",
                        "message": info_p['message'],
                        "player": True
                    })
                
                if terminated_p:
                    # Player won
                    stats["round_in_progress"] = False
                    user_stats[user_id]["wins"] += 1
                    user_stats[user_id]["games_played"] += 1
                    points_earned = int(user_env.player_score)
                    user_stats[user_id]["total_points"] += points_earned
                    
                    await websocket.send_json({
                        "type": "win",
                        "winner": "player",
                        "points_earned": points_earned,
                        "user_stats": user_stats[user_id]
                    })
                else:
                    # AI moves
                    action_ai = global_agent.get_action(user_env._get_state(), training=False)
                    _, reward_ai, terminated_ai, _, info_ai = user_env.step(action_ai, is_player=False)
                    
                    if terminated_ai:
                        # AI won
                        stats["round_in_progress"] = False
                        user_stats[user_id]["losses"] += 1
                        user_stats[user_id]["games_played"] += 1
                        
                        await websocket.send_json({
                            "type": "win",
                            "winner": "ai",
                            "user_stats": user_stats[user_id]
                        })
                
                # Send updated state
                maze_state = user_env.get_maze_state()
                await websocket.send_json({
                    "type": "state",
                    "maze": maze_state["maze"],
                    "agent_pos": user_env.agent_pos,
                    "player_pos": user_env.player_pos,
                    "user_stats": user_stats[user_id],
                    "current_level": stats["level"],
                    "stats": {
                        "steps": stats["steps"],
                        "epsilon": global_agent.epsilon,
                        "player_score": user_env.player_score,
                        "agent_score": user_env.agent_score,
                        "total_training_episodes": global_agent.total_training_episodes
                    }
                })
                        
            elif data["action"] == "reset":
                state, _ = user_env.reset()
                stats["steps"] = 0
                stats["round_in_progress"] = False
                
                maze_state = user_env.get_maze_state()
                await websocket.send_json({
                    "type": "state",
                    "maze": maze_state["maze"],
                    "agent_pos": maze_state["agent_pos"],
                    "player_pos": maze_state["player_pos"],
                    "user_stats": user_stats[user_id],
                    "current_level": stats["level"],
                    "level_info": LEVEL_CONFIG[stats["level"]],
                    "stats": {
                        "steps": stats["steps"],
                        "epsilon": global_agent.epsilon,
                        "player_score": user_env.player_score,
                        "agent_score": user_env.agent_score,
                        "total_training_episodes": global_agent.total_training_episodes
                    }
                })
                    
    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        print("INFO:     connection closed")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)