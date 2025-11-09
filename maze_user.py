import asyncio
import json
import numpy as np
from typing import Optional
import uvicorn

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
                    reward = 50
                    self.maze[new_pos[0], new_pos[1]] = 0
                    info['message'] = "🍌 Collected banana! +50"
                
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

# --- Q-Learning Agent (with new Pre-training fields) ---
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
        self.is_pretrained = False # <-- NEW

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

    def save_model(self, path="model.npy"): # <-- NEW
        np.save(path, self.q_table)
    
    def load_model(self, path="model.npy"): # <-- NEW
        try:
            self.q_table = np.load(path)
            self.is_pretrained = True
            self.epsilon = self.min_epsilon # Already trained
            return True
        except:
            return False

global_training_lock = asyncio.Lock()

# Create ONE global agent and ONE environment just for training
global_agent = QLearningAgent(64, 4)
training_env = MazeEnv()
pretraining_task: Optional[asyncio.Task] = None

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
async def train_global_agent(episodes=10, report_interval=10, websocket: Optional[WebSocket] = None):
    """
    Safely train the global agent in the background.
    Will show a full, step-by-step visualization every N episodes.
    """
    global global_agent

    # --- TUNING KNOB ---
    # Show the full step-by-step animation every N episodes.
    # 1 = Show every episode.
    # 3 = Show episode 3, 6, 9, etc.
    VISUALIZATION_EPISODE_INTERVAL = 3 # <-- Adjust this to change speed
    
    # Use the lock to ensure only one training happens at a time
    async with global_training_lock:
        print(f"Starting global training for {episodes} episodes...")
        if websocket:
             await websocket.send_json({"type": "log", "message": f"Training started for {episodes} episodes..."})
        
        for episode in range(episodes):
            state, _ = training_env.reset()
            done = False
            steps = 0

            # --- Decide if this episode should be visualized ---
            is_visualized_episode = (episode + 1) % VISUALIZATION_EPISODE_INTERVAL == 0
            is_last_episode = (episode + 1) == episodes
            
            # We only send updates if the websocket exists AND it's an interval episode (or the last one)
            should_visualize = websocket and (is_visualized_episode or is_last_episode)

            if should_visualize:
                # --- VISUALIZED PATH (Slower, step-by-step) ---
                
                # Send the start-of-episode reset frame
                maze_state = training_env.get_maze_state()
                await websocket.send_json({
                    "type": "training_update",
                    "maze": maze_state["maze"],
                    "agent_pos": maze_state["agent_pos"], # (0, 0)
                    "player_pos": [-1, -1], # Hide player
                    "stats": {
                        "episode": episode + 1, "total_episodes": episodes,
                        "steps": steps, "epsilon": global_agent.epsilon
                    }
                })
                # Add a brief pause BETWEEN episodes so the user can see the reset
                await asyncio.sleep(0.1) 

                # Run the episode step-by-step WITH websocket updates
                while not done and steps < 200:
                    action = global_agent.get_action(state, training=True)
                    next_state, reward, terminated, truncated, _ = training_env.step(action)
                    done = terminated or truncated
                    global_agent.update(state, action, reward, next_state)
                    state = next_state
                    steps += 1
                    
                    # Send step update
                    maze_state = training_env.get_maze_state()
                    await websocket.send_json({
                        "type": "training_update",
                        "maze": maze_state["maze"],
                        "agent_pos": maze_state["agent_pos"],
                        "player_pos": [-1, -1], # Hide player
                        "stats": {
                            "episode": episode + 1, "total_episodes": episodes,
                            "steps": steps, "epsilon": global_agent.epsilon
                        }
                    })
                    await asyncio.sleep(0) # Yield for rendering
            
            else:
                # --- NON-VISUALIZED PATH (Full speed) ---
                # Run the episode step-by-step WITHOUT websocket updates
                while not done and steps < 200:
                    action = global_agent.get_action(state, training=True)
                    next_state, reward, terminated, truncated, _ = training_env.step(action)
                    done = terminated or truncated
                    global_agent.update(state, action, reward, next_state)
                    state = next_state
                    steps += 1
                    # No websocket.send_json() or sleep() here

            # --- Logic for ALL episodes ---
            global_agent.decay_epsilon()
            
            if (episode + 1) % report_interval == 0:
                print(f"Global Training... Episode {episode + 1}/{episodes}")
            
        # --- Training Complete ---
        global_agent.is_pretrained = True
        global_agent.save_model()
        print(f"!!! GLOBAL TRAINING COMPLETE for {episodes} episodes! Model saved. !!!")
        if websocket:
             await websocket.send_json({"type": "log", "message": "Training complete. Preparing race..."})

# --- MODIFIED: Startup now just loads the model ---
@app.on_event("startup")
async def startup_event():
    """On startup, just load the previously saved model"""
    if global_agent.load_model():
        print("Loaded pre-trained model from disk.")
    else:
        print("No saved model found. Agent will be trained by users.")

@app.get("/")
async def get_root():
    return FileResponse("static/index.html")

# --- New WebSocket Endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("INFO:     connection open")
    
    # Create a LOCAL, per-user environment
    user_env = MazeEnv() 
    
    # Use the GLOBAL pre-trained agent
    global global_agent 

    stats = {"episode": 0, "steps": 0}
    
    try:
        # Send initial state
        maze_state = user_env.get_maze_state()
        await websocket.send_json({
            "type": "state", "maze": maze_state["maze"],
            "agent_pos": maze_state["agent_pos"], "player_pos": maze_state["player_pos"],
            "stats": {
                "episode": stats["episode"], "steps": stats["steps"], "epsilon": global_agent.epsilon,
                "total_reward": 0, "player_score": user_env.player_score, "agent_score": user_env.agent_score
            }
        })
        
        while True:
            data = await websocket.receive_json()

            if data["action"] == "start_auto_train":
                # This is Level 1. It ACTUALLY trains the agent.
                stats["level"] = 1
                await websocket.send_json({"type": "log", "message": "Starting Level 1 training (10 episodes)..."})
                
                # Run the REAL training
                await train_global_agent(episodes=10, report_interval=1, websocket=websocket)
                
                # Update stats for display
                state, _ = user_env.reset()
                stats["episode"] = 10 

                maze_state = user_env.get_maze_state()
                
                print("!!! LEVEL 1 TRAINING COMPLETE. SENDING MESSAGE. !!!")
                await websocket.send_json({
                "type": "state", "maze": maze_state["maze"],
                "agent_pos": maze_state["agent_pos"], "player_pos": maze_state["player_pos"],
                "stats": {
                    "episode": stats["episode"], "steps": stats["steps"], "epsilon": global_agent.epsilon,
                    "total_reward": 0, "player_score": user_env.player_score, "agent_score": user_env.agent_score
                }
            })
            
            elif data["action"] == "race":
                # This is for Level 2+.
                stats["level"] += 1
                new_episodes = stats["level"] * 10 # 20, 30, 40...
                
                await websocket.send_json({"type": "log", "message": f"Starting Level {stats['level']} training ({new_episodes} episodes)..."})
                
                # Run the REAL training
                await train_global_agent(episodes=new_episodes, report_interval=max(1, new_episodes // 10), websocket=websocket)                
                # Update stats for display
                stats["episode"] += new_episodes
                
                # Reset the board for the race
                state, _ = user_env.reset()
                stats["steps"] = 0
                
                maze_state = user_env.get_maze_state()
                await websocket.send_json({
                    "type": "state", "maze": maze_state["maze"],
                    "agent_pos": maze_state["agent_pos"], "player_pos": maze_state["player_pos"],
                    "stats": {
                        "episode": stats["episode"], "steps": stats["steps"], "epsilon": global_agent.epsilon,
                        "total_reward": 0, "player_score": user_env.player_score, "agent_score": user_env.agent_score
                    }
                })
                
            elif data["action"] == "player_move":
                # --- This is the new turn-based logic ---
                
                # 1. Player moves
                direction = data["direction"]
                _, reward_p, terminated_p, _, info_p = user_env.step(direction, is_player=True)
                
                if 'message' in info_p and info_p['message']:
                    await websocket.send_json({"type": "log", "message": info_p['message'], "player": True})
                
                if terminated_p:
                    # Player won
                    await websocket.send_json({"type": "win", "winner": "player"})
                else:
                    # 2. AI moves immediately after
                    action_ai = global_agent.get_action(user_env._get_state(), training=False)
                    _, reward_ai, terminated_ai, _, info_ai = user_env.step(action_ai, is_player=False)
                    
                    if terminated_ai:
                        # AI won
                        await websocket.send_json({"type": "win", "winner": "ai"})
                
                # 3. Send ONE update with both new positions
                maze_state = user_env.get_maze_state()
                await websocket.send_json({
                    "type": "state", "maze": maze_state["maze"],
                    "agent_pos": user_env.agent_pos, # AI's new position
                    "player_pos": user_env.player_pos, # Player's new position
                    "stats": {
                        "episode": stats["episode"], "steps": stats["steps"], "epsilon": global_agent.epsilon,
                        "total_reward": 0, "player_score": user_env.player_score, "agent_score": user_env.agent_score
                    }
                })
                        
            elif data["action"] == "reset":
                state, _ = user_env.reset()
                stats["level"] = 0
                stats["episode"] = 0
                stats["steps"] = 0
                
                maze_state = user_env.get_maze_state()
                await websocket.send_json({
                    "type": "state", "maze": maze_state["maze"],
                    "agent_pos": maze_state["agent_pos"], "player_pos": maze_state["player_pos"],
                    "stats": {
                        "episode": stats["episode"], "steps": stats["steps"], "epsilon": global_agent.epsilon,
                        "total_reward": 0, "player_score": user_env.player_score, "agent_score": user_env.agent_score
                    }
                })
                    
    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        # Clean up the AI task when the player disconnects
        print("INFO:     connection closed")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)