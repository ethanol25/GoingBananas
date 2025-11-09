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
                    reward = 15
                    self.maze[new_pos[0], new_pos[1]] = 0
                    info['message'] = "🍌 Collected banana! +15"
                
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

# --- New Pre-training and Server Setup ---

# Create ONE global agent and ONE environment just for training
global_agent = QLearningAgent(64, 4)
training_env = MazeEnv()
pretraining_task: Optional[asyncio.Task] = None

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

async def pretrain_agent(episodes=5000, report_interval=500):
    """Pre-train the global agent in the background"""
    global global_agent
    print(f"Starting global pre-training for {episodes} episodes...")
    
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
        
        if (episode + 1) % report_interval == 0:
            print(f"Global Pre-training... Episode {episode + 1}/{episodes}")
        
        # Allow other server tasks to run
        await asyncio.sleep(0)
    
    global_agent.is_pretrained = True
    global_agent.save_model()
    print("!!! GLOBAL PRE-TRAINING COMPLETE! Model saved. !!!")

@app.on_event("startup")
async def startup_event():
    """Run pre-training on server startup"""
    global pretraining_task
    if not global_agent.load_model():
        print("No saved model found. Starting pre-training in background...")
        pretraining_task = asyncio.create_task(pretrain_agent())
    else:
        print("Loaded pre-trained model from disk.")

@app.get("/")
async def get_root():
    return FileResponse("static/index.html")

# --- New Concurrent AI Loop ---
async def ai_game_loop(websocket: WebSocket, env: MazeEnv, agent: QLearningAgent, shared_state: dict, stats: dict):
    """A separate, concurrent loop to manage AI movement during a race."""
    
    while True:
        try:
            if shared_state["racing"]:
                action = agent.get_action(env._get_state(), training=False)
                next_state, reward, terminated, truncated, info = env.step(action, is_player=False)
                stats["steps"] += 1
                
                maze_state = env.get_maze_state()
                await websocket.send_json({
                    "type": "state", "maze": maze_state["maze"],
                    "agent_pos": maze_state["agent_pos"], "player_pos": maze_state["player_pos"],
                    "stats": {
                        "episode": stats["episode"], "steps": stats["steps"], "epsilon": agent.epsilon,
                        "total_reward": 0, "player_score": env.player_score, "agent_score": env.agent_score
                    }
                })
                
                if terminated:
                    shared_state["racing"] = False
                    await websocket.send_json({"type": "win", "winner": "ai"})
                
                await asyncio.sleep(0.1) # AI move speed
            else:
                # Loop is idle, just sleep
                await asyncio.sleep(0.1)
        
        except WebSocketDisconnect:
            break # Client disconnected
        except Exception as e:
            print(f"Error in AI loop: {e}")
            await asyncio.sleep(0.1)

# --- New WebSocket Endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("INFO:     connection open")
    
    # Create a LOCAL, per-user environment
    user_env = MazeEnv() 
    
    # Use the GLOBAL pre-trained agent
    global global_agent 
    
    # Constants for the "dummy" training visualization
    DUMMY_TRAIN_EPISODES = 10
    DUMMY_TRAIN_VISUAL_UPDATE_RATE = 1

    shared_state = {"racing": False, "training": False} # 'training' is not used by AI loop
    stats = {"episode": 0, "steps": 0}
    
    # Start the concurrent AI loop for this user
    ai_task = asyncio.create_task(
        ai_game_loop(websocket, user_env, global_agent, shared_state, stats)
    )
    
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
        
        # --- PLAYER MESSAGE LOOP ---
        while True:
            data = await websocket.receive_json()

            if data["action"] == "start_auto_train":
                # This is now a "dummy" loop just for visualization.
                # It does NOT train the global agent.
                await websocket.send_json({
                    "type": "log",
                    "message": f"Starting visual training for {DUMMY_TRAIN_EPISODES} episodes..."
                })
                
                for ep in range(DUMMY_TRAIN_EPISODES):
                    state, _ = user_env.reset()
                    stats["episode"] += 1
                    
                    if ep % DUMMY_TRAIN_VISUAL_UPDATE_RATE == 0:
                        maze_state = user_env.get_maze_state()
                        await websocket.send_json({
                            "type": "state", "maze": maze_state["maze"],
                            "agent_pos": user_env.agent_pos, "player_pos": user_env.player_pos,
                            "stats": {
                                "episode": stats["episode"], "steps": 0, "epsilon": global_agent.epsilon,
                                "total_reward": 0, "player_score": user_env.player_score, "agent_score": user_env.agent_score
                            }
                        })
                        await asyncio.sleep(0.05) # Visual delay
                
                # Check if the *real* agent is ready (it should be)
                if not global_agent.is_pretrained:
                    print("WARN: User is ready, but global pre-training is not yet complete.")
                    await websocket.send_json({"type": "log", "message": "Global agent is still warming up..."})
                
                print("!!! DUMMY TRAINING COMPLETE. SENDING MESSAGE. !!!")
                await websocket.send_json({"type": "training_complete"})
            
            elif data["action"] == "race":
                shared_state["racing"] = True
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
                if shared_state["racing"]:
                    direction = data["direction"]
                    _, reward, terminated, _, info = user_env.step(direction, is_player=True)
                    
                    if 'message' in info and info['message']:
                        await websocket.send_json({"type": "log", "message": info['message'], "player": True})
                    
                    maze_state = user_env.get_maze_state()
                    await websocket.send_json({
                        "type": "state", "maze": maze_state["maze"],
                        "agent_pos": maze_state["agent_pos"], "player_pos": maze_state["player_pos"],
                        "stats": {
                            "episode": stats["episode"], "steps": stats["steps"], "epsilon": global_agent.epsilon,
                            "total_reward": 0, "player_score": user_env.player_score, "agent_score": user_env.agent_score
                        }
                    })
                    
                    if terminated:
                        shared_state["racing"] = False # Stop the AI loop
                        await websocket.send_json({"type": "win", "winner": "player"})
                        
            elif data["action"] == "reset":
                state, _ = user_env.reset()
                stats["episode"] = 0
                stats["steps"] = 0
                shared_state["racing"] = False
                
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
        ai_task.cancel()
        print("INFO:     connection closed")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)