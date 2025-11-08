from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import asyncio
from typing import List, Tuple
import random

app = FastAPI()

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
        ], dtype=object)
        
        self.start_pos = (0, 0)
        self.goal_pos = (7, 7)
        self.agent_pos = list(self.start_pos)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = list(self.start_pos)
        return self._get_state(), {}
    
    def _get_state(self):
        return self.agent_pos[0] * self.size + self.agent_pos[1]
    
    def step(self, action):
        # Actions: 0=Up, 1=Right, 2=Down, 3=Left
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        new_pos = [
            self.agent_pos[0] + moves[action][0],
            self.agent_pos[1] + moves[action][1]
        ]
        
        # Check boundaries
        if (0 <= new_pos[0] < self.size and 
            0 <= new_pos[1] < self.size and 
            self.maze[new_pos[0], new_pos[1]] != 1):
            self.agent_pos = new_pos
        
        # Calculate reward
        reward = -0.1  # Small penalty for each step
        terminated = False
        
        if tuple(self.agent_pos) == self.goal_pos:
            reward = 100
            terminated = True
        
        return self._get_state(), reward, terminated, False, {}
    
    def get_maze_state(self):
        """Return current maze state for visualization"""
        return {
            "maze": self.maze.tolist(),
            "agent_pos": self.agent_pos,
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
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Q-Learning Maze Game</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                display: flex;
                flex-direction: column;
                align-items: center;
                background: #1a1a2e;
                color: #eee;
                margin: 0;
                padding: 20px;
            }
            h1 {
                color: #00d4ff;
            }
            #maze-container {
                display: inline-block;
                border: 3px solid #00d4ff;
                border-radius: 10px;
                padding: 10px;
                background: #16213e;
                box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
            }
            .maze-grid {
                display: grid;
                gap: 2px;
                background: #0f3460;
                padding: 5px;
                border-radius: 5px;
            }
            .cell {
                width: 50px;
                height: 50px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 24px;
                border-radius: 4px;
                transition: all 0.3s;
            }
            .path { background: #2a2a4e; }
            .wall { background: #0f3460; border: 2px solid #1a1a3e; }
            .start { background: #4caf50; }
            .goal { background: #ff5722; }
            .agent { background: #00d4ff; position: relative; }
            .agent::after {
                content: '🤖';
                font-size: 32px;
            }
            #controls {
                margin: 20px;
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                justify-content: center;
            }
            button {
                padding: 12px 24px;
                font-size: 16px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                background: #00d4ff;
                color: #1a1a2e;
                font-weight: bold;
                transition: all 0.3s;
            }
            button:hover {
                background: #00b8d4;
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0, 212, 255, 0.3);
            }
            button:disabled {
                background: #666;
                cursor: not-allowed;
                transform: none;
            }
            #stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                width: 100%;
                max-width: 600px;
                margin: 20px 0;
            }
            .stat {
                background: #16213e;
                padding: 15px;
                border-radius: 8px;
                border: 2px solid #0f3460;
            }
            .stat-label {
                color: #888;
                font-size: 12px;
                text-transform: uppercase;
            }
            .stat-value {
                color: #00d4ff;
                font-size: 24px;
                font-weight: bold;
            }
            #log {
                background: #16213e;
                border: 2px solid #0f3460;
                border-radius: 8px;
                padding: 15px;
                width: 100%;
                max-width: 600px;
                height: 150px;
                overflow-y: auto;
                font-family: monospace;
                font-size: 12px;
            }
            .log-entry {
                margin: 5px 0;
                padding: 5px;
                border-left: 3px solid #00d4ff;
                padding-left: 10px;
            }
        </style>
    </head>
    <body>
        <h1>🎮 Q-Learning Maze Game</h1>
        
        <div id="stats">
            <div class="stat">
                <div class="stat-label">Episode</div>
                <div class="stat-value" id="episode">0</div>
            </div>
            <div class="stat">
                <div class="stat-label">Steps</div>
                <div class="stat-value" id="steps">0</div>
            </div>
            <div class="stat">
                <div class="stat-label">Epsilon</div>
                <div class="stat-value" id="epsilon">1.00</div>
            </div>
            <div class="stat">
                <div class="stat-label">Reward</div>
                <div class="stat-value" id="reward">0</div>
            </div>
        </div>
        
        <div id="maze-container">
            <div id="maze" class="maze-grid"></div>
        </div>
        
        <div id="controls">
            <button onclick="startTraining()">Start Training</button>
            <button onclick="stopTraining()">Stop Training</button>
            <button onclick="testAgent()">Test Agent</button>
            <button onclick="resetEnvironment()">Reset</button>
        </div>
        
        <div id="log"></div>
        
        <script>
            const ws = new WebSocket(`ws://${window.location.host}/ws`);
            let training = false;
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                if (data.type === 'state') {
                    renderMaze(data.maze, data.agent_pos);
                    updateStats(data.stats);
                } else if (data.type === 'log') {
                    addLog(data.message);
                }
            };
            
            function renderMaze(maze, agentPos) {
                const mazeDiv = document.getElementById('maze');
                mazeDiv.style.gridTemplateColumns = `repeat(${maze.length}, 50px)`;
                mazeDiv.innerHTML = '';
                
                maze.forEach((row, i) => {
                    row.forEach((cell, j) => {
                        const cellDiv = document.createElement('div');
                        cellDiv.className = 'cell';
                        
                        if (agentPos[0] === i && agentPos[1] === j) {
                            cellDiv.className += ' agent';
                        } else if (cell === 0) {
                            cellDiv.className += ' path';
                        } else if (cell === 1) {
                            cellDiv.className += ' wall';
                        } else if (cell === 2) {
                            cellDiv.className += ' start';
                        } else if (cell === 3) {
                            cellDiv.className += ' goal';
                        }
                        
                        mazeDiv.appendChild(cellDiv);
                    });
                });
            }
            
            function updateStats(stats) {
                document.getElementById('episode').textContent = stats.episode;
                document.getElementById('steps').textContent = stats.steps;
                document.getElementById('epsilon').textContent = stats.epsilon.toFixed(3);
                document.getElementById('reward').textContent = stats.total_reward.toFixed(1);
            }
            
            function addLog(message) {
                const logDiv = document.getElementById('log');
                const entry = document.createElement('div');
                entry.className = 'log-entry';
                entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
                logDiv.appendChild(entry);
                logDiv.scrollTop = logDiv.scrollHeight;
            }
            
            function startTraining() {
                ws.send(JSON.stringify({action: 'train'}));
                addLog('Training started...');
            }
            
            function stopTraining() {
                ws.send(JSON.stringify({action: 'stop'}));
                addLog('Training stopped');
            }
            
            function testAgent() {
                ws.send(JSON.stringify({action: 'test'}));
                addLog('Testing trained agent...');
            }
            
            function resetEnvironment() {
                ws.send(JSON.stringify({action: 'reset'}));
                addLog('Environment reset');
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    episode = 0
    total_reward = 0
    steps = 0
    training = False
    
    try:
        # Send initial state
        maze_state = env.get_maze_state()
        await websocket.send_json({
            "type": "state",
            "maze": maze_state["maze"],
            "agent_pos": maze_state["agent_pos"],
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
                    await websocket.send_json({
                        "type": "log",
                        "message": "Training started"
                    })
                elif data["action"] == "stop":
                    training = False
                elif data["action"] == "test":
                    # Test the agent
                    state, _ = env.reset()
                    done = False
                    test_steps = 0
                    
                    while not done and test_steps < 100: # Runs max 100 steps against user
                        action = agent.get_action(state, training=False)
                        state, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
                        
                        maze_state = env.get_maze_state()
                        await websocket.send_json({
                            "type": "state",
                            "maze": maze_state["maze"],
                            "agent_pos": maze_state["agent_pos"],
                            "stats": {
                                "episode": episode,
                                "steps": test_steps,
                                "epsilon": agent.epsilon,
                                "total_reward": reward
                            }
                        })
                        
                        test_steps += 1
                        await asyncio.sleep(0.1)
                    
                    if terminated:
                        await websocket.send_json({
                            "type": "log",
                            "message": f"Goal reached in {test_steps} steps!"
                        })
                    else:
                        await websocket.send_json({
                            "type": "log",
                            "message": "Test failed to reach goal"
                        })
                        
                elif data["action"] == "reset":
                    state, _ = env.reset()
                    episode = 0
                    total_reward = 0
                    steps = 0
                    
            except asyncio.TimeoutError:
                pass
            
            # Training loop
            if training:
                state, _ = env.reset()
                done = False
                episode += 1
                total_reward = 0
                steps = 0
                
                while not done and steps < 200: # Runs max 200 steps for training
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
            else:
                await asyncio.sleep(0.1)
                
    except WebSocketDisconnect:
        print("Client disconnected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)