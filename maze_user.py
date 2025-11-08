from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import asyncio
from typing import List, Tuple

app = FastAPI()

# Custom Maze Environment
class MazeEnv(gym.Env):
    """Custom Maze Environment compatible with gym interface"""
    
    def __init__(self, size=8):
        super(MazeEnv, self).__init__()
        self.size = size
        self.action_space = spaces.Discrete(4)  # Up, Right, Down, Left
        self.observation_space = spaces.Discrete(size * size)
        
        # Define maze (0=path, 1=wall, 2=start, 3=goal, 4=trap, 5=powerup, 6=checkpoint)
        self.maze = np.array([
            [2, 0, 0, 1, 0, 5, 0, 0],
            [0, 1, 0, 1, 0, 1, 1, 0],
            [0, 1, 4, 0, 0, 0, 5, 6],
            [0, 0, 0, 1, 1, 1, 1, 0],
            [1, 1, 0, 4, 0, 0, 0, 5],
            [0, 5, 0, 1, 0, 1, 1, 0],
            [0, 1, 0, 0, 4, 0, 1, 6],
            [0, 1, 1, 1, 0, 0, 0, 3],
        ])
        
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
                if cell_type == 4:  # Trap
                    reward = -20
                    info['message'] = "💥 Hit a trap! -20"
                    
                elif cell_type == 5:  # Power-up
                    reward = 15
                    self.maze[new_pos[0], new_pos[1]] = 0  # Remove after collection
                    info['message'] = "⭐ Collected power-up! +15"
                    
                elif cell_type == 6:  # Checkpoint
                    checkpoint_id = tuple(new_pos)
                    if checkpoint_id not in self.checkpoints:
                        self.checkpoints.add(checkpoint_id)
                        reward = 10
                        info['message'] = "🚩 Checkpoint reached! +10"
                
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
        self.learning_rate = 0.8
        self.discount_factor = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
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
        <title>Q-Learning Maze Race</title>
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
            .subtitle {
                color: #888;
                margin-top: -10px;
                margin-bottom: 20px;
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
                position: relative;
            }
            .path { background: #2a2a4e; }
            .wall { background: #0f3460; border: 2px solid #1a1a3e; }
            .start { background: #4caf50; }
            .goal { background: #ff5722; }
            .trap { 
                background: #d32f2f;
                animation: pulse 1s infinite;
            }
            .trap::before {
                content: '💥';
                font-size: 28px;
            }
            .powerup {
                background: #ffd700;
                animation: glow 1.5s infinite;
            }
            .powerup::before {
                content: '⭐';
                font-size: 28px;
            }
            .checkpoint {
                background: #9c27b0;
                animation: fade 2s infinite;
            }
            .checkpoint::before {
                content: '🚩';
                font-size: 28px;
            }
            @keyframes pulse {
                0%, 100% { box-shadow: 0 0 5px rgba(211, 47, 47, 0.5); }
                50% { box-shadow: 0 0 20px rgba(211, 47, 47, 1); }
            }
            @keyframes glow {
                0%, 100% { box-shadow: 0 0 10px rgba(255, 215, 0, 0.5); }
                50% { box-shadow: 0 0 25px rgba(255, 215, 0, 1); }
            }
            @keyframes fade {
                0%, 100% { opacity: 0.7; }
                50% { opacity: 1; }
            }
            .agent { 
                background: #00d4ff;
                box-shadow: 0 0 15px rgba(0, 212, 255, 0.6);
            }
            .agent::after {
                content: '🤖';
                font-size: 32px;
                position: absolute;
            }
            .player {
                background: #ffd700;
                box-shadow: 0 0 15px rgba(255, 215, 0, 0.6);
            }
            .player::after {
                content: '👤';
                font-size: 32px;
                position: absolute;
            }
            .both-sprites {
                background: linear-gradient(135deg, #00d4ff 50%, #ffd700 50%);
                box-shadow: 0 0 20px rgba(255, 215, 0, 0.8);
            }
            .both-sprites::after {
                content: '🤖👤';
                font-size: 24px;
                position: absolute;
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
            .player-button {
                background: #ffd700;
            }
            .player-button:hover {
                background: #ffed4e;
            }
            #stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                width: 100%;
                max-width: 800px;
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
            .player-stat .stat-value {
                color: #ffd700;
            }
            #log {
                background: #16213e;
                border: 2px solid #0f3460;
                border-radius: 8px;
                padding: 15px;
                width: 100%;
                max-width: 800px;
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
            .log-player {
                border-left-color: #ffd700;
            }
            .log-win {
                border-left-color: #4caf50;
                background: rgba(76, 175, 80, 0.1);
            }
            .controls-help {
                background: #16213e;
                border: 2px solid #0f3460;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
                max-width: 800px;
                text-align: center;
            }
            .controls-help h3 {
                color: #ffd700;
                margin-top: 0;
            }
            .key-hint {
                display: inline-block;
                background: #0f3460;
                padding: 5px 10px;
                border-radius: 4px;
                margin: 0 5px;
                font-weight: bold;
                color: #ffd700;
            }
        </style>
    </head>
    <body>
        <h1>🎮 Q-Learning Maze Race</h1>
        <p class="subtitle">Human vs AI - Race to the goal!</p>
        
        <div class="controls-help">
            <h3>🎯 Player Controls & Rewards</h3>
            <p>Use arrow keys: 
                <span class="key-hint">↑</span>
                <span class="key-hint">→</span>
                <span class="key-hint">↓</span>
                <span class="key-hint">←</span>
                or WASD to move
            </p>
            <p style="margin-top: 10px; font-size: 14px;">
                <span style="color: #ffd700;">⭐ Power-up: +15</span> | 
                <span style="color: #d32f2f;">💥 Trap: -20</span> | 
                <span style="color: #9c27b0;">🚩 Checkpoint: +10</span> | 
                <span style="color: #666;">Wall: -5</span>
            </p>
        </div>
        
        <div id="stats">
            <div class="stat">
                <div class="stat-label">🤖 AI Episode</div>
                <div class="stat-value" id="episode">0</div>
            </div>
            <div class="stat">
                <div class="stat-label">🤖 AI Steps</div>
                <div class="stat-value" id="steps">0</div>
            </div>
            <div class="stat">
                <div class="stat-label">🤖 Epsilon</div>
                <div class="stat-value" id="epsilon">1.00</div>
            </div>
            <div class="stat player-stat">
                <div class="stat-label">👤 Player Steps</div>
                <div class="stat-value" id="player-steps">0</div>
            </div>
            <div class="stat">
                <div class="stat-label">🤖 AI Wins</div>
                <div class="stat-value" id="ai-wins">0</div>
            </div>
            <div class="stat player-stat">
                <div class="stat-label">👤 Player Wins</div>
                <div class="stat-value" id="player-wins">0</div>
            </div>
        </div>
        
        <div id="maze-container">
            <div id="maze" class="maze-grid"></div>
        </div>
        
        <div id="controls">
            <button onclick="startTraining()">🤖 Start AI Training</button>
            <button onclick="stopTraining()">⏸ Stop Training</button>
            <button onclick="startRace()">🏁 Start Race!</button>
            <button onclick="resetEnvironment()">🔄 Reset</button>
        </div>
        
        <div id="log"></div>
        
        <script>
            const ws = new WebSocket(`ws://${window.location.host}/ws`);
            let training = false;
            let racing = false;
            let playerSteps = 0;
            let aiWins = 0;
            let playerWins = 0;
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                if (data.type === 'state') {
                    renderMaze(data.maze, data.agent_pos, data.player_pos);
                    updateStats(data.stats);
                } else if (data.type === 'log') {
                    addLog(data.message, data.player);
                } else if (data.type === 'win') {
                    handleWin(data.winner);
                }
            };
            
            // Keyboard controls for player
            document.addEventListener('keydown', function(e) {
                if (!racing) return;
                
                let action = null;
                if (e.key === 'ArrowUp' || e.key === 'w' || e.key === 'W') {
                    action = 0;
                } else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') {
                    action = 1;
                } else if (e.key === 'ArrowDown' || e.key === 's' || e.key === 'S') {
                    action = 2;
                } else if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') {
                    action = 3;
                }
                
                if (action !== null) {
                    e.preventDefault();
                    playerSteps++;
                    document.getElementById('player-steps').textContent = playerSteps;
                    ws.send(JSON.stringify({action: 'player_move', direction: action}));
                }
            });
            
            function renderMaze(maze, agentPos, playerPos) {
                const mazeDiv = document.getElementById('maze');
                mazeDiv.style.gridTemplateColumns = `repeat(${maze.length}, 50px)`;
                mazeDiv.innerHTML = '';
                
                maze.forEach((row, i) => {
                    row.forEach((cell, j) => {
                        const cellDiv = document.createElement('div');
                        cellDiv.className = 'cell';
                        
                        const hasAgent = agentPos[0] === i && agentPos[1] === j;
                        const hasPlayer = playerPos[0] === i && playerPos[1] === j;
                        
                        if (hasAgent && hasPlayer) {
                            cellDiv.className += ' both-sprites';
                        } else if (hasAgent) {
                            cellDiv.className += ' agent';
                        } else if (hasPlayer) {
                            cellDiv.className += ' player';
                        } else if (cell === 0) {
                            cellDiv.className += ' path';
                        } else if (cell === 1) {
                            cellDiv.className += ' wall';
                        } else if (cell === 2) {
                            cellDiv.className += ' start';
                        } else if (cell === 3) {
                            cellDiv.className += ' goal';
                        } else if (cell === 4) {
                            cellDiv.className += ' trap';
                        } else if (cell === 5) {
                            cellDiv.className += ' powerup';
                        } else if (cell === 6) {
                            cellDiv.className += ' checkpoint';
                        }
                        
                        mazeDiv.appendChild(cellDiv);
                    });
                });
            }
            
            function updateStats(stats) {
                document.getElementById('episode').textContent = stats.episode;
                document.getElementById('steps').textContent = stats.steps;
                document.getElementById('epsilon').textContent = stats.epsilon.toFixed(3);
            }
            
            function addLog(message, isPlayer = false) {
                const logDiv = document.getElementById('log');
                const entry = document.createElement('div');
                entry.className = 'log-entry' + (isPlayer ? ' log-player' : '');
                const icon = isPlayer ? '👤' : '🤖';
                entry.textContent = `[${new Date().toLocaleTimeString()}] ${icon} ${message}`;
                logDiv.appendChild(entry);
                logDiv.scrollTop = logDiv.scrollHeight;
            }
            
            function handleWin(winner) {
                racing = false;
                const entry = document.createElement('div');
                entry.className = 'log-entry log-win';
                
                if (winner === 'player') {
                    playerWins++;
                    entry.textContent = `[${new Date().toLocaleTimeString()}] 🏆 PLAYER WINS in ${playerSteps} steps!`;
                    document.getElementById('player-wins').textContent = playerWins;
                } else {
                    aiWins++;
                    entry.textContent = `[${new Date().toLocaleTimeString()}] 🏆 AI WINS!`;
                    document.getElementById('ai-wins').textContent = aiWins;
                }
                
                document.getElementById('log').appendChild(entry);
                document.getElementById('log').scrollTop = document.getElementById('log').scrollHeight;
            }
            
            function startTraining() {
                ws.send(JSON.stringify({action: 'train'}));
                addLog('Training started...');
                training = true;
                racing = false;
            }
            
            function stopTraining() {
                ws.send(JSON.stringify({action: 'stop'}));
                addLog('Training stopped');
                training = false;
            }
            
            function startRace() {
                ws.send(JSON.stringify({action: 'race'}));
                addLog('Race started! Use arrow keys or WASD to move!', true);
                racing = true;
                training = false;
                playerSteps = 0;
                document.getElementById('player-steps').textContent = playerSteps;
            }
            
            function resetEnvironment() {
                ws.send(JSON.stringify({action: 'reset'}));
                addLog('Environment reset');
                racing = false;
                training = false;
                playerSteps = 0;
                document.getElementById('player-steps').textContent = playerSteps;
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