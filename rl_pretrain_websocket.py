"""
RL Model Pre-training with FastAPI WebSocket
Demonstrates pre-training an RL agent before user interactions
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import asyncio
import json
import numpy as np
from typing import Optional
from dataclasses import dataclass
import uvicorn

# Simple RL Environment (replace with your actual environment)
class SimpleEnvironment:
    def __init__(self):
        self.state = 0
        self.max_steps = 100
        self.current_step = 0
    
    def reset(self):
        self.state = np.random.randint(0, 10)
        self.current_step = 0
        return self.state
    
    def step(self, action):
        # Simple logic: action moves state, reward based on reaching target
        self.current_step += 1
        self.state = max(0, min(10, self.state + action - 1))
        
        reward = -abs(self.state - 5)  # Target state is 5
        done = self.current_step >= self.max_steps or self.state == 5
        
        return self.state, reward, done

# Simple RL Agent (replace with your actual model)
class RLAgent:
    def __init__(self):
        self.q_table = np.zeros((11, 3))  # 11 states, 3 actions
        self.epsilon = 0.1
        self.alpha = 0.1
        self.gamma = 0.95
        self.is_pretrained = False
    
    def select_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, 3)
        return int(np.argmax(self.q_table[state]))
    
    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
    
    def save_model(self, path="model.npy"):
        np.save(path, self.q_table)
    
    def load_model(self, path="model.npy"):
        try:
            self.q_table = np.load(path)
            self.is_pretrained = True
            return True
        except:
            return False

# Global agent and environment
agent = RLAgent()
env = SimpleEnvironment()
pretraining_task: Optional[asyncio.Task] = None

app = FastAPI()

# Pre-training function
async def pretrain_agent(episodes=1000, report_interval=100):
    """Pre-train the agent before user interactions"""
    global agent
    
    print(f"Starting pre-training for {episodes} episodes...")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.select_action(state, training=True)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
            
            # Allow other tasks to run
            await asyncio.sleep(0)
        
        if (episode + 1) % report_interval == 0:
            print(f"Episode {episode + 1}/{episodes} - Reward: {total_reward:.2f}, Steps: {steps}")
    
    agent.is_pretrained = True
    agent.save_model()
    print("Pre-training complete! Model saved.")

@app.on_event("startup")
async def startup_event():
    """Run pre-training on server startup"""
    global pretraining_task
    
    # Try to load existing model
    if agent.load_model():
        print("Loaded pre-trained model from disk")
    else:
        # Start pre-training in background
        pretraining_task = asyncio.create_task(pretrain_agent(episodes=500))

@app.get("/")
async def get():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RL Agent Interaction</title>
        <style>
            body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
            #status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .ready { background: #d4edda; color: #155724; }
            .training { background: #fff3cd; color: #856404; }
            button { padding: 10px 20px; margin: 5px; font-size: 16px; cursor: pointer; }
            #log { height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin: 10px 0; }
            .log-entry { margin: 5px 0; }
        </style>
    </head>
    <body>
        <h1>RL Agent Interaction</h1>
        <div id="status" class="training">Checking status...</div>
        <div>
            <p>Current State: <strong id="state">-</strong></p>
            <p>Total Reward: <strong id="reward">0</strong></p>
        </div>
        <div>
            <button onclick="sendAction(0)">Action 0 (Left)</button>
            <button onclick="sendAction(1)">Action 1 (Stay)</button>
            <button onclick="sendAction(2)">Action 2 (Right)</button>
            <button onclick="resetEnv()">Reset Environment</button>
        </div>
        <h3>Log:</h3>
        <div id="log"></div>
        
        <script>
            const ws = new WebSocket(`ws://${window.location.host}/ws`);
            let totalReward = 0;
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                const log = document.getElementById('log');
                
                if (data.type === 'status') {
                    const statusDiv = document.getElementById('status');
                    if (data.pretrained) {
                        statusDiv.className = 'ready';
                        statusDiv.textContent = '✓ Agent is ready (pre-trained)';
                    } else {
                        statusDiv.className = 'training';
                        statusDiv.textContent = '⏳ Agent is pre-training... Please wait.';
                    }
                } else if (data.type === 'state') {
                    document.getElementById('state').textContent = data.state;
                    totalReward += data.reward;
                    document.getElementById('reward').textContent = totalReward.toFixed(2);
                    
                    const entry = document.createElement('div');
                    entry.className = 'log-entry';
                    entry.textContent = `State: ${data.state}, Reward: ${data.reward.toFixed(2)}, Done: ${data.done}`;
                    log.appendChild(entry);
                    log.scrollTop = log.scrollHeight;
                    
                    if (data.done) {
                        const doneEntry = document.createElement('div');
                        doneEntry.className = 'log-entry';
                        doneEntry.style.fontWeight = 'bold';
                        doneEntry.textContent = `Episode finished! Total Reward: ${totalReward.toFixed(2)}`;
                        log.appendChild(doneEntry);
                    }
                }
            };
            
            function sendAction(action) {
                ws.send(JSON.stringify({type: 'action', action: action}));
            }
            
            function resetEnv() {
                totalReward = 0;
                document.getElementById('reward').textContent = '0';
                ws.send(JSON.stringify({type: 'reset'}));
            }
            
            // Request initial status
            ws.onopen = function() {
                ws.send(JSON.stringify({type: 'status'}));
                ws.send(JSON.stringify({type: 'reset'}));
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        # Send initial status
        await websocket.send_json({
            "type": "status",
            "pretrained": agent.is_pretrained
        })
        
        user_env = SimpleEnvironment()
        state = user_env.reset()
        
        await websocket.send_json({
            "type": "state",
            "state": int(state),
            "reward": 0,
            "done": False
        })
        
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "status":
                await websocket.send_json({
                    "type": "status",
                    "pretrained": agent.is_pretrained
                })
            
            elif data["type"] == "reset":
                state = user_env.reset()
                await websocket.send_json({
                    "type": "state",
                    "state": int(state),
                    "reward": 0,
                    "done": False
                })
            
            elif data["type"] == "action":
                action = data["action"]
                next_state, reward, done = user_env.step(action)
                
                # Agent can still learn from user interactions if desired
                # agent.update(state, action, reward, next_state)
                
                state = next_state
                
                await websocket.send_json({
                    "type": "state",
                    "state": int(state),
                    "reward": float(reward),
                    "done": done
                })
                
                if done:
                    state = user_env.reset()
    
    except WebSocketDisconnect:
        print("Client disconnected")

@app.get("/pretrain/status")
async def pretrain_status():
    """Check pre-training status"""
    return {
        "is_pretrained": agent.is_pretrained,
        "training_in_progress": pretraining_task is not None and not pretraining_task.done()
    }

@app.post("/pretrain/start")
async def start_pretraining(episodes: int = 500):
    """Manually trigger pre-training"""
    global pretraining_task
    
    if pretraining_task and not pretraining_task.done():
        return {"error": "Pre-training already in progress"}
    
    pretraining_task = asyncio.create_task(pretrain_agent(episodes=episodes))
    return {"message": f"Pre-training started for {episodes} episodes"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)