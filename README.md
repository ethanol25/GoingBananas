# 🍌 Are you going Bananas?
An Evil Monkey has stolen your bananas, but most importantly, your beautiful banana queen. He is learning the forest as we speak! You must save her, and collect as many bananas as possible! Race against Evil Monkey powered by Q-learning. Built with FastAPI, WebSockets, and Gymnasium, this game has you race against a reinforcement learning agent!

## 🤖 Evil Monkey (AI Agent)

**Q-Learning Algorithm:** Implements tabular Q-learning with epsilon-greedy exploration
**Real-time Training Visualization:** Watch the Evil Monkey learn optimal paths through the maze
**Dynamic Learning:** Epsilon decay strategy for exploration-exploitation balance

## 🎯 Interactive Gameplay

**Human vs Evil Monkey:** Compete against the toughened Evil Monkey
**Keyboard Controls:** Use arrow keys (↑→↓←) or WASD to navigate
**Real-time Feedback:** See both sprites move simultaneously via WebSockets
**Win Tracking:** Keeps score of Evil Monkey wins vs Player wins

## 🎁 Reward System

| Element          | Reward | Description                |
| ---------------- | ------ | -------------------------- |
| ⭐ Power-up       | +15    | Collectible bonus items    |
| 🏆 Goal          | +100   | Reach the destination      |
| 🏃 Speed Bonus   | +20    | Complete in under 50 steps |
| 🧱 Wall Hit      | -5     | Collision penalty          |
| 📍 Exploration   | +1     | Visit new cells            |
| 📏 Distance      | ±0.5   | Move closer/away from goal |
| ⏱️ Time Pressure | -0.5   | After 100 steps            |

## 🚀 Installation

Install Dependencies
pip install fastapi uvicorn gymnasium numpy websockets

## 💻 Usage

Start the Server
python something.py
The server will start on http://localhost:8000

Train the AI
* Click "🤖 Start AI Training" to begin Q-learning
* Watch the AI navigate through randomly generated mazes
* Monitor statistics: Episode count, Steps, Epsilon value
* Click "⏸ Stop Training" when satisfied with performance
* Race Against the AI
* After training, click "🏁 Start Race!"
* Use arrow keys or WASD to move your sprite (👤)
* Try to reach the goal (🔴) before the AI (🤖)
* First to reach the goal wins!

## 📊 Statistics Tracked

**Episode:** Current training iteration
**Steps:** Moves taken in current episode/race
**Epsilon:** Current exploration rate (training only)
**Player Steps:** Human player move count
**AI Wins:** Number of AI victories
**Player Wins:** Number of human victories

Happy Racing! 🏁
