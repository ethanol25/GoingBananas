# 🍌 Monkey See, Monkey Do!
An Evil Monkey has stolen your bananas, but most importantly, your beautiful banana queen. He is learning the forest as we speak! You must save her, and collect as many bananas as possible! Race against Evil Monkey powered by Q-learning. Built with FastAPI, WebSockets, and Gymnasium, this game has you race against a reinforcement learning agent!

## 🐵 Evil Monkey (AI Agent)

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
| 🍌 Banana       | +50    | Collectible bonus items    |
| 🏆 Goal          | +100   | Reach the destination      |
| 🏃 Speed Bonus   | +20    | Complete in under 50 steps |
| 🧱 Wall Hit      | -5     | Collision penalty          |
| 📍 Exploration   | +1     | Visit new cells            |
| 📏 Distance      | ±0.5   | Move closer/away from goal |
| ⏱️ Time Pressure | -0.5   | After 100 steps            |

## 🚀 Installation
pip install fastapi uvicorn gymnasium numpy websockets

## 💻 Usage
* run python maze_user.py
* The server will start on http://localhost:8000

## 🐒 Train the Evil Monkey
* Watch the Evil Monkey navigate through randomly generated mazes
* Race Against the Evil Monkey after it trains
* Use arrow keys/WASD to move your sprite 
* Try to reach the goal before the Evil Monkey
* First to reach the goal wins!

Happy Racing! 🏁
