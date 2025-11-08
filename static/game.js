// static/game.js
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
            } else if (cell === 4) { // <-- Added logic for banana
                cellDiv.className += ' banana';
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