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
                cellDiv.className += ' banana'; // <-- Updated class
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