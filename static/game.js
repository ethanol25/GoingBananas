const ws = new WebSocket(`ws://${window.location.host}/ws`);
let training = false;
let racing = false;
let playerSteps = 0;
let aiWins = 0;
let playerWins = 0;

// --- Get DOM Elements ---
const mainContentBlocks = document.querySelectorAll('.main-content-block');
const raceButton = document.getElementById('race-button');

// --- Content Swapping ---
function showMainContent(contentId) {
    // Hide all content blocks
    mainContentBlocks.forEach(block => {
        block.style.display = 'none';
    });
    
    // Show the target block
    const blockToShow = document.getElementById(contentId);
    if (blockToShow) {
        // Use 'flex' or 'grid' based on the block
        if (contentId === 'maze') {
            blockToShow.style.display = 'grid';
        } else {
            blockToShow.style.display = 'flex';
        }
    }
}

// --- WebSocket Handlers ---
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    if (data.type === 'state') {
        renderMaze(data.maze, data.agent_pos, data.player_pos);
        updateStats(data.stats);
    } else if (data.type === 'log') {
        addLog(data.message, data.player);
    } else if (data.type === 'win') {
        handleWin(data.winner);
    } else if (data.type === 'training_complete') {
        // --- THIS IS THE NEW LOGIC ---
        addLog('🤖 AI training complete! Ready to race!');
        // Show the "START RACE" button
        showMainContent('ready-content'); 
    }
};

// --- Page Load & Button Events ---
window.onload = () => {
    // Show the start menu first
    showMainContent('start-content');
};

document.getElementById('start-button').onclick = () => {
    showMainContent('lore-content');
};

document.getElementById('save-queen-button').onclick = () => {
    // Show the maze
    showMainContent('maze');
    
    // Tell server to start auto-training
    addLog('🤖 AI is learning the forest... Please wait.');
    ws.send(JSON.stringify({ action: 'start_auto_train' }));
};

document.getElementById('start-race-button').onclick = () => {
    // Show the maze
    showMainContent('maze');
    // Start the race
    startRace();
};

// --- Game Control Button Listeners (for hidden buttons) ---
document.getElementById('train-button').onclick = startTraining;
document.getElementById('stop-button').onclick = stopTraining;
document.getElementById('race-button').onclick = startRace;
document.getElementById('reset-button').onclick = resetEnvironment;

function createEntity(type) {
    const img = document.createElement('img');
    img.src = `/static/assets/low_res/sprite-${type}.png`;
    img.className = type; // e.g., class="agent"
    return img;
}

// --- Maze & Game Logic ---
function renderMaze(maze, agentPos, playerPos) {
    const mazeDiv = document.getElementById('maze');
    if (!mazeDiv) return;
    
    mazeDiv.style.gridTemplateColumns = `repeat(${maze.length}, 1fr)`;
    mazeDiv.innerHTML = ''; // Clear the old maze
    
    // Define our tile classes
    const entityMap = {
        1: 'wall',
        2: 'start',
        3: 'goal',
        4: 'banana'
    };

    maze.forEach((row, i) => {
        row.forEach((cell, j) => {
            const cellDiv = document.createElement('div');
            
            // 1. Set the floor class (e.g., "cell path")
            cellDiv.className = 'cell';
            
            // 2. Add entity sprites
            
            const entityType = entityMap[cell];
            if (entityType) {
                cellDiv.appendChild(createEntity(entityType));
            }

            // Check for player and agent
            const hasPlayer = playerPos[0] === i && playerPos[1] === j;
            const hasAgent = agentPos[0] === i && agentPos[1] === j;

            if (hasPlayer) {
                cellDiv.appendChild(createEntity('player'));
            }
            if (hasAgent) {
                cellDiv.appendChild(createEntity('agent'));
            }
            
            mazeDiv.appendChild(cellDiv);
        });
    });
}

function updateStats(stats) {
    // Update visible stats
    document.getElementById('episode').textContent = stats.episode;
    
    // This is new: update scores from the 'stats' object
    // (We'll add this in the next step)
    if (stats.player_score !== undefined) {
        document.getElementById('player-wins').textContent = stats.player_score.toFixed(0);
    }
    if (stats.agent_score !== undefined) {
        document.getElementById('ai-wins').textContent = stats.agent_score.toFixed(0);
    }

    // Update hidden stats
    document.getElementById('steps').textContent = stats.steps;
    document.getElementById('epsilon').textContent = stats.epsilon.toFixed(3);
}

// Keyboard controls for player
document.addEventListener('keydown', function(e) {
    // Only allow movement if the race is on AND the maze is visible
    if (!racing || document.getElementById('maze').style.display === 'none') return;
    
    let action = null;
    if (e.key === 'ArrowUp' || e.key === 'w' || e.key === 'W') action = 0;
    else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') action = 1;
    else if (e.key === 'ArrowDown' || e.key === 's' || e.key === 'S') action = 2;
    else if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') action = 3;
    
    if (action !== null) {
        e.preventDefault();
        playerSteps++; 
        ws.send(JSON.stringify({action: 'player_move', direction: action}));
    }
});


function addLog(message, isPlayer = false) {
    const logDiv = document.getElementById('log');
    if (!logDiv) return; // Failsafe
    const entry = document.createElement('div');
    entry.className = 'log-entry' + (isPlayer ? ' log-player' : '');
    const icon = isPlayer ? '👤' : '🤖';
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${icon} ${message}`;
    logDiv.appendChild(entry);
    logDiv.scrollTop = logDiv.scrollHeight;
}

function handleWin(winner) {
    racing = false;
    let winMessage = '';
    
    if (winner === 'player') {
        playerWins++;
        winMessage = `🏆 PLAYER WINS in ${playerSteps} steps!`;
        document.getElementById('player-wins').textContent = playerWins;
    } else {
        aiWins++;
        winMessage = `🏆 AI WINS!`;
        document.getElementById('ai-wins').textContent = aiWins;
    }
    
    addLog(winMessage);
    
    // Alert the user
    alert(winMessage + "\n\nPress OK to race again.");
    
    // --- MODIFIED ---
    // Reset the environment and show the "Ready" screen, not the start menu
    ws.send(JSON.stringify({action: 'reset'}));
    racing = false;
    training = false;
    playerSteps = 0;
    showMainContent('ready-content');
}

// --- Button Functions ---
function startTraining() {
    // This function is now just for manual training
    ws.send(JSON.stringify({action: 'train'}));
    addLog('Manual training started...');
    training = true;
    racing = false;
}

function stopTraining() {
    ws.send(JSON.stringify({action: 'stop'}));
    addLog('Training stopped');
    training = false;
}

function startRace() {
    // This is now the ONLY way to start a race
    ws.send(JSON.stringify({action: 'race'}));
    addLog('Race started! Use arrow keys or WASD to move!', true);
    racing = true;
    training = false;
    playerSteps = 0; // Reset local player step count
}

function resetEnvironment() {
    ws.send(JSON.stringify({action: 'reset'}));
    addLog('Environment reset. Going to Main Menu.');
    racing = false;
    training = false;
    playerSteps = 0;
    aiWins = 0; 
    playerWins = 0; 
    document.getElementById('ai-wins').textContent = aiWins;
    document.getElementById('player-wins').textContent = playerWins;
    
    // Go back to the very start screen
    showMainContent('start-content');
}