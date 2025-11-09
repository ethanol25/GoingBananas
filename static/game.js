const ws = new WebSocket(`ws://${window.location.host}/ws`);
let training = false;
let racing = false;
let playerSteps = 0;
let aiWins = 0;
let playerWins = 0;
let isInitialState = true;

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
        // This is the "real" game state, after training or a move
        renderMaze(data.maze, data.agent_pos, data.player_pos);
        updateStats(data.stats); // Update with final race scores

        if (isInitialState) {
            isInitialState = false;
        } else {
            // This is the signal that training is over and the race is starting
            showMainContent('maze');
            racing = true;
            addLog('🏁 RACE START! Use WASD or Arrows!', true);
        }
    } else if (data.type === 'training_update') {
        // --- THIS IS THE NEW HANDLER ---
        // This shows the agent's training steps in real-time
        racing = false; // Player can't move during training
        showMainContent('maze'); // Ensure maze is visible
        renderMaze(data.maze, data.agent_pos, data.player_pos);
        updateTrainingStats(data.stats); // Use a separate stats-updater for training

    } else if (data.type === 'log') {
        addLog(data.message, data.player);
    } else if (data.type === 'win') {
        handleWin(data.winner);
    }
    // Note: The 'training_complete' handler is GONE, 
    // because the 'state' message now signals the end of training.
};

function updateTrainingStats(stats) {
    // Update visible stats for TRAINING
    document.getElementById('episode').textContent = `${stats.episode}/${stats.total_episodes}`;
    
    // Set scores to 0 during training, as they are irrelevant
    document.getElementById('player-wins').textContent = "0";
    document.getElementById('ai-wins').textContent = "0";

    // Update hidden stats
    document.getElementById('steps').textContent = stats.steps;
    document.getElementById('epsilon').textContent = stats.epsilon.toFixed(3);
}

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

// // --- Game Control Button Listeners (for hidden buttons) ---
// document.getElementById('train-button').onclick = startTraining;
// document.getElementById('stop-button').onclick = stopTraining;
// document.getElementById('race-button').onclick = startRace;
// document.getElementById('reset-button').onclick = resetEnvironment;

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

            if (hasAgent) {
                cellDiv.appendChild(createEntity('agent'));
            }
            if (hasPlayer) {
                cellDiv.appendChild(createEntity('player'));
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
    let winMessage = '';
    racing = false;

    if (winner === 'player') {
        playerWins++;
        winMessage = `🏆 PLAYER WINS!`;
        document.getElementById('player-wins').textContent = playerWins;
    } else if (winner === 'ai') {
        aiWins++;
        winMessage = `🏆 AI WINS!`;
        document.getElementById('ai-wins').textContent = aiWins;
    } else {
        // --- NEW: Handle the "draw" case ---
        winMessage = `💥 IT'S A DRAW!`;
        // We don't increment either score
    }

    addLog(winMessage);

    // Alert the user
    alert(winMessage + `\n\nPress OK to advance to the next level.`);

    // Go back to the "Ready" screen
    showMainContent('ready-content');
}

// // --- Button Functions ---
// function startTraining() {
//     // This function is now just for manual training
//     ws.send(JSON.stringify({action: 'train'}));
//     addLog('Manual training started...');
//     training = true;
//     racing = false;
// }

// function stopTraining() {
//     ws.send(JSON.stringify({action: 'stop'}));
//     addLog('Training stopped');
//     training = false;
// }

function startRace() {
    // This function is triggered by the "START RACE" button
    // It tells the server to train for the next level
    ws.send(JSON.stringify({action: 'race'}));

    // This LOGIC MOVED from 'resetEnvironment'
    // Reset local scores for the new race
    playerSteps = 0; 
    document.getElementById('player-wins').textContent = "0"; // Reset live score
    document.getElementById('ai-wins').textContent = "0"; // Reset live score

    // Show the maze and log
    addLog('Race started! Use arrow keys or WASD to move!', true);
}

function resetEnvironment() {
    // This function is for the hidden button, to go to Main Menu
    ws.send(JSON.stringify({action: 'reset'}));
    addLog('Environment reset. Going to Main Menu.');

    // Reset win counts
    playerWins = 0;
    aiWins = 0;
    document.getElementById('player-wins').textContent = "0";
    document.getElementById('ai-wins').textContent = "0";

    // Go back to the very start screen
    showMainContent('start-content');
}