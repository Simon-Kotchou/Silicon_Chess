// main.js

document.addEventListener('DOMContentLoaded', function() {
    loadChessData();
});

function loadChessData() {
    console.log('Attempting to fetch chess_analysis_data.json');
    fetch('chess_analysis_data.json')
        .then(response => {
            console.log('Response status:', response.status);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Data successfully loaded:', data);
            const container = document.getElementById('analysis-container');
            Object.entries(data).forEach(([gameName, gameData]) => {
                createGameAnalysis(container, gameName, gameData);
            });
        })
        .catch(error => {
            console.error('Error loading chess data:', error);
            document.getElementById('analysis-container').innerHTML = `<p>Error loading data: ${error.message}</p>`;
        });
}

function createGameAnalysis(container, gameName, gameData) {
    const gameContainer = document.createElement('div');
    gameContainer.className = 'game-container';
    gameContainer.innerHTML = `<h2>${gameName}</h2>`;
    container.appendChild(gameContainer);

    const charts = [
        { data: gameData.stockfish_eval, label: 'Stockfish Evaluation', yAxis: { title: { display: true, text: 'Evaluation' } } },
        { data: gameData.control, label: 'Control', yAxis: { title: { display: true, text: 'Control' }, min: -1, max: 1 } },
        { data: [gameData.total_influence, gameData.tension], labels: ['Total Influence', 'Tension'], title: 'Influence and Tension', yAxis: { title: { display: true, text: 'Value' } } },
        { data: [gameData.white_mobility, gameData.black_mobility], labels: ['White Mobility', 'Black Mobility'], title: 'Piece Mobility', yAxis: { title: { display: true, text: 'Mobility' } } },
        { data: gameData.js_divergence, label: 'Jensen-Shannon Divergence', yAxis: { title: { display: true, text: 'Divergence' }, min: 0, max: 1 } }
    ];

    charts.forEach(chart => {
        if (!chart.data || (Array.isArray(chart.data) && chart.data.some(d => !d))) {
            console.warn(`Missing data for ${chart.label || chart.title}`);
            return;
        }

        const chartContainer = document.createElement('div');
        chartContainer.className = 'chart-container';
        gameContainer.appendChild(chartContainer);

        const canvas = document.createElement('canvas');
        chartContainer.appendChild(canvas);

        try {
            if (Array.isArray(chart.data) && !chart.data[0].hasOwnProperty('move')) {
                createMultiLineChart(canvas, chart.data, chart.labels, chart.title, chart.yAxis);
            } else {
                createSingleLineChart(canvas, chart.data, chart.label, chart.yAxis);
            }
        } catch (error) {
            console.error(`Error creating chart for ${chart.label || chart.title}:`, error);
            chartContainer.innerHTML += `<p>Error creating chart: ${error.message}</p>`;
        }
    });
}