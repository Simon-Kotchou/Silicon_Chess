// Load the JSON data
d3.json('chess_analysis_data.json').then(data => {
    const chartContainer = d3.select('#charts');
    
    // Create charts for each game
    Object.entries(data).forEach(([gameName, gameData]) => {
        createCharts(gameName, gameData, chartContainer);
    });
});

function createCharts(gameName, gameData, container) {
    const gameContainer = container.append('div')
        .attr('class', 'game-container')
        .attr('id', `${gameName}-container`);
    
    gameContainer.append('h2').text(gameName);
    
    // Stockfish Evaluation Chart
    createLineChart(gameContainer, gameData.stockfish_eval, 'Stockfish Evaluation', 'stockfish-eval');
    
    // Control Chart
    createLineChart(gameContainer, gameData.control, 'Control', 'control');
    
    // Influence and Tension Chart
    createMultiLineChart(gameContainer, 
        [gameData.total_influence, gameData.tension], 
        ['Total Influence', 'Tension'], 
        'Influence and Tension', 
        'influence-tension'
    );
    
    // Mobility Chart
    createMultiLineChart(gameContainer, 
        [gameData.white_mobility, gameData.black_mobility], 
        ['White Mobility', 'Black Mobility'], 
        'Piece Mobility', 
        'mobility'
    );
    
    // Jensen-Shannon Divergence Chart
    createLineChart(gameContainer, gameData.js_divergence, 'Jensen-Shannon Divergence', 'js-divergence');
    
    // Board State Visualizations
    createBoardVisualizations(gameContainer, gameData, gameName);
}

function createLineChart(container, data, title, id) {
    const margin = {top: 20, right: 20, bottom: 30, left: 50};
    const width = 500 - margin.left - margin.right;
    const height = 300 - margin.top - margin.bottom;
    
    const svg = container.append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);
    
    const x = d3.scaleLinear().range([0, width]);
    const y = d3.scaleLinear().range([height, 0]);
    
    const line = d3.line()
        .x((d, i) => x(i))
        .y(d => y(d));
    
    x.domain([0, data.length - 1]);
    y.domain(d3.extent(data));
    
    svg.append('path')
        .datum(data)
        .attr('fill', 'none')
        .attr('stroke', 'steelblue')
        .attr('stroke-width', 1.5)
        .attr('d', line);
    
    svg.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(x));
    
    svg.append('g')
        .call(d3.axisLeft(y));
    
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', 0)
        .attr('text-anchor', 'middle')
        .style('font-size', '16px')
        .text(title);
}

function createMultiLineChart(container, dataArrays, labels, title, id) {
    const margin = {top: 20, right: 20, bottom: 30, left: 50};
    const width = 500 - margin.left - margin.right;
    const height = 300 - margin.top - margin.bottom;
    
    const svg = container.append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);
    
    const x = d3.scaleLinear().range([0, width]);
    const y = d3.scaleLinear().range([height, 0]);
    
    const color = d3.scaleOrdinal(d3.schemeCategory10);
    
    const line = d3.line()
        .x((d, i) => x(i))
        .y(d => y(d));
    
    x.domain([0, d3.max(dataArrays, arr => arr.length) - 1]);
    y.domain([
        d3.min(dataArrays, arr => d3.min(arr)),
        d3.max(dataArrays, arr => d3.max(arr))
    ]);
    
    dataArrays.forEach((data, index) => {
        svg.append('path')
            .datum(data)
            .attr('fill', 'none')
            .attr('stroke', color(index))
            .attr('stroke-width', 1.5)
            .attr('d', line);
    });
    
    svg.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(x));
    
    svg.append('g')
        .call(d3.axisLeft(y));
    
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', 0)
        .attr('text-anchor', 'middle')
        .style('font-size', '16px')
        .text(title);
    
    const legend = svg.selectAll('.legend')
        .data(labels)
        .enter().append('g')
        .attr('class', 'legend')
        .attr('transform', (d, i) => `translate(0,${i * 20})`);
    
    legend.append('rect')
        .attr('x', width - 18)
        .attr('width', 18)
        .attr('height', 18)
        .style('fill', (d, i) => color(i));
    
    legend.append('text')
        .attr('x', width - 24)
        .attr('y', 9)
        .attr('dy', '.35em')
        .style('text-anchor', 'end')
        .text(d => d);
}

function createBoardVisualizations(container, gameData, gameName) {
    const numMoves = gameData.white_influence.length;
    const keyPositions = [0, Math.floor(numMoves / 3), Math.floor(2 * numMoves / 3), numMoves - 1];
    
    keyPositions.forEach((position, index) => {
        const boardContainer = container.append('div')
            .attr('class', 'board-container')
            .attr('id', `${gameName}-board-${index}`);
        
        boardContainer.append('h3').text(`${gameName} - Position ${position + 1}`);
        
        createInfluenceMap(boardContainer, gameData.white_influence[position], gameData.black_influence[position]);
        createJEPAMask(boardContainer, gameData.jepa_mask[position]);
    });
}

function createInfluenceMap(container, whiteInfluence, blackInfluence) {
    const size = 400;
    const squareSize = size / 8;
    
    const svg = container.append('svg')
        .attr('width', size)
        .attr('height', size);
    
    const colorScale = d3.scaleLinear()
        .domain([-1, 0, 1])
        .range(['blue', 'white', 'red']);
    
    for (let i = 0; i < 8; i++) {
        for (let j = 0; j < 8; j++) {
            const influence = whiteInfluence[i][j] - blackInfluence[i][j];
            svg.append('rect')
                .attr('x', j * squareSize)
                .attr('y', i * squareSize)
                .attr('width', squareSize)
                .attr('height', squareSize)
                .attr('fill', colorScale(influence));
        }
    }
}

function createJEPAMask(container, jepaMask) {
    const size = 400;
    const squareSize = size / 8;
    
    const svg = container.append('svg')
        .attr('width', size)
        .attr('height', size);
    
    for (let i = 0; i < 8; i++) {
        for (let j = 0; j < 8; j++) {
            svg.append('rect')
                .attr('x', j * squareSize)
                .attr('y', i * squareSize)
                .attr('width', squareSize)
                .attr('height', squareSize)
                .attr('fill', jepaMask[i][j] ? 'rgba(0, 0, 0, 0.5)' : 'rgba(255, 255, 255, 0.5)');
        }
    }
}

// Call the main function to create all visualizations
createVisualizations();