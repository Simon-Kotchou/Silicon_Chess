// Load the JSON data
d3.json('chess_analysis_data.json').then(data => {
    const chartContainer = d3.select('#charts');
    
    // Create charts for each game
    Object.entries(data).forEach(([gameName, gameData]) => {
        createCharts(gameName, gameData, chartContainer);
    });
}).catch(error => {
    console.error('Error loading data:', error);
});

function createCharts(gameName, gameData, container) {
    const gameContainer = container.append('div')
        .attr('class', 'game-container')
        .attr('id', `${gameName}-container`);
    
    gameContainer.append('h2').text(gameName);
    
    createLineChart(gameContainer, gameData.stockfish_eval, 'Stockfish Evaluation', 'stockfish-eval', [-5, 5]);
    createLineChart(gameContainer, gameData.control, 'Control', 'control', [-1, 1]);
    createMultiLineChart(gameContainer, 
        [gameData.total_influence, gameData.tension], 
        ['Total Influence', 'Tension'], 
        'Influence and Tension', 
        'influence-tension'
    );
    createMultiLineChart(gameContainer, 
        [gameData.white_mobility, gameData.black_mobility], 
        ['White Mobility', 'Black Mobility'], 
        'Piece Mobility', 
        'mobility'
    );
    createLineChart(gameContainer, gameData.js_divergence, 'Jensen-Shannon Divergence', 'js-divergence', [0, 1]);
}

function createLineChart(container, data, title, id, yDomain) {
    const margin = {top: 20, right: 20, bottom: 30, left: 50};
    const width = 600 - margin.left - margin.right;
    const height = 300 - margin.top - margin.bottom;
    
    const svg = container.append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);
    
    const x = d3.scaleLinear().range([0, width]);
    const y = d3.scaleLinear().range([height, 0]);
    
    const line = d3.line()
        .x(d => x(d.move))
        .y(d => y(d.value));
    
    x.domain(d3.extent(data, d => d.move));
    y.domain(yDomain || d3.extent(data, d => d.value));
    
    svg.append('path')
        .datum(data)
        .attr('fill', 'none')
        .attr('stroke', 'steelblue')
        .attr('stroke-width', 1.5)
        .attr('d', line);
    
    svg.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(x).ticks(10));
    
    svg.append('g')
        .call(d3.axisLeft(y));
    
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', 0 - margin.top / 2)
        .attr('text-anchor', 'middle')
        .style('font-size', '16px')
        .text(title);
}

function createMultiLineChart(container, dataArrays, labels, title, id) {
    const margin = {top: 20, right: 20, bottom: 30, left: 50};
    const width = 600 - margin.left - margin.right;
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
        .x(d => x(d.move))
        .y(d => y(d.value));
    
    x.domain(d3.extent(dataArrays[0], d => d.move));
    y.domain([
        d3.min(dataArrays, arr => d3.min(arr, d => d.value)),
        d3.max(dataArrays, arr => d3.max(arr, d => d.value))
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
        .call(d3.axisBottom(x).ticks(10));
    
    svg.append('g')
        .call(d3.axisLeft(y));
    
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', 0 - margin.top / 2)
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