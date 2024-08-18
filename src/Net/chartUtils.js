// chartUtils.js

function createSingleLineChart(ctx, data, label, yAxisConfig) {
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(d => d.move),
            datasets: [{
                label: label,
                data: data.map(d => d.value),
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: {
                        display: true,
                        text: 'Move Number'
                    }
                },
                y: yAxisConfig
            },
            plugins: {
                title: {
                    display: true,
                    text: label
                }
            }
        }
    });
}

function createMultiLineChart(ctx, dataArrays, labels, title, yAxisConfig) {
    const datasets = dataArrays.map((data, index) => ({
        label: labels[index],
        data: data.map(d => d.value),
        borderColor: `hsl(${index * 137.5}, 70%, 50%)`,
        tension: 0.1
    }));

    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: dataArrays[0].map(d => d.move),
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: {
                        display: true,
                        text: 'Move Number'
                    }
                },
                y: yAxisConfig
            },
            plugins: {
                title: {
                    display: true,
                    text: title
                }
            }
        }
    });
}