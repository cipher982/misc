/**
 * Performance comparison charts using Plotly.js
 * Provides interactive charts for comparing different backends
 */

class PerformanceCharts {
    constructor() {
        this.colors = {
            'python': '#FF6B35',
            'numpy': '#F7931E', 
            'torch': '#4ECDC4',
            'default': '#95A5A6'
        };
    }
    
    /**
     * Render performance comparison chart
     */
    renderPerformanceComparison(containerId, benchmarkData) {
        const backends = Object.keys(benchmarkData);
        const metrics = ['speed', 'memory', 'accuracy'];
        
        const traces = metrics.map(metric => ({
            x: backends,
            y: backends.map(backend => benchmarkData[backend][metric] || 0),
            name: metric.charAt(0).toUpperCase() + metric.slice(1),
            type: 'bar',
            marker: {
                color: backends.map(backend => this.colors[backend] || this.colors.default),
                opacity: 0.7
            }
        }));
        
        const layout = {
            title: {
                text: 'Backend Performance Comparison',
                font: { size: 16 }
            },
            xaxis: { title: 'Backend' },
            yaxis: { title: 'Performance Score' },
            barmode: 'group',
            showlegend: true,
            margin: { l: 50, r: 50, b: 50, t: 80 }
        };
        
        Plotly.newPlot(containerId, traces, layout, {
            responsive: true,
            displayModeBar: true
        });
    }
    
    /**
     * Render speed comparison over model sizes
     */
    renderSpeedComparison(containerId, speedData) {
        const backends = Object.keys(speedData);
        const traces = backends.map(backend => ({
            x: speedData[backend].model_sizes || [],
            y: speedData[backend].execution_times || [],
            name: backend.charAt(0).toUpperCase() + backend.slice(1),
            type: 'scatter',
            mode: 'lines+markers',
            line: {
                color: this.colors[backend] || this.colors.default,
                width: 3
            },
            marker: {
                size: 8,
                color: this.colors[backend] || this.colors.default
            }
        }));
        
        const layout = {
            title: {
                text: 'Execution Time vs Model Size',
                font: { size: 16 }
            },
            xaxis: { 
                title: 'Model Parameters (K)',
                type: 'log'
            },
            yaxis: { 
                title: 'Execution Time (ms)',
                type: 'log'
            },
            showlegend: true,
            hovermode: 'closest',
            margin: { l: 60, r: 50, b: 50, t: 80 }
        };
        
        Plotly.newPlot(containerId, traces, layout, {
            responsive: true,
            displayModeBar: true
        });
    }
    
    /**
     * Render memory usage comparison
     */
    renderMemoryComparison(containerId, memoryData) {
        const backends = Object.keys(memoryData);
        
        // Create stacked bar chart for different memory types
        const traces = [
            {
                x: backends,
                y: backends.map(backend => memoryData[backend].parameters_mb || 0),
                name: 'Parameters',
                type: 'bar',
                marker: { color: '#3498DB' }
            },
            {
                x: backends,
                y: backends.map(backend => memoryData[backend].activations_mb || 0),
                name: 'Activations',
                type: 'bar',
                marker: { color: '#E74C3C' }
            },
            {
                x: backends,
                y: backends.map(backend => memoryData[backend].gradients_mb || 0),
                name: 'Gradients',
                type: 'bar',
                marker: { color: '#F39C12' }
            }
        ];
        
        const layout = {
            title: {
                text: 'Memory Usage Breakdown by Backend',
                font: { size: 16 }
            },
            xaxis: { title: 'Backend' },
            yaxis: { title: 'Memory Usage (MB)' },
            barmode: 'stack',
            showlegend: true,
            margin: { l: 50, r: 50, b: 50, t: 80 }
        };
        
        Plotly.newPlot(containerId, traces, layout, {
            responsive: true,
            displayModeBar: true
        });
    }
    
    /**
     * Render training loss comparison
     */
    renderLossComparison(containerId, lossData) {
        const backends = Object.keys(lossData);
        const traces = backends.map(backend => ({
            y: lossData[backend].loss_history || [],
            name: backend.charAt(0).toUpperCase() + backend.slice(1),
            type: 'scatter',
            mode: 'lines',
            line: {
                color: this.colors[backend] || this.colors.default,
                width: 2
            }
        }));
        
        const layout = {
            title: {
                text: 'Training Loss Comparison',
                font: { size: 16 }
            },
            xaxis: { title: 'Training Step' },
            yaxis: { title: 'Loss' },
            showlegend: true,
            hovermode: 'x unified',
            margin: { l: 50, r: 50, b: 50, t: 80 }
        };
        
        Plotly.newPlot(containerId, traces, layout, {
            responsive: true,
            displayModeBar: true
        });
    }
    
    /**
     * Render accuracy vs speed scatter plot
     */
    renderAccuracySpeedScatter(containerId, data) {
        const backends = Object.keys(data);
        const traces = [{
            x: backends.map(backend => data[backend].speed || 0),
            y: backends.map(backend => data[backend].accuracy || 0),
            mode: 'markers+text',
            type: 'scatter',
            text: backends.map(backend => backend.charAt(0).toUpperCase() + backend.slice(1)),
            textposition: 'top center',
            marker: {
                size: 15,
                color: backends.map(backend => this.colors[backend] || this.colors.default),
                line: { color: 'white', width: 2 }
            }
        }];
        
        const layout = {
            title: {
                text: 'Speed vs Accuracy Trade-off',
                font: { size: 16 }
            },
            xaxis: { 
                title: 'Speed Score (higher is better)',
                range: [0, Math.max(...backends.map(b => data[b].speed || 0)) * 1.1]
            },
            yaxis: { 
                title: 'Accuracy Score (higher is better)',
                range: [0, Math.max(...backends.map(b => data[b].accuracy || 0)) * 1.1]
            },
            showlegend: false,
            hovermode: 'closest',
            margin: { l: 60, r: 50, b: 50, t: 80 }
        };
        
        Plotly.newPlot(containerId, traces, layout, {
            responsive: true,
            displayModeBar: true
        });
    }
    
    /**
     * Render real-time training metrics
     */
    renderRealtimeMetrics(containerId, metricsData) {
        const currentTime = new Date();
        const timeWindow = 50; // Show last 50 points
        
        const traces = [
            {
                x: metricsData.timestamps || [],
                y: metricsData.loss || [],
                name: 'Loss',
                type: 'scatter',
                mode: 'lines',
                line: { color: '#E74C3C', width: 2 }
            },
            {
                x: metricsData.timestamps || [],
                y: metricsData.learning_rate || [],
                name: 'Learning Rate',
                type: 'scatter',
                mode: 'lines',
                yaxis: 'y2',
                line: { color: '#3498DB', width: 2 }
            }
        ];
        
        const layout = {
            title: {
                text: 'Real-time Training Metrics',
                font: { size: 16 }
            },
            xaxis: { 
                title: 'Time',
                type: 'date'
            },
            yaxis: { 
                title: 'Loss',
                side: 'left'
            },
            yaxis2: {
                title: 'Learning Rate',
                side: 'right',
                overlaying: 'y',
                type: 'log'
            },
            showlegend: true,
            margin: { l: 60, r: 60, b: 50, t: 80 }
        };
        
        Plotly.newPlot(containerId, traces, layout, {
            responsive: true,
            displayModeBar: true
        });
    }
    
    /**
     * Update existing chart with new data
     */
    updateChart(containerId, newData, traceIndex = 0) {
        Plotly.extendTraces(containerId, newData, [traceIndex]);
        
        // Keep only recent data points for performance
        const currentData = document.getElementById(containerId).data;
        if (currentData[traceIndex].x.length > 100) {
            Plotly.relayout(containerId, {
                'xaxis.range': [
                    currentData[traceIndex].x[currentData[traceIndex].x.length - 50],
                    currentData[traceIndex].x[currentData[traceIndex].x.length - 1]
                ]
            });
        }
    }
}

// Export for use in Streamlit
window.PerformanceCharts = PerformanceCharts;