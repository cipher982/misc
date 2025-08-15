/**
 * Interactive network architecture visualization
 * Shows transformer structure with data flow and layer details
 */

class NetworkArchitectureViz {
    constructor(containerId) {
        this.container = d3.select(`#${containerId}`);
        this.width = 1000;
        this.height = 800;
        this.margin = { top: 50, right: 50, bottom: 50, left: 50 };
        
        this.svg = this.container
            .append("svg")
            .attr("width", this.width)
            .attr("height", this.height);
            
        this.g = this.svg
            .append("g")
            .attr("transform", `translate(${this.margin.left},${this.margin.top})`);
        
        // Define layer colors
        this.layerColors = {
            'embedding': '#FF6B35',
            'positional': '#F7931E',
            'attention': '#4ECDC4',
            'feedforward': '#45B7D1',
            'normalization': '#96CEB4',
            'output': '#FECA57'
        };
    }
    
    /**
     * Render complete transformer architecture
     */
    renderArchitecture(config) {
        this.g.selectAll("*").remove();
        
        const layers = this.createLayerStructure(config);
        const layerHeight = 80;
        const layerSpacing = 100;
        const startY = 50;
        
        // Create layer groups
        const layerGroups = this.g.selectAll(".layer-group")
            .data(layers)
            .enter()
            .append("g")
            .attr("class", "layer-group")
            .attr("transform", (d, i) => `translate(0, ${startY + i * layerSpacing})`);
        
        // Draw layer rectangles
        layerGroups.append("rect")
            .attr("class", "layer-rect")
            .attr("x", d => d.x)
            .attr("y", 0)
            .attr("width", d => d.width)
            .attr("height", layerHeight)
            .attr("fill", d => this.layerColors[d.type] || '#95A5A6')
            .attr("stroke", "#34495E")
            .attr("stroke-width", 2)
            .attr("rx", 10)
            .attr("opacity", 0.8);
        
        // Add layer labels
        layerGroups.append("text")
            .attr("class", "layer-label")
            .attr("x", d => d.x + d.width / 2)
            .attr("y", layerHeight / 2)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "middle")
            .attr("font-size", "14px")
            .attr("font-weight", "bold")
            .attr("fill", "white")
            .text(d => d.name);
        
        // Add parameter counts
        layerGroups.append("text")
            .attr("class", "param-count")
            .attr("x", d => d.x + d.width / 2)
            .attr("y", layerHeight / 2 + 20)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "middle")
            .attr("font-size", "10px")
            .attr("fill", "white")
            .text(d => d.params ? `${d.params} params` : '');
        
        // Draw connections between layers
        this.drawConnections(layers, layerHeight, layerSpacing, startY);
        
        // Add data flow animation
        this.addDataFlowAnimation(layers, layerHeight, layerSpacing, startY);
        
        // Add interactive details
        this.addInteractiveDetails(layerGroups, config);
    }
    
    createLayerStructure(config) {
        const layers = [];
        const centerX = (this.width - this.margin.left - this.margin.right) / 2;
        const layerWidth = 200;
        
        // Input embedding layer
        layers.push({
            name: "Token Embedding",
            type: "embedding",
            x: centerX - layerWidth / 2,
            width: layerWidth,
            params: `${config.vocab_size * config.hidden_dim / 1000}K`,
            details: {
                vocab_size: config.vocab_size,
                hidden_dim: config.hidden_dim,
                description: "Converts tokens to dense vectors"
            }
        });
        
        // Positional encoding
        layers.push({
            name: "Positional Encoding",
            type: "positional",
            x: centerX - layerWidth / 2,
            width: layerWidth,
            params: config.pos_encoding_type === 'Learnable' ? 
                `${config.max_seq_len * config.hidden_dim / 1000}K` : '0',
            details: {
                type: config.pos_encoding_type,
                max_seq_len: config.max_seq_len,
                description: "Adds position information to embeddings"
            }
        });
        
        // Transformer blocks
        for (let i = 0; i < config.num_layers; i++) {
            // Multi-head attention
            layers.push({
                name: `Multi-Head Attention ${i + 1}`,
                type: "attention",
                x: centerX - layerWidth / 2,
                width: layerWidth,
                params: `${4 * config.hidden_dim * config.hidden_dim / 1000}K`,
                details: {
                    num_heads: config.num_heads,
                    head_dim: config.hidden_dim / config.num_heads,
                    description: "Self-attention mechanism"
                }
            });
            
            // Layer normalization after attention
            if (config.norm_type !== 'None') {
                layers.push({
                    name: `${config.norm_type} ${i + 1}a`,
                    type: "normalization",
                    x: centerX + layerWidth / 2 + 20,
                    width: 100,
                    params: `${2 * config.hidden_dim}`,
                    details: {
                        type: config.norm_type,
                        description: "Normalizes layer inputs"
                    }
                });
            }
            
            // Feed-forward network
            layers.push({
                name: `Feed Forward ${i + 1}`,
                type: "feedforward",
                x: centerX - layerWidth / 2,
                width: layerWidth,
                params: `${2 * config.hidden_dim * config.ff_dim / 1000}K`,
                details: {
                    ff_dim: config.ff_dim,
                    activation: config.activation_type,
                    description: "Position-wise feed-forward network"
                }
            });
            
            // Layer normalization after feed-forward
            if (config.norm_type !== 'None') {
                layers.push({
                    name: `${config.norm_type} ${i + 1}b`,
                    type: "normalization",
                    x: centerX + layerWidth / 2 + 20,
                    width: 100,
                    params: `${2 * config.hidden_dim}`,
                    details: {
                        type: config.norm_type,
                        description: "Normalizes layer inputs"
                    }
                });
            }
        }
        
        // Final layer norm
        if (config.residual_type === 'Pre-LN' && config.norm_type !== 'None') {
            layers.push({
                name: `Final ${config.norm_type}`,
                type: "normalization",
                x: centerX - layerWidth / 2,
                width: layerWidth,
                params: `${2 * config.hidden_dim}`,
                details: {
                    type: config.norm_type,
                    description: "Final normalization before output"
                }
            });
        }
        
        // Output projection
        layers.push({
            name: "Output Projection",
            type: "output",
            x: centerX - layerWidth / 2,
            width: layerWidth,
            params: `${config.hidden_dim * config.vocab_size / 1000}K`,
            details: {
                vocab_size: config.vocab_size,
                description: "Projects to vocabulary space"
            }
        });
        
        return layers;
    }
    
    drawConnections(layers, layerHeight, layerSpacing, startY) {
        const connections = [];
        
        for (let i = 0; i < layers.length - 1; i++) {
            const source = layers[i];
            const target = layers[i + 1];
            
            connections.push({
                x1: source.x + source.width / 2,
                y1: startY + i * layerSpacing + layerHeight,
                x2: target.x + target.width / 2,
                y2: startY + (i + 1) * layerSpacing,
                type: 'forward'
            });
        }
        
        // Draw connection lines
        this.g.selectAll(".connection")
            .data(connections)
            .enter()
            .append("line")
            .attr("class", "connection")
            .attr("x1", d => d.x1)
            .attr("y1", d => d.y1)
            .attr("x2", d => d.x2)
            .attr("y2", d => d.y2)
            .attr("stroke", "#7F8C8D")
            .attr("stroke-width", 2)
            .attr("marker-end", "url(#arrowhead)");
        
        // Add arrowhead marker
        this.svg.append("defs")
            .append("marker")
            .attr("id", "arrowhead")
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 8)
            .attr("refY", 0)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M0,-5L10,0L0,5")
            .attr("fill", "#7F8C8D");
    }
    
    addDataFlowAnimation(layers, layerHeight, layerSpacing, startY) {
        const animationButton = this.container
            .append("div")
            .style("position", "absolute")
            .style("top", "10px")
            .style("right", "10px")
            .append("button")
            .text("Animate Data Flow")
            .style("padding", "10px")
            .style("background-color", "#3498DB")
            .style("color", "white")
            .style("border", "none")
            .style("border-radius", "5px")
            .style("cursor", "pointer");
        
        animationButton.on("click", () => {
            // Remove existing animation dots
            this.g.selectAll(".data-dot").remove();
            
            // Create animation dots
            for (let i = 0; i < layers.length - 1; i++) {
                const source = layers[i];
                const target = layers[i + 1];
                
                this.g.append("circle")
                    .attr("class", "data-dot")
                    .attr("cx", source.x + source.width / 2)
                    .attr("cy", startY + i * layerSpacing + layerHeight)
                    .attr("r", 5)
                    .attr("fill", "#E74C3C")
                    .transition()
                    .duration(1000)
                    .delay(i * 200)
                    .attr("cx", target.x + target.width / 2)
                    .attr("cy", startY + (i + 1) * layerSpacing)
                    .on("end", function() {
                        if (i === layers.length - 2) {
                            // Animation complete, remove all dots
                            setTimeout(() => {
                                d3.selectAll(".data-dot").remove();
                            }, 500);
                        }
                    });
            }
        });
    }
    
    addInteractiveDetails(layerGroups, config) {
        const detailsPanel = this.container
            .append("div")
            .attr("class", "details-panel")
            .style("position", "absolute")
            .style("top", "80px")
            .style("right", "20px")
            .style("width", "300px")
            .style("background", "white")
            .style("border", "1px solid #ccc")
            .style("border-radius", "5px")
            .style("padding", "15px")
            .style("box-shadow", "0 2px 10px rgba(0,0,0,0.1)")
            .style("font-family", "Arial, sans-serif")
            .style("display", "none");
        
        layerGroups
            .on("mouseover", function(event, d) {
                d3.select(this).select(".layer-rect")
                    .transition()
                    .duration(200)
                    .attr("opacity", 1)
                    .attr("stroke-width", 3);
                
                // Show details
                detailsPanel.style("display", "block");
                detailsPanel.html(`
                    <h3>${d.name}</h3>
                    <p><strong>Type:</strong> ${d.type}</p>
                    <p><strong>Parameters:</strong> ${d.params || 'N/A'}</p>
                    <p><strong>Description:</strong> ${d.details?.description || 'No description'}</p>
                    ${d.details ? Object.keys(d.details)
                        .filter(key => key !== 'description')
                        .map(key => `<p><strong>${key}:</strong> ${d.details[key]}</p>`)
                        .join('') : ''}
                `);
            })
            .on("mouseout", function(event, d) {
                d3.select(this).select(".layer-rect")
                    .transition()
                    .duration(200)
                    .attr("opacity", 0.8)
                    .attr("stroke-width", 2);
                
                detailsPanel.style("display", "none");
            });
    }
    
    /**
     * Highlight specific computation path
     */
    highlightPath(pathType) {
        this.g.selectAll(".layer-rect")
            .transition()
            .duration(300)
            .attr("opacity", d => {
                if (pathType === 'attention' && d.type === 'attention') return 1;
                if (pathType === 'feedforward' && d.type === 'feedforward') return 1;
                if (pathType === 'normalization' && d.type === 'normalization') return 1;
                return 0.3;
            });
    }
    
    /**
     * Reset highlighting
     */
    resetHighlight() {
        this.g.selectAll(".layer-rect")
            .transition()
            .duration(300)
            .attr("opacity", 0.8);
    }
}

// Export for use in Streamlit
window.NetworkArchitectureViz = NetworkArchitectureViz;