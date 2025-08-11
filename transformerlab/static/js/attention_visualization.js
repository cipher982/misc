/**
 * Advanced attention visualization using D3.js
 * Provides interactive attention heatmaps and flow diagrams
 */

class AttentionVisualizer {
    constructor(containerId) {
        this.container = d3.select(`#${containerId}`);
        this.margin = { top: 50, right: 50, bottom: 50, left: 50 };
        this.width = 800 - this.margin.left - this.margin.right;
        this.height = 600 - this.margin.top - this.margin.bottom;
        
        this.svg = this.container
            .append("svg")
            .attr("width", this.width + this.margin.left + this.margin.right)
            .attr("height", this.height + this.margin.top + this.margin.bottom);
            
        this.g = this.svg
            .append("g")
            .attr("transform", `translate(${this.margin.left},${this.margin.top})`);
    }
    
    /**
     * Render attention heatmap with interactive features
     */
    renderHeatmap(attentionWeights, tokens, headIdx = 0) {
        // Clear previous content
        this.g.selectAll("*").remove();
        
        const numHeads = attentionWeights.shape[1];
        const seqLen = attentionWeights.shape[2];
        
        // Extract weights for specific head
        const weights = [];
        for (let i = 0; i < seqLen; i++) {
            for (let j = 0; j < seqLen; j++) {
                weights.push({
                    source: i,
                    target: j,
                    weight: attentionWeights[0][headIdx][i][j],
                    sourceToken: tokens[i],
                    targetToken: tokens[j]
                });
            }
        }
        
        // Color scale
        const colorScale = d3.scaleSequential(d3.interpolateViridis)
            .domain([0, d3.max(weights, d => d.weight)]);
        
        // Cell size
        const cellSize = Math.min(this.width, this.height) / seqLen;
        
        // Create cells
        const cells = this.g.selectAll(".attention-cell")
            .data(weights)
            .enter()
            .append("rect")
            .attr("class", "attention-cell")
            .attr("x", d => d.target * cellSize)
            .attr("y", d => d.source * cellSize)
            .attr("width", cellSize - 1)
            .attr("height", cellSize - 1)
            .attr("fill", d => colorScale(d.weight))
            .attr("stroke", "#fff")
            .attr("stroke-width", 0.5);
        
        // Add interactivity
        const tooltip = d3.select("body")
            .append("div")
            .attr("class", "attention-tooltip")
            .style("opacity", 0)
            .style("position", "absolute")
            .style("background", "rgba(0, 0, 0, 0.8)")
            .style("color", "white")
            .style("padding", "10px")
            .style("border-radius", "5px")
            .style("font-size", "12px");
        
        cells
            .on("mouseover", function(event, d) {
                d3.select(this).attr("stroke-width", 2);
                tooltip.transition().duration(200).style("opacity", .9);
                tooltip.html(`
                    <strong>Attention Weight:</strong> ${d.weight.toFixed(4)}<br/>
                    <strong>From:</strong> "${d.sourceToken}" (pos ${d.source})<br/>
                    <strong>To:</strong> "${d.targetToken}" (pos ${d.target})
                `)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 28) + "px");
            })
            .on("mouseout", function(d) {
                d3.select(this).attr("stroke-width", 0.5);
                tooltip.transition().duration(500).style("opacity", 0);
            });
        
        // Add token labels
        this.g.selectAll(".token-label-x")
            .data(tokens)
            .enter()
            .append("text")
            .attr("class", "token-label-x")
            .attr("x", (d, i) => i * cellSize + cellSize / 2)
            .attr("y", -10)
            .attr("text-anchor", "middle")
            .attr("font-size", "10px")
            .text(d => d.length > 8 ? d.substring(0, 8) + "..." : d);
        
        this.g.selectAll(".token-label-y")
            .data(tokens)
            .enter()
            .append("text")
            .attr("class", "token-label-y")
            .attr("x", -10)
            .attr("y", (d, i) => i * cellSize + cellSize / 2)
            .attr("text-anchor", "end")
            .attr("dominant-baseline", "middle")
            .attr("font-size", "10px")
            .text(d => d.length > 8 ? d.substring(0, 8) + "..." : d);
        
        // Add head selector
        this.addHeadSelector(numHeads, headIdx, (newHeadIdx) => {
            this.renderHeatmap(attentionWeights, tokens, newHeadIdx);
        });
        
        // Add colorbar
        this.addColorbar(colorScale);
    }
    
    /**
     * Render attention flow diagram
     */
    renderFlowDiagram(attentionWeights, tokens, headIdx = 0, threshold = 0.1) {
        this.g.selectAll("*").remove();
        
        const seqLen = tokens.length;
        const radius = Math.min(this.width, this.height) / 2 - 50;
        
        // Position tokens in a circle
        const tokenPositions = tokens.map((token, i) => ({
            token: token,
            index: i,
            x: radius * Math.cos(2 * Math.PI * i / seqLen - Math.PI / 2),
            y: radius * Math.sin(2 * Math.PI * i / seqLen - Math.PI / 2)
        }));
        
        // Create center group
        const centerG = this.g.append("g")
            .attr("transform", `translate(${this.width/2}, ${this.height/2})`);
        
        // Draw attention flows
        const flows = [];
        for (let i = 0; i < seqLen; i++) {
            for (let j = 0; j < seqLen; j++) {
                const weight = attentionWeights[0][headIdx][i][j];
                if (weight > threshold && i !== j) {
                    flows.push({
                        source: tokenPositions[i],
                        target: tokenPositions[j],
                        weight: weight
                    });
                }
            }
        }
        
        // Color and width scales
        const colorScale = d3.scaleSequential(d3.interpolateBlues)
            .domain([0, d3.max(flows, d => d.weight)]);
        const widthScale = d3.scaleLinear()
            .domain([0, d3.max(flows, d => d.weight)])
            .range([1, 8]);
        
        // Draw flow lines
        centerG.selectAll(".attention-flow")
            .data(flows)
            .enter()
            .append("path")
            .attr("class", "attention-flow")
            .attr("d", d => {
                const dx = d.target.x - d.source.x;
                const dy = d.target.y - d.source.y;
                const dr = Math.sqrt(dx * dx + dy * dy) * 1.5;
                return `M${d.source.x},${d.source.y}A${dr},${dr} 0 0,1 ${d.target.x},${d.target.y}`;
            })
            .attr("fill", "none")
            .attr("stroke", d => colorScale(d.weight))
            .attr("stroke-width", d => widthScale(d.weight))
            .attr("opacity", 0.6)
            .append("title")
            .text(d => `${d.source.token} → ${d.target.token}: ${d.weight.toFixed(4)}`);
        
        // Draw token nodes
        centerG.selectAll(".token-node")
            .data(tokenPositions)
            .enter()
            .append("circle")
            .attr("class", "token-node")
            .attr("cx", d => d.x)
            .attr("cy", d => d.y)
            .attr("r", 15)
            .attr("fill", "#4CAF50")
            .attr("stroke", "#2E7D32")
            .attr("stroke-width", 2);
        
        // Add token labels
        centerG.selectAll(".token-text")
            .data(tokenPositions)
            .enter()
            .append("text")
            .attr("class", "token-text")
            .attr("x", d => d.x)
            .attr("y", d => d.y)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "middle")
            .attr("font-size", "10px")
            .attr("font-weight", "bold")
            .attr("fill", "white")
            .text(d => d.token.length > 3 ? d.token.substring(0, 3) : d.token);
        
        // Add threshold slider
        this.addThresholdSlider(threshold, (newThreshold) => {
            this.renderFlowDiagram(attentionWeights, tokens, headIdx, newThreshold);
        });
    }
    
    addHeadSelector(numHeads, currentHead, callback) {
        const selector = this.container
            .append("div")
            .style("margin", "10px 0");
        
        selector.append("label")
            .text("Attention Head: ")
            .style("margin-right", "10px");
        
        const select = selector.append("select")
            .on("change", function() {
                callback(+this.value);
            });
        
        select.selectAll("option")
            .data(d3.range(numHeads))
            .enter()
            .append("option")
            .attr("value", d => d)
            .property("selected", d => d === currentHead)
            .text(d => `Head ${d + 1}`);
    }
    
    addThresholdSlider(currentThreshold, callback) {
        const slider = this.container
            .append("div")
            .style("margin", "10px 0");
        
        slider.append("label")
            .text("Attention Threshold: ")
            .style("margin-right", "10px");
        
        const input = slider.append("input")
            .attr("type", "range")
            .attr("min", "0.01")
            .attr("max", "0.5")
            .attr("step", "0.01")
            .attr("value", currentThreshold)
            .on("input", function() {
                callback(+this.value);
            });
        
        slider.append("span")
            .text(` ${currentThreshold}`)
            .attr("id", "threshold-value");
    }
    
    addColorbar(colorScale) {
        const colorbarWidth = 200;
        const colorbarHeight = 20;
        
        const colorbar = this.svg
            .append("g")
            .attr("class", "colorbar")
            .attr("transform", `translate(${this.width - colorbarWidth}, ${this.height + 40})`);
        
        const gradient = this.svg.append("defs")
            .append("linearGradient")
            .attr("id", "attention-gradient")
            .attr("x1", "0%")
            .attr("y1", "0%")
            .attr("x2", "100%")
            .attr("y2", "0%");
        
        const domain = colorScale.domain();
        gradient.selectAll("stop")
            .data(d3.range(0, 1.1, 0.1))
            .enter()
            .append("stop")
            .attr("offset", d => `${d * 100}%`)
            .attr("stop-color", d => colorScale(domain[0] + d * (domain[1] - domain[0])));
        
        colorbar.append("rect")
            .attr("width", colorbarWidth)
            .attr("height", colorbarHeight)
            .attr("fill", "url(#attention-gradient)");
        
        colorbar.append("text")
            .attr("x", 0)
            .attr("y", -5)
            .text(domain[0].toFixed(3));
        
        colorbar.append("text")
            .attr("x", colorbarWidth)
            .attr("y", -5)
            .attr("text-anchor", "end")
            .text(domain[1].toFixed(3));
    }
}

// Export for use in Streamlit
window.AttentionVisualizer = AttentionVisualizer;