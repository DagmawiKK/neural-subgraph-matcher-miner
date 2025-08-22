import React, { useEffect, useRef } from 'react';
import styled from 'styled-components';
import * as d3 from 'd3';

const VisualizerCard = styled.div`
  background: rgba(255, 255, 255, 0.95);
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  height: 500px;
  display: flex;
  flex-direction: column;
`;

const SVGContainer = styled.div`
  flex: 1;
  overflow: hidden;
  border: 1px solid #dee2e6;
  border-radius: 5px;
`;

const StepInfo = styled.div`
  background: #f8f9fa;
  padding: 10px;
  border-radius: 5px;
  margin-bottom: 10px;
  font-size: 0.9em;
`;

const PatternGrowthVisualizer = ({ currentPattern, growthSteps }) => {
  const svgRef = useRef();
  const latestStep = growthSteps[growthSteps.length - 1];

  useEffect(() => {
    if (!currentPattern || !currentPattern.nodes) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = 400;
    const height = 350;
    
    svg.attr("width", width).attr("height", height);

    // Prepare data
    const nodes = currentPattern.nodes.map((node, i) => ({
      id: node.id,
      label: node.label,
      x: width * Math.random(),
      y: height * Math.random(),
      isAnchor: node.is_anchor,
      isNew: latestStep && node.id === latestStep.selected_node
    }));

    const links = currentPattern.edges.map(edge => ({
      source: edge.source,
      target: edge.target
    }));

    // Create force simulation
    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.id).distance(50))
      .force("charge", d3.forceManyBody().strength(-200))
      .force("center", d3.forceCenter(width / 2, height / 2));

    // Create links
    const link = svg.append("g")
      .selectAll("line")
      .data(links)
      .enter().append("line")
      .attr("stroke", "#999")
      .attr("stroke-opacity", 0.6)
      .attr("stroke-width", 2);

    // Create nodes
    const node = svg.append("g")
      .selectAll("circle")
      .data(nodes)
      .enter().append("circle")
      .attr("r", d => d.isAnchor ? 12 : 8)
      .attr("fill", d => {
        if (d.isAnchor) return "#ff4757";
        if (d.isNew) return "#2ed573";
        return "#5352ed";
      })
      .attr("stroke", "#fff")
      .attr("stroke-width", 2);

    // Add labels
    const labels = svg.append("g")
      .selectAll("text")
      .data(nodes)
      .enter().append("text")
      .text(d => d.label)
      .attr("font-size", "12px")
      .attr("dx", 15)
      .attr("dy", 4);

    // Update positions on tick
    simulation.on("tick", () => {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      node
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);

      labels
        .attr("x", d => d.x)
        .attr("y", d => d.y);
    });

  }, [currentPattern, latestStep]);

  return (
    <VisualizerCard>
      <h3>Pattern Growth</h3>
      
      {latestStep && (
        <StepInfo>
          <strong>Step {latestStep.step_idx}:</strong> 
          Pattern size: {latestStep.pattern_size}, 
          Selected node: {latestStep.selected_node}, 
          Score: {latestStep.selection_score?.toFixed(3)}
        </StepInfo>
      )}
      
      <SVGContainer>
        <svg ref={svgRef}></svg>
      </SVGContainer>
    </VisualizerCard>
  );
};

export default PatternGrowthVisualizer;