import React from 'react';
import styled from 'styled-components';

const MetricsCard = styled.div`
  background: rgba(255, 255, 255, 0.9);
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  display: flex;
  gap: 30px;
  min-width: 400px;
`;

const MetricItem = styled.div`
  text-align: center;
`;

const MetricValue = styled.div`
  font-size: 2em;
  font-weight: bold;
  color: #5352ed;
`;

const MetricLabel = styled.div`
  color: #666;
  font-size: 0.9em;
  margin-top: 5px;
`;

const MetricsPanel = ({ metrics }) => {
  const formatTime = (seconds) => {
    if (seconds < 60) return `${Math.floor(seconds)}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}m ${remainingSeconds}s`;
  };

  return (
    <MetricsCard>
      <MetricItem>
        <MetricValue>{formatTime(metrics.totalTime || 0)}</MetricValue>
        <MetricLabel>Total Time</MetricLabel>
      </MetricItem>
      
      <MetricItem>
        <MetricValue>{(metrics.patternsPerSecond || 0).toFixed(1)}</MetricValue>
        <MetricLabel>Patterns/sec</MetricLabel>
      </MetricItem>
      
      <MetricItem>
        <MetricValue>{(metrics.avgPatternSize || 0).toFixed(1)}</MetricValue>
        <MetricLabel>Avg Pattern Size</MetricLabel>
      </MetricItem>
      
      <MetricItem>
        <MetricValue>{metrics.totalPatterns || 0}</MetricValue>
        <MetricLabel>Total Patterns</MetricLabel>
      </MetricItem>
    </MetricsCard>
  );
};

export default MetricsPanel;