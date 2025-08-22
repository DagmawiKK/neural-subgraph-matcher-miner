import React from 'react';
import styled from 'styled-components';

const StatusCard = styled.div`
  background: rgba(255, 255, 255, 0.9);
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  min-width: 300px;
`;

const StatusBadge = styled.span`
  display: inline-block;
  padding: 5px 15px;
  border-radius: 20px;
  color: white;
  font-weight: bold;
  margin-bottom: 10px;
  background: ${props => {
    switch (props.status) {
      case 'idle': return '#6c757d';
      case 'starting': return '#ffc107';
      case 'running': return '#28a745';
      case 'completed': return '#17a2b8';
      default: return '#6c757d';
    }
  }};
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 20px;
  background: #e9ecef;
  border-radius: 10px;
  overflow: hidden;
  margin: 10px 0;
`;

const ProgressFill = styled.div`
  height: 100%;
  background: linear-gradient(90deg, #28a745, #20c997);
  border-radius: 10px;
  transition: width 0.3s ease;
  width: ${props => props.percentage}%;
`;

const SearchStatus = ({ status, currentTrial, totalTrials }) => {
  const percentage = totalTrials > 0 ? (currentTrial / totalTrials) * 100 : 0;

  return (
    <StatusCard>
      <h3>Search Status</h3>
      <StatusBadge status={status}>
        {status.toUpperCase()}
      </StatusBadge>
      
      <div>
        <strong>Progress:</strong> {currentTrial} / {totalTrials} trials
      </div>
      
      <ProgressBar>
        <ProgressFill percentage={percentage} />
      </ProgressBar>
      
      <div style={{ fontSize: '0.9em', color: '#666' }}>
        {percentage.toFixed(1)}% complete
      </div>
    </StatusCard>
  );
};

export default SearchStatus;