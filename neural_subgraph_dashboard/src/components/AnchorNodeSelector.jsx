import React from 'react';
import styled from 'styled-components';

const SelectorCard = styled.div`
  background: rgba(255, 255, 255, 0.9);
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  height: 300px;
  display: flex;
  flex-direction: column;
`;

const AnchorList = styled.div`
  flex: 1;
  overflow-y: auto;
  border: 1px solid #dee2e6;
  border-radius: 5px;
  padding: 10px;
`;

const AnchorItem = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px;
  border-bottom: 1px solid #f1f3f4;
  font-size: 0.85em;
  
  &:last-child {
    border-bottom: none;
  }
`;

const ScoreBadge = styled.span`
  background: ${props => {
    const score = parseFloat(props.score) || 0;
    if (score > 0.7) return '#28a745';
    if (score > 0.5) return '#ffc107';
    return '#dc3545';
  }};
  color: white;
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 0.8em;
`;

const AnchorNodeSelector = ({ anchorHistory }) => {
  const recentAnchors = anchorHistory.slice(-10).reverse();

  return (
    <SelectorCard>
      <h3>Anchor Selection History</h3>
      
      <AnchorList>
        {recentAnchors.length === 0 ? (
          <div style={{ textAlign: 'center', color: '#666', marginTop: '20px' }}>
            No anchor selections yet
          </div>
        ) : (
          recentAnchors.map((anchor, index) => (
            <AnchorItem key={index}>
              <div>
                <div><strong>Node {anchor.node_id}</strong></div>
                <div style={{ color: '#666', fontSize: '0.8em' }}>
                  Graph {anchor.graph_idx}
                </div>
                <div style={{ color: '#666', fontSize: '0.8em' }}>
                  {anchor.reason}
                </div>
              </div>
              <ScoreBadge score={anchor.score}>
                {anchor.score?.toFixed(3)}
              </ScoreBadge>
            </AnchorItem>
          ))
        )}
      </AnchorList>
    </SelectorCard>
  );
};

export default AnchorNodeSelector;