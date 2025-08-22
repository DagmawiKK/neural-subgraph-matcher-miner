import React from 'react';
import styled from 'styled-components';

const LibraryCard = styled.div`
  background: rgba(255, 255, 255, 0.9);
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  height: 400px;
  display: flex;
  flex-direction: column;
`;

const PatternList = styled.div`
  flex: 1;
  overflow-y: auto;
  border: 1px solid #dee2e6;
  border-radius: 5px;
  padding: 10px;
`;

const PatternItem = styled.div`
  border: 1px solid #e9ecef;
  border-radius: 5px;
  padding: 10px;
  margin-bottom: 10px;
  background: white;
`;

const PatternStats = styled.div`
  display: flex;
  justify-content: space-between;
  font-size: 0.8em;
  color: #666;
  margin-top: 5px;
`;

const PatternLibrary = ({ discoveredPatterns }) => {
  const sortedPatterns = discoveredPatterns
    .slice()
    .sort((a, b) => b.significance - a.significance)
    .slice(0, 10);

  return (
    <LibraryCard>
      <h3>Discovered Patterns ({discoveredPatterns.length})</h3>
      
      <PatternList>
        {sortedPatterns.length === 0 ? (
          <div style={{ textAlign: 'center', color: '#666', marginTop: '20px' }}>
            No patterns discovered yet
          </div>
        ) : (
          sortedPatterns.map((pattern, index) => (
            <PatternItem key={index}>
              <div>
                <strong>Pattern {index + 1}</strong>
                <span style={{ color: '#666', marginLeft: '10px' }}>
                  ({pattern.size} nodes, {pattern.pattern_data?.edges?.length || 0} edges)
                </span>
              </div>
              
              <PatternStats>
                <span>Frequency: {pattern.frequency}</span>
                <span>Significance: {pattern.significance?.toFixed(2)}</span>
                <span>Density: {pattern.density?.toFixed(3)}</span>
              </PatternStats>
            </PatternItem>
          ))
        )}
      </PatternList>
    </LibraryCard>
  );
};

export default PatternLibrary;