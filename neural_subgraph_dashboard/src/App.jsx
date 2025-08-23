// filepath: /home/dagim/Pictures/neural-subgraph-matcher-miner/dashboard/src/App.jsx
import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import Dashboard from './Dashboard';
import useWebSocket from './hooks/useWebSocket';

const AppContainer = styled.div`
  width: 100vw;
  height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  overflow: hidden;
`;

const ConnectionStatus = styled.div`
  position: fixed;
  top: 20px;
  right: 20px;
  padding: 10px 20px;
  border-radius: 5px;
  color: white;
  font-weight: bold;
  z-index: 1000;
  background: ${props => props.connected ? '#4CAF50' : '#f44336'};
`;

function App() {
  const [searchData, setSearchData] = useState({
    status: 'idle',
    currentTrial: 0,
    totalTrials: 0,
    currentPattern: null,
    discoveredPatterns: [],
    anchorHistory: [],
    growthSteps: [],
    metrics: {
      totalTime: 0,
      patternsPerSecond: 0,
      avgPatternSize: 0,
      totalPatterns: 0
    }
  });

  const { connected, lastMessage } = useWebSocket('ws://localhost:8765', {
    onMessage: (message) => {
      const data = JSON.parse(message.data);
      
      switch (data.type) {
        case 'initial_state':
          setSearchData(prevData => ({
            ...prevData,
            ...data.data,
            discoveredPatterns: data.data.discovered_patterns || [],
            anchorHistory: data.data.anchor_selection_history || [],
            growthSteps: data.data.pattern_growth_steps || [],
            metrics: data.data.search_metrics || prevData.metrics
          }));
          break;
          
        case 'search_status':
          setSearchData(prevData => ({
            ...prevData,
            status: data.data.status,
            currentTrial: data.data.current_trial || prevData.currentTrial,
            totalTrials: data.data.total_trials || prevData.totalTrials,
            metrics: data.data.search_metrics || prevData.metrics
          }));
          break;
          
        case 'anchor_selected':
          setSearchData(prevData => ({
            ...prevData,
            anchorHistory: [...prevData.anchorHistory, data.data]
          }));
          break;
          
        case 'pattern_growth':
          setSearchData(prevData => ({
            ...prevData,
            currentPattern: data.data.pattern_data,
            growthSteps: [...prevData.growthSteps, data.data]
          }));
          break;
          
        case 'pattern_discovered':
          setSearchData(prevData => ({
            ...prevData,
            discoveredPatterns: [...prevData.discoveredPatterns, data.data]
          }));
          break;
          
        default:
          console.log('Unknown message type:', data.type);
      }
    }
  });

  return (
    <AppContainer>
      <ConnectionStatus connected={connected}>
        {connected ? 'Connected' : 'Disconnected'}
      </ConnectionStatus>
      <Dashboard searchData={searchData} />
    </AppContainer>
  );
}

export default App;