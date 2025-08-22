import React from 'react';
import styled from 'styled-components';
import SearchStatus from './components/SearchStatus';
import PatternGrowthVisualizer from './components/PatternGrowthVisualizer';
import AnchorNodeSelector from './components/AnchorNodeSelector';
import PatternLibrary from './components/PatternLibrary';
import MetricsPanel from './components/MetricsPanel';

const DashboardContainer = styled.div`
  display: grid;
  grid-template-columns: 1fr 2fr 1fr;
  grid-template-rows: auto 1fr auto;
  gap: 20px;
  padding: 20px;
  height: 100vh;
  box-sizing: border-box;
`;

const TopPanel = styled.div`
  grid-column: 1 / -1;
  display: flex;
  gap: 20px;
`;

const LeftPanel = styled.div`
  display: flex;
  flex-direction: column;
  gap: 20px;
`;

const CenterPanel = styled.div`
  display: flex;
  flex-direction: column;
`;

const RightPanel = styled.div`
  display: flex;
  flex-direction: column;
  gap: 20px;
`;

const BottomPanel = styled.div`
  grid-column: 1 / -1;
`;

function Dashboard({ searchData }) {
  return (
    <DashboardContainer>
      <TopPanel>
        <SearchStatus 
          status={searchData.status}
          currentTrial={searchData.currentTrial}
          totalTrials={searchData.totalTrials}
        />
        <MetricsPanel metrics={searchData.metrics} />
      </TopPanel>
      
      <LeftPanel>
        <AnchorNodeSelector 
          anchorHistory={searchData.anchorHistory}
        />
      </LeftPanel>
      
      <CenterPanel>
        <PatternGrowthVisualizer 
          currentPattern={searchData.currentPattern}
          growthSteps={searchData.growthSteps}
        />
      </CenterPanel>
      
      <RightPanel>
        <PatternLibrary 
          discoveredPatterns={searchData.discoveredPatterns}
        />
      </RightPanel>
    </DashboardContainer>
  );
}

export default Dashboard;