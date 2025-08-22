import asyncio
import websockets
import json
import threading
import time
from queue import Queue, Empty
import networkx as nx
from typing import Dict, List, Any

class SearchDashboardServer:
    def __init__(self, port=8765):
        self.port = port
        self.clients = set()
        self.search_data_queue = Queue()
        self.current_search_state = {
            'status': 'idle',
            'current_trial': 0,
            'total_trials': 0,
            'current_pattern': None,
            'discovered_patterns': [],
            'anchor_selection_history': [],
            'pattern_growth_steps': [],
            'search_metrics': {
                'total_time': 0,
                'patterns_per_second': 0,
                'avg_pattern_size': 0
            }
        }
    
    async def register_client(self, websocket, path):
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")
        
        # Send current state to new client
        await websocket.send(json.dumps({
            'type': 'initial_state',
            'data': self.current_search_state
        }))
        
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
            print(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def broadcast_update(self, message_type: str, data: Dict[str, Any]):
        if not self.clients:
            return
            
        message = json.dumps({
            'type': message_type,
            'data': data,
            'timestamp': time.time()
        })
        
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected
    
    def update_search_status(self, status: str, **kwargs):
        self.current_search_state['status'] = status
        self.current_search_state.update(kwargs)
        
        # Queue for async broadcast
        self.search_data_queue.put({
            'type': 'search_status',
            'data': self.current_search_state.copy()
        })
    
    def log_anchor_selection(self, graph_idx: int, node_id: str, score: float, reason: str):
        anchor_data = {
            'graph_idx': graph_idx,
            'node_id': node_id,
            'score': score,
            'reason': reason,
            'timestamp': time.time()
        }
        
        self.current_search_state['anchor_selection_history'].append(anchor_data)
        
        self.search_data_queue.put({
            'type': 'anchor_selected',
            'data': anchor_data
        })
    
    def log_pattern_growth_step(self, trial_idx: int, step_idx: int, 
                               current_pattern: nx.Graph, candidate_nodes: List[str],
                               selected_node: str, selection_score: float):
        # Convert NetworkX graph to JSON-serializable format
        pattern_data = self._serialize_graph(current_pattern)
        
        growth_step = {
            'trial_idx': trial_idx,
            'step_idx': step_idx,
            'pattern_size': len(current_pattern),
            'pattern_data': pattern_data,
            'candidate_nodes': candidate_nodes,
            'selected_node': selected_node,
            'selection_score': selection_score,
            'timestamp': time.time()
        }
        
        self.current_search_state['pattern_growth_steps'].append(growth_step)
        self.current_search_state['current_pattern'] = pattern_data
        
        self.search_data_queue.put({
            'type': 'pattern_growth',
            'data': growth_step
        })
    
    def log_pattern_discovered(self, pattern: nx.Graph, frequency: int, significance: float):
        pattern_data = self._serialize_graph(pattern)
        pattern_info = {
            'pattern_data': pattern_data,
            'frequency': frequency,
            'significance': significance,
            'size': len(pattern),
            'edges': pattern.number_of_edges(),
            'density': pattern.number_of_edges() / (len(pattern) * (len(pattern) - 1)) if len(pattern) > 1 else 0,
            'discovered_at': time.time()
        }
        
        self.current_search_state['discovered_patterns'].append(pattern_info)
        
        self.search_data_queue.put({
            'type': 'pattern_discovered',
            'data': pattern_info
        })
    
    def _serialize_graph(self, graph: nx.Graph) -> Dict[str, Any]:
        """Convert NetworkX graph to JSON-serializable format"""
        return {
            'nodes': [
                {
                    'id': str(node),
                    'label': graph.nodes[node].get('label', str(node)),
                    'attributes': {k: v for k, v in graph.nodes[node].items() 
                                 if k not in ['id', 'label']},
                    'is_anchor': graph.nodes[node].get('anchor', 0) == 1
                }
                for node in graph.nodes()
            ],
            'edges': [
                {
                    'source': str(u),
                    'target': str(v),
                    'attributes': data
                }
                for u, v, data in graph.edges(data=True)
            ],
            'directed': graph.is_directed()
        }
    
    async def process_queue(self):
        """Process queued messages for broadcasting"""
        while True:
            try:
                message = self.search_data_queue.get_nowait()
                await self.broadcast_update(message['type'], message['data'])
            except Empty:
                await asyncio.sleep(0.1)
    
    def start_server(self):
        """Start the WebSocket server"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Start queue processor
        loop.create_task(self.process_queue())
        
        # Start WebSocket server
        start_server = websockets.serve(self.register_client, "localhost", self.port)
        
        print(f"Dashboard server starting on ws://localhost:{self.port}")
        loop.run_until_complete(start_server)
        loop.run_forever()

# Global instance
dashboard_server = None

def init_dashboard_server(port=8765):
    global dashboard_server
    if dashboard_server is None:
        dashboard_server = SearchDashboardServer(port)
        
        # Start server in separate thread
        server_thread = threading.Thread(target=dashboard_server.start_server, daemon=True)
        server_thread.start()
        time.sleep(1)  # Give server time to start
    
    return dashboard_server

def log_search_event(event_type: str, **kwargs):
    """Convenience function to log search events"""
    global dashboard_server
    if dashboard_server:
        if event_type == 'anchor_selection':
            dashboard_server.log_anchor_selection(**kwargs)
        elif event_type == 'pattern_growth':
            dashboard_server.log_pattern_growth_step(**kwargs)
        elif event_type == 'pattern_discovered':
            dashboard_server.log_pattern_discovered(**kwargs)
        elif event_type == 'search_status':
            dashboard_server.update_search_status(**kwargs)