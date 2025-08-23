import asyncio
import json
import threading
import time
from queue import Queue, Empty
from typing import Dict, List, Any
import networkx as nx
import websockets

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
                'avg_pattern_size': 0,
                'total_patterns': 0
            }
        }

    async def register_client(self, websocket, _path):
        self.clients.add(websocket)
        try:
            await websocket.send(json.dumps({
                'type': 'initial_state',
                'data': self.current_search_state
            }))
            await websocket.wait_closed()
        finally:
            self.clients.discard(websocket)

    async def broadcast_update(self, message_type: str, data: Dict[str, Any]):
        if not self.clients:
            return
        msg = json.dumps({'type': message_type, 'data': data, 'timestamp': time.time()})
        dead = set()
        for ws in self.clients:
            try:
                await ws.send(msg)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.clients.discard(ws)

    def update_search_status(self, status: str, **kwargs):
        self.current_search_state['status'] = status
        self.current_search_state.update(kwargs)
        self.search_data_queue.put({'type': 'search_status', 'data': self.current_search_state.copy()})

    def log_anchor_selection(self, graph_idx: int, node_id: str, score: float, reason: str):
        item = {'graph_idx': graph_idx, 'node_id': node_id, 'score': score, 'reason': reason, 'timestamp': time.time()}
        self.current_search_state['anchor_selection_history'].append(item)
        self.search_data_queue.put({'type': 'anchor_selected', 'data': item})

    def log_pattern_growth_step(self, trial_idx: int, step_idx: int,
                                current_pattern: nx.Graph, candidate_nodes: List[str],
                                selected_node: str, selection_score: float):
        pattern_data = self._serialize_graph(current_pattern)
        item = {
            'trial_idx': trial_idx,
            'step_idx': step_idx,
            'pattern_size': len(current_pattern),
            'pattern_data': pattern_data,
            'candidate_nodes': candidate_nodes,
            'selected_node': selected_node,
            'selection_score': selection_score,
            'timestamp': time.time()
        }
        self.current_search_state['pattern_growth_steps'].append(item)
        self.current_search_state['current_pattern'] = pattern_data
        self.search_data_queue.put({'type': 'pattern_growth', 'data': item})

    def log_pattern_discovered(self, pattern: nx.Graph, frequency: int, significance: float):
        pattern_data = self._serialize_graph(pattern)
        density = (pattern.number_of_edges() / (len(pattern) * (len(pattern) - 1))) if len(pattern) > 1 else 0
        info = {
            'pattern_data': pattern_data,
            'frequency': frequency,
            'significance': significance,
            'size': len(pattern),
            'edges': pattern.number_of_edges(),
            'density': density,
            'discovered_at': time.time()
        }
        self.current_search_state['discovered_patterns'].append(info)
        self.search_data_queue.put({'type': 'pattern_discovered', 'data': info})

    def _serialize_graph(self, graph: nx.Graph) -> Dict[str, Any]:
        # Gracefully handle edgeless graphs for UI (no DeepSNAP calls here)
        nodes = []
        for n in graph.nodes():
            nd = graph.nodes[n]
            nodes.append({
                'id': str(n),
                'label': nd.get('label', str(n)),
                'attributes': {k: v for k, v in nd.items() if k not in ['id', 'label']},
                'is_anchor': nd.get('anchor', 0) == 1
            })
        edges = [{'source': str(u), 'target': str(v), 'attributes': dict(d)}
                 for u, v, d in graph.edges(data=True)]
        return {'nodes': nodes, 'edges': edges, 'directed': graph.is_directed()}

    async def process_queue(self):
        while True:
            try:
                msg = self.search_data_queue.get_nowait()
                await self.broadcast_update(msg['type'], msg['data'])
            except Empty:
                await asyncio.sleep(0.05)

    def start_server(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.create_task(self.process_queue())
        server = websockets.serve(self.register_client, '127.0.0.1', self.port)
        loop.run_until_complete(server)
        print(f"Dashboard WS server running at ws://127.0.0.1:{self.port}", flush=True)
        loop.run_forever()

# Global instance
dashboard_server = None

def init_dashboard_server(port=8765):
    global dashboard_server
    if dashboard_server is None:
        dashboard_server = SearchDashboardServer(port)
        t = threading.Thread(target=dashboard_server.start_server, daemon=True)
        t.start()
        time.sleep(0.5)
    return dashboard_server

def log_search_event(event_type: str, **kwargs):
    global dashboard_server
    if not dashboard_server:
        return
    if event_type == 'search_status':
        dashboard_server.update_search_status(**kwargs)
    elif event_type == 'anchor_selection':
        dashboard_server.log_anchor_selection(**kwargs)
    elif event_type == 'pattern_growth':
        dashboard_server.log_pattern_growth_step(**kwargs)
    elif event_type == 'pattern_discovered':
        dashboard_server.log_pattern_discovered(**kwargs)