import time
import random
import numpy as np
import networkx as nx
from subgraph_mining.search_agents import GreedySearchAgent
from websocket_server import log_search_event
import torch

class DashboardGreedySearchAgent(GreedySearchAgent):
    """Enhanced Greedy Search Agent with real-time dashboard logging"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dashboard_enabled = True
        self.trial_start_time = None
        
    def run_search(self, n_trials=1000):
        """Run search with dashboard logging"""
        if self.dashboard_enabled:
            log_search_event('search_status', 
                           status='starting', 
                           total_trials=n_trials)
        
        start_time = time.time()
        out_graphs = []
        
        for trial_idx in range(n_trials):
            if self.dashboard_enabled:
                log_search_event('search_status',
                               status='running',
                               current_trial=trial_idx + 1,
                               total_trials=n_trials)
            
            self.trial_start_time = time.time()
            pattern = self._run_single_trial_with_logging(trial_idx)
            
            if pattern and len(pattern) >= self.min_pattern_size:
                out_graphs.append(pattern)
                
                if self.dashboard_enabled:
                    # Calculate frequency (simplified for demo)
                    frequency = self._calculate_pattern_frequency(pattern)
                    significance = len(pattern) * frequency  # Simplified significance
                    
                    log_search_event('pattern_discovered',
                                   pattern=pattern,
                                   frequency=frequency,
                                   significance=significance)
            
            # Update metrics
            if self.dashboard_enabled and trial_idx % 10 == 0:
                elapsed_time = time.time() - start_time
                patterns_per_second = len(out_graphs) / elapsed_time if elapsed_time > 0 else 0
                avg_size = np.mean([len(p) for p in out_graphs]) if out_graphs else 0
                
                log_search_event('search_status',
                               status='running',
                               current_trial=trial_idx + 1,
                               total_trials=n_trials,
                               search_metrics={
                                   'total_time': elapsed_time,
                                   'patterns_per_second': patterns_per_second,
                                   'avg_pattern_size': avg_size,
                                   'total_patterns': len(out_graphs)
                               })
        
        if self.dashboard_enabled:
            log_search_event('search_status',
                           status='completed',
                           total_patterns=len(out_graphs))
        
        return out_graphs
    
    def _run_single_trial_with_logging(self, trial_idx):
        """Run a single trial with detailed logging"""
        # Select seed graph
        ps = np.array([len(g) for g in self.graphs], dtype=np.float32)
        ps /= np.sum(ps)
        graph_idx = np.random.choice(range(len(ps)), p=ps)
        graph = self.graphs[graph_idx]
        
        # Select anchor node
        anchor_node = random.choice(list(graph.nodes()))
        anchor_score = self._calculate_anchor_score(graph, anchor_node)
        
        if self.dashboard_enabled:
            log_search_event('anchor_selection',
                           graph_idx=graph_idx,
                           node_id=str(anchor_node),
                           score=anchor_score,
                           reason="Random selection with graph size weighting")
        
        # Initialize pattern
        pattern = nx.Graph()
        pattern.add_node(0, **graph.nodes[anchor_node])
        if self.node_anchored:
            pattern.nodes[0]['anchor'] = 1
        
        # Grow pattern step by step
        step_idx = 0
        while len(pattern) < self.max_pattern_size:
            # Get candidate nodes from the original graph
            current_nodes = list(pattern.nodes())
            if not current_nodes:
                break
                
            # Find neighbors in original graph
            neighbors = set()
            for pattern_node in current_nodes:
                original_node = self._map_to_original_node(pattern_node, graph, anchor_node)
                if original_node in graph:
                    neighbors.update(graph.neighbors(original_node))
            
            # Remove already included nodes
            candidates = [n for n in neighbors if not self._is_node_in_pattern(n, pattern, graph, anchor_node)]
            
            if not candidates:
                break
            
            # Score candidates
            candidate_scores = {}
            for candidate in candidates:
                score = self._score_candidate_node(pattern, candidate, graph)
                candidate_scores[candidate] = score
            
            # Select best candidate
            if candidate_scores:
                best_candidate = max(candidate_scores.keys(), key=lambda x: candidate_scores[x])
                best_score = candidate_scores[best_candidate]
                
                # Add to pattern
                new_node_id = len(pattern)
                pattern.add_node(new_node_id, **graph.nodes[best_candidate])
                
                # Add edges to existing nodes in pattern
                for pattern_node in current_nodes:
                    original_pattern_node = self._map_to_original_node(pattern_node, graph, anchor_node)
                    if graph.has_edge(original_pattern_node, best_candidate):
                        pattern.add_edge(pattern_node, new_node_id)
                
                if self.dashboard_enabled:
                    log_search_event('pattern_growth',
                                   trial_idx=trial_idx,
                                   step_idx=step_idx,
                                   current_pattern=pattern.copy(),
                                   candidate_nodes=[str(c) for c in candidates[:5]],  # Top 5
                                   selected_node=str(best_candidate),
                                   selection_score=float(best_score))
                
                step_idx += 1
            else:
                break
        
        return pattern if len(pattern) >= self.min_pattern_size else None
    
    def _calculate_anchor_score(self, graph, node):
        """Calculate score for anchor node selection"""
        degree = graph.degree(node)
        centrality = degree / (len(graph) - 1) if len(graph) > 1 else 0
        return centrality
    
    def _map_to_original_node(self, pattern_node_id, graph, anchor_node):
        """Map pattern node back to original graph node"""
        if pattern_node_id == 0:
            return anchor_node
        # Simplified mapping - in practice, you'd track this properly
        return anchor_node
    
    def _is_node_in_pattern(self, node, pattern, graph, anchor_node):
        """Check if node is already in pattern"""
        return node == anchor_node  # Simplified
    
    def _score_candidate_node(self, pattern, candidate, graph):
        """Score a candidate node for addition to pattern"""
        # Use the model to score the candidate
        try:
            with torch.no_grad():
                # Create temporary pattern with candidate
                temp_pattern = pattern.copy()
                temp_pattern.add_node(len(temp_pattern), **graph.nodes[candidate])
                
                # Get embedding
                from common import utils
                anchors = [0] if self.node_anchored else None
                batch = utils.batch_nx_graphs([temp_pattern], anchors=anchors)
                emb = self.model.emb_model(batch)
                
                # Score against existing embeddings (simplified)
                if self.embs and len(self.embs) > 0:
                    similarities = []
                    for emb_batch in self.embs:
                        if self.model_type == "order":
                            sim = -torch.sum(torch.max(torch.zeros_like(emb), 
                                                     emb_batch[:1] - emb)**2, dim=1)
                        else:
                            sim = torch.nn.functional.cosine_similarity(emb.expand_as(emb_batch[:1]), emb_batch[:1])
                        similarities.append(torch.mean(sim).item())
                    
                    return np.mean(similarities) if similarities else 0.0
                
                return random.random()  # Fallback
        except:
            return random.random()
    
    def _calculate_pattern_frequency(self, pattern):
        """Calculate how frequently this pattern appears (simplified)"""
        # In practice, you'd do proper subgraph isomorphism
        return random.randint(1, 10)