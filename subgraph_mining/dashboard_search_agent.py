import time
import random
import numpy as np
import networkx as nx
from subgraph_mining.search_agents import GreedySearchAgent
from neural_subgraph_dashboard.websocket_server import log_search_event
import torch
from common import utils
import scipy.stats as stats

class DashboardGreedySearchAgent(GreedySearchAgent):
    """Enhanced Greedy Search Agent with real-time dashboard logging"""
    
    def __init__(self, min_pattern_size, max_pattern_size, model, graphs, embs, node_anchored=False,
                 analyze=False, model_type="order", out_batch_size=20, n_beams=1, n_workers=1):
        super().__init__(min_pattern_size, max_pattern_size, model, graphs, embs, node_anchored,
                         analyze, model_type, out_batch_size, n_beams, n_workers)
        self.graphs = graphs
        self.dashboard_enabled = True
        self.trial_start_time = None
        # self.args is set in decoder.py and is available here
        
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
            if pattern and len(pattern) >= self.min_pattern_size and pattern.number_of_edges() > 0:
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
        """
        Run a single trial with detailed logging, using a corrected pattern growth logic.
        """
        # Select seed graph
        ps = np.array([len(g) for g in self.graphs], dtype=np.float32)
        ps /= np.sum(ps)
        graph_idx = np.random.choice(range(len(ps)), p=ps)
        graph = self.graphs[graph_idx]
        
        # Select anchor node
        start_node = random.choice(list(graph.nodes()))
        
        if self.dashboard_enabled:
            log_search_event('anchor_selection',
                           graph_idx=graph_idx,
                           node_id=str(start_node),
                           score=self._calculate_anchor_score(graph, start_node),
                           reason="Random selection with graph size weighting")
        
        # `neigh` stores the original node IDs of the pattern
        neigh = [start_node]
        
        # Determine frontier based on graph type
        if self.args.graph_type == "undirected":
            frontier = list(set(graph.neighbors(start_node)) - set(neigh))
        elif self.args.graph_type == "directed":
            frontier = list(set(graph.successors(start_node)) - set(neigh))
        else:
            frontier = []
        visited = {start_node}

        step_idx = 0
        while len(neigh) < self.max_pattern_size and frontier:
            cand_neighs, anchors = [], []
            # Limit candidates per step to avoid huge batches
            candidates_to_score = frontier[:128] 
            for cand_node in candidates_to_score:
                cand_neighs.append(graph.subgraph(neigh + [cand_node]))
                if self.node_anchored:
                    anchors.append(neigh[0])

            if not cand_neighs:
                break

            # Filter out candidate graphs with no edges BEFORE batching
            valid_cand_neighs = [g for g in cand_neighs if g.number_of_edges() > 0]
            valid_candidates = [c for g, c in zip(cand_neighs, candidates_to_score) if g.number_of_edges() > 0]
            valid_anchors = [a for g, a in zip(cand_neighs, anchors) if g.number_of_edges() > 0] if self.node_anchored else None

            if not valid_cand_neighs:
                break

            # Batch process embeddings for valid candidates
            with torch.no_grad():
                cand_embs = self.model.emb_model(utils.batch_nx_graphs(
                    valid_cand_neighs, anchors=valid_anchors))

            best_score = float("inf")
            best_node = None

            # Score the valid candidates
            for cand_node, cand_emb in zip(valid_candidates, cand_embs):
                score = 0
                for emb_batch in self.embs:
                    with torch.no_grad():
                        if self.model_type == "order":
                            pred = self.model.predict((emb_batch.to(utils.get_device()), cand_emb)).unsqueeze(1)
                            score -= torch.sum(torch.argmax(self.model.clf_model(pred), axis=1)).item()
                        elif self.model_type == "mlp":
                            pred = self.model(emb_batch.to(utils.get_device()), cand_emb.unsqueeze(0).expand(len(emb_batch), -1))
                            score += torch.sum(pred[:,0]).item()
                
                if score < best_score:
                    best_score = score
                    best_node = cand_node

            if best_node is None:
                break

            # Update frontier and visited sets
            if self.args.graph_type == "undirected":
                frontier = list(((set(frontier) | set(graph.neighbors(best_node))) - visited) - {best_node})
            elif self.args.graph_type == "directed":
                frontier = list(((set(frontier) | set(graph.successors(best_node))) - visited) - {best_node})
            
            visited.add(best_node)
            neigh.append(best_node)

            # Create the pattern graph for logging
            pattern = graph.subgraph(neigh).copy()
            
            # Relabel nodes to 0, 1, 2... for consistent hashing and visualization
            mapping = {node: i for i, node in enumerate(neigh)}
            pattern = nx.relabel_nodes(pattern, mapping)

            # Set anchor attribute on the new relabeled graph
            if self.node_anchored:
                pattern.nodes[0]['anchor'] = 1

            # Emit only when the pattern has at least one edge
            if self.dashboard_enabled and pattern.number_of_edges() > 0:
                log_search_event('pattern_growth',
                               trial_idx=trial_idx,
                               step_idx=step_idx,
                               current_pattern=pattern,
                               candidate_nodes=[str(c) for c in candidates_to_score[:5]],
                               selected_node=str(best_node),
                               selection_score=float(best_score))
            
            step_idx += 1

        # Final check on the grown pattern
        final_pattern_graph = graph.subgraph(neigh)
        if len(final_pattern_graph) >= self.min_pattern_size and final_pattern_graph.number_of_edges() > 0:
            # Relabel the final pattern before returning
            mapping = {node: i for i, node in enumerate(neigh)}
            final_pattern = nx.relabel_nodes(final_pattern_graph, mapping)
            if self.node_anchored:
                final_pattern.nodes[0]['anchor'] = 1
            return final_pattern
            
        return None
    
    def _calculate_anchor_score(self, graph, node):
        """Calculate score for anchor node selection"""
        degree = graph.degree(node)
        centrality = degree / (len(graph) - 1) if len(graph) > 1 else 0
        return centrality
    
    def _calculate_pattern_frequency(self, pattern):
        """Calculate how frequently this pattern appears (simplified)"""
        # In practice, you'd do proper subgraph isomorphism
        return random.randint(1, 10)