import argparse
import csv
from itertools import combinations
import time
import os
import pickle
from deepsnap.batch import Batch
import numpy as nph import Batch
import torch as np
import torch.optim as optim
import torch.nn as nn optim
import torch.nn.functional as F
from tqdm import tqdmional as F
from tqdm import tqdm
import torch_geometric.utils as pyg_utils
from torch_geometric.datasets import TUDataset, PPI
import torch_geometric.nn as pyg_nnt Planetoid, KarateClub, QM7b
from matplotlib import cm import DataLoader
import torch_geometric.utils as pyg_utils
from common import data
from common import models as pyg_nn
from common import utilsm
from common import combined_syn
from subgraph_mining.config import parse_decoder
from subgraph_matching.config import parse_encoder
from common import utils
import matplotlib.pyplot as plt
from subgraph_mining.config import parse_decoder
import random_matching.config import parse_encoder
from scipy.io import mmreadimport visualize_pattern_graph_ext
import scipy.stats as stats_agents import GreedySearchAgent, MCTSSearchAgent, MemoryEfficientMCTSAgent, MemoryEfficientGreedyAgent, BeamSearchAgent
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from collections import defaultdict
from itertools import permutations
from queue import PriorityQueue
import matplotlib.colors as mcolors
import networkx as nx import TSNE
import picklecluster import KMeans, AgglomerativeClustering
import torch.multiprocessing as mpt
mp.set_start_method('spawn', force=True)
from sklearn.decomposition import PCA
from functools import lru_cachelors
import torch.nn as nn
class SearchAgent:
    """ Class for search strategies to identify frequent subgraphs in embedding space.
from sklearn.decomposition import PCA
    The problem is formulated as a search. The first action chooses a seed node to grow from.
    Subsequent actions chooses a node in dataset to connect to the existing subgraph pattern,
    increasing the pattern size by 1.
def bfs_chunk(graph, start_node, max_size):
    See paper for rationale and algorithm details.
    """ue = [start_node]
    def __init__(self, min_pattern_size, max_pattern_size, model, dataset,
        embs, node_anchored=False, analyze=False, model_type="order",
        out_batch_size=20):ph.neighbors(node):
        """ Subgraph pattern search by walking in embedding space.
                visited.add(neighbor)
        Args:   queue.append(neighbor)
            min_pattern_size: minimum size of frequent subgraphs to be identified.
            max_pattern_size: maximum size of frequent subgraphs to be identified.
            model: the trained subgraph matching model (PyTorch nn.Module).
            dataset: the DeepSNAP dataset for which to mine the frequent subgraph pattern.
            embs: embeddings of sampled node neighborhoods (see paper).
            node_anchored: an option to specify whether to identify node_anchored subgraph patterns.
                node_anchored search procedure has to use a node_anchored model (specified in subgraph
                matching config.py).
            analyze: whether to enable analysis visualization.
            model_type: type of the subgraph matching model (requires to be consistent with the model parameter).
            out_batch_size: the number of frequent subgraphs output by the mining algorithm for each size.
                They are predicted to be the out_batch_size most frequent subgraphs in the dataset.
        """graph_chunks
        self.min_pattern_size = min_pattern_size
        self.max_pattern_size = max_pattern_size
        self.model = modelyn.get_generator([size])
        self.dataset = dataset
        self.embs = embs)
        self.node_anchored = node_anchored
        self.analyze = analyzels=True)
        self.model_type = model_type-pattern.png")
        self.out_batch_size = out_batch_size
    graphs = []
    def run_search(self, n_trials=1000): 
        self.cand_patterns = defaultdict(list)
        self.counts = defaultdict(lambda: defaultdict(list))
        self.n_trials = n_trialsn(graph, pattern)
        for j in range(1, 3):
        self.init_search()dint(0, n_old - 1)
        while not self.is_search_done():n(graph) - 1)
            self.step()dge(u, v)
        return self.finish_search()
    return graphs
    def init_search():
        raise NotImplementedError
    chunk_dataset, task, args, chunk_index, total_chunks = args_tuple
    def step(self):me.time()
        """ Abstract method for executing a search step.
        Every step adds a new node to the subgraph pattern.tpid()} started chunk {chunk_index+1}/{total_chunks}", flush=True)
        Run_search calls step at least min_pattern_size times to generate a pattern of at least this
        size. To be inherited by concrete search strategy implementations.
        """le result is None:
        raise NotImplementedError
            if now - last_print >= 10:
class MCTSSearchAgent(SearchAgent):time('%H:%M:%S')}] Worker PID {os.getpid()} still processing chunk {chunk_index+1}/{total_chunks} ({int(now-start_time)}s elapsed)", flush=True)
    def __init__(self, min_pattern_size, max_pattern_size, model, dataset,
        embs, node_anchored=False, analyze=False, model_type="order",
        out_batch_size=20, c_uct=0.7)::%S')}] Worker PID {os.getpid()} finished chunk {chunk_index+1}/{total_chunks} in {int(time.time()-start_time)}s", flush=True)
        """ MCTS implementation of the subgraph pattern search.
        Uses MCTS strategy to search for the most common pattern.
        print(f"Error processing chunk {chunk_index}: {e}", flush=True)
        Args:n []
            c_uct: the exploration constant used in UCT criteria (See paper).
        """_growth_streaming(dataset, task, args):
        super().__init__(min_pattern_size, max_pattern_size, model, dataset,
            embs, node_anchored=node_anchored, analyze=analyze,ize=args.chunk_size)
            model_type=model_type, out_batch_size=out_batch_size)
        self.c_uct = c_uct
        assert not analyzes = []

    def init_search(self):aset)
        self.wl_hash_to_graphs = defaultdict(list) total_chunks) for idx, chunk_dataset in enumerate(dataset)]
        self.cum_action_values = defaultdict(lambda: defaultdict(float))
        self.visit_counts = defaultdict(lambda: defaultdict(float))
        self.visited_seed_nodes = set()nk, chunk_args)
        self.max_size = self.min_pattern_size
    for chunk_out_graphs in results:
    def is_search_done(self):
        return self.max_size == self.max_pattern_size + 1hs)

    def has_min_reachable_nodes(self, graph, start_node, n):
        for depth_limit in range(n+1):
            edges = nx.bfs_edges(graph, start_node, depth_limit=depth_limit)
            nodes = set([v for u, v in edges])
            if len(nodes) + 1 >= n:
                return True.number_of_edges()
        return False = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
    def step(self): max(12, min(20, num_nodes * 2))
        ps = np.array([len(g) for g in self.dataset], dtype=float)
        ps /= np.sum(ps)ase_size * 1.2, base_size)
        graph_dist = stats.rv_discrete(values=(np.arange(len(self.dataset)), ps))
            figsize = (base_size, base_size * 0.8)
        print("Size", self.max_size)
        print(len(self.visited_seed_nodes), "distinct seeds")
        for simulation_n in tqdm(range(self.n_trials //
            (self.max_pattern_size+1-self.min_pattern_size))):
            # pick seed nodees():
            best_graph_idx, best_start_node, best_score = None, None, -float("inf")
            for cand_graph_idx, cand_start_node in self.visited_seed_nodes:
                state = cand_graph_idx, cand_start_noden')
                my_visit_counts = sum(self.visit_counts[state].values())
                q_score = (sum(self.cum_action_values[state].values()) /
                    (my_visit_counts or 1))
                uct_score = self.c_uct * np.sqrt(np.log(simulation_n or 1) /
                    (my_visit_counts or 1)), 'label', 'anchor'] and v is not None}
                node_score = q_score + uct_score
                if node_score > best_score:
                    best_score = node_scorers.items():
                    best_graph_idx = cand_graph_idx
                    best_start_node = cand_start_node(value) > 8:  
            # if existing seed beats choosing a new seed
            if best_score >= self.c_uct * np.sqrt(np.log(simulation_n or 1)):
                graph_idx, start_node = best_graph_idx, best_start_node
                assert best_start_node in self.dataset[graph_idx].nodes
                graph = self.dataset[graph_idx]+ "..."
            else:   elif isinstance(value, (int, float)):
                found = Falseinstance(value, float):
                while not found:e = f"{value:.2f}" if abs(value) < 1000 else f"{value:.1e}"
                    graph_idx = np.arange(len(self.dataset))[graph_dist.rvs()]
                    graph = self.dataset[graph_idx]
                    start_node = random.choice(list(graph.nodes))
                    # don't pick isolated nodes or small islands
                    if self.has_min_reachable_nodes(graph, start_node,
                        self.min_pattern_size):
                        found = True
                self.visited_seed_nodes.add((graph_idx, start_node))
            neigh = [start_node]
            frontier = list(set(graph.neighbors(start_node)) - set(neigh))
            visited = set([start_node])
            neigh_g = nx.Graph()
            neigh_g.add_node(start_node, anchor=1)
            cur_state = graph_idx, start_nodeern, scale=3)
            state_list = [cur_state]
            while frontier and len(neigh) < self.max_size:d=42, iterations=100)
                cand_neighs, anchors = [], []
                for cand_node in frontier:, k=2.0, seed=42, iterations=50)
                    cand_neigh = graph.subgraph(neigh + [cand_node])
                    cand_neighs.append(cand_neigh)].get('label', 'unknown') for n in pattern.nodes()))
                    if self.node_anchored:et3(i) for i, label in enumerate(unique_labels)}
                        anchors.append(neigh[0])
                cand_embs = self.model.emb_model(utils.batch_nx_graphs(u, v, data in pattern.edges(data=True)))
                    cand_neighs, anchors=anchors if self.node_anchored else None))erate(unique_edge_types)}
                best_v_score, best_node_score, best_node = 0, -float("inf"), None
                for cand_node, cand_emb in zip(frontier, cand_embs):
                    score, n_embs = 0, 0
                    for emb_batch in self.embs:
                        score += torch.sum(self.model.predict((
                            emb_batch.to(utils.get_device()), cand_emb))).item()
                        n_embs += len(emb_batch)
                    EPS = 1e-10  
                    if n_embs > 0:e_node_size * 1.3
                        v_score = -np.log(score/n_embs + 1) + 1
                    else:e = 3500
                        v_score = 0  ode_size * 1.2
                    neigh_g = graph.subgraph(neigh + [cand_node]).copy()
                    neigh_g.remove_edges_from(nx.selfloop_edges(neigh_g))
                    for v in neigh_g.nodes:ze * 1.2
                        neigh_g.nodes[v]["anchor"] = 1 if v == neigh[0] else 0
                    next_state = utils.wl_hash(neigh_g,
                        node_anchored=self.node_anchored)
                    # compute node scorelabel', 'unknown')
                    parent_visit_counts = sum(self.visit_counts[cur_state].values())
                    my_visit_counts = sum(self.visit_counts[next_state].values())
                    q_score = (sum(self.cum_action_values[next_state].values()) /
                        (my_visit_counts or 1))
                    uct_score = self.c_uct * np.sqrt(np.log(parent_visit_counts or
                        1) / (my_visit_counts or 1))
                    node_score = q_score + uct_score
                    if node_score > best_node_score:abel])
                        best_node_score = node_score
                        best_v_score = v_score
                        best_node = cand_node
                frontier = list(((set(frontier) |
                    set(graph.neighbors(best_node))) - visited) -
                    set([best_node]))
                visited.add(best_node)
                neigh.append(best_node)
        regular_sizes = []
                # update visit counts, wl cache
                neigh_g = graph.subgraph(neigh).copy()
                neigh_g.remove_edges_from(nx.selfloop_edges(neigh_g))
                for v in neigh_g.nodes:e)
                    neigh_g.nodes[v]["anchor"] = 1 if v == neigh[0] else 0
                prev_state = cur_statede_sizes[i])
                cur_state = utils.wl_hash(neigh_g, node_anchored=self.node_anchored)
                state_list.append(cur_state)
                self.wl_hash_to_graphs[cur_state].append(neigh_g)
                regular_sizes.append(node_sizes[i])
            # backprop value
            for i in range(0, len(state_list) - 1):
                self.cum_action_values[state_list[i]][
                    state_list[i+1]] += best_v_score
                self.visit_counts[state_list[i]][state_list[i+1]] += 1
        self.max_size += 1ize=anchor_sizes, 
                    node_shape='s',
    def finish_search(self):rs='black', 
        counts = defaultdict(lambda: defaultdict(int))
        for _, v in self.visit_counts.items():
            for s2, count in v.items():
                counts[len(random.choice(self.wl_hash_to_graphs[s2]))][s2] += count
            nx.draw_networkx_nodes(pattern, pos, 
        cand_patterns_uniq = []gular_nodes,
        for pattern_size in range(self.min_pattern_size, self.max_pattern_size+1):
            for wl_hash, count in sorted(counts[pattern_size].items(), key=lambda
                x: x[1], reverse=True)[:self.out_batch_size]:
                cand_patterns_uniq.append(random.choice(
                    self.wl_hash_to_graphs[wl_hash]))
                print("- outputting", count, "motifs of size", pattern_size)
        return cand_patterns_uniq
def default_dd_list():y > 0.5:  
    return defaultdict(list)
            edge_alpha = 0.6
worker_model = Noneensity > 0.3:  
worker_graphs = Noneth = 2
worker_embs = Nonelpha = 0.7
worker_args = None
worker_worker_id = None            edge_width = 3

def init_greedy_worker(model, graphs, embs, args, worker_id): 
    """Add worker_id parameter for deterministic seeding"""
    global worker_model, worker_graphs, worker_embs, worker_args, worker_worker_id0.5 else 15)
         connectionstyle = "arc3,rad=0.1" if edge_density < 0.5 else "arc3,rad=0.15"
    worker_seed = getattr(args, 'seed', 42)
    process_seed = worker_seed + worker_id * 1000  # Use worker_id instead of PID        for u, v, data in pattern.edges(data=True):
    ype', 'default')
    random.seed(process_seed)[edge_type]
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)            nx.draw_networkx_edges(
    pos,
    print(f"[{time.strftime('%H:%M:%S')}] Worker {worker_id} initializing with seed {process_seed}...", flush=True), v)],
    h,
    worker_model = model                edge_color=[edge_color],
    worker_graphs = graphs
    worker_embs = embs                arrows=True,
    worker_args = argswsize=arrow_size,
    worker_worker_id = worker_idtyle='-|>',
    print(f"[{time.strftime('%H:%M:%S')}] Worker {worker_id} initialization complete.", flush=True)nnectionstyle=connectionstyle,
de_size=max(node_sizes) * 1.3,

def run_greedy_trial(trial_idx):                    min_target_margin=15
    global worker_model, worker_graphs, worker_embs, worker_args, worker_worker_id                )
    
    base_seed = getattr(worker_args, 'seed', 42)     for u, v, data in pattern.edges(data=True):
    trial_seed = base_seed + trial_idx * 10000 + worker_worker_id * 1000  # Use worker_id', 'default')
    
    random.seed(trial_seed)         
    np.random.seed(trial_seed)
    torch.manual_seed(trial_seed)                pattern, pos,

    ps = np.array([len(g) for g in worker_graphs], dtype=np.float32)
    ps /= np.sum(ps)
    graph_dist = stats.rv_discrete(values=(np.arange(len(worker_graphs)), ps))                alpha=edge_alpha,
False  
    graph_idx = np.arange(len(worker_graphs))[graph_dist.rvs()]
    graph = worker_graphs[graph_idx]
            
    # Use sorted nodes for deterministic selectionys() 
    nodes = sorted(list(graph.nodes))                 if k not in ['id', 'label', 'anchor'] and pattern.nodes[n][k] is not None]) 
    start_node = random.choice(nodes)
        
    neigh = [start_node]
    if worker_args.graph_type == "undirected":, 150 // (num_nodes + max_attrs_per_node * 5)))
        frontier = sorted(list(set(graph.neighbors(start_node)) - set(neigh)))    elif edge_density > 0.3:  
    elif worker_args.graph_type == "directed":_nodes + max_attrs_per_node * 3)))
        frontier = sorted(list(set(graph.successors(start_node)) - set(neigh)))
    visited = {start_node}, 250 // (num_nodes + max_attrs_per_node * 2)))
        
    trial_patterns = defaultdict(list) in pos.items():
    trial_counts = defaultdict(default_dd_list)

    while len(neigh) < worker_args.max_pattern_size and frontier:, 0) == 1
        cand_neighs, anchors = [], []
        y > 0.5:
        # Sort frontier for deterministic processing                pad = 0.15
        frontier = sorted(frontier)
        
        for cand_node in frontier:            else:
            cand_neigh = graph.subgraph(neigh + [cand_node])
            cand_neighs.append(cand_neigh)
            if worker_args.node_anchored:    bbox_props = dict(
                anchors.append(neigh[0])else (1, 0.8, 0.8, 0.6),
 if is_anchor else 'gray',
        if not cand_neighs:        alpha=0.8,
            breakad={pad}'

        with torch.no_grad():
            cand_embs = worker_model.emb_model(utils.batch_nx_graphs(
                cand_neighs, anchors=anchors if worker_args.node_anchored else None))
                    fontweight='bold' if is_anchor else 'normal',
        best_score = float("inf")black',
        best_node = None   ha='center', va='center',
                    bbox=bbox_props)
        for cand_node, cand_emb in zip(frontier, cand_embs):
            score = 0
            for emb_batch in worker_embs:
                with torch.no_grad():            for u, v, data in pattern.edges(data=True):
                    if worker_args.method_type == "order":.get('type') or 
                        pred = worker_model.predict((emb_batch.to(utils.get_device()), cand_emb)).unsqueeze(1)   data.get('label') or 
                        score -= torch.sum(torch.argmax(worker_model.clf_model(pred), axis=1)).item()                           data.get('input_label') or
                    elif worker_args.method_type == "mlp":
                        pred = worker_model(emb_batch.to(utils.get_device()), cand_emb.unsqueeze(0).expand(len(emb_batch), -1))      data.get('edge_type'))
                        score += torch.sum(pred[:,0]).item()
)] = str(edge_type)
            if score < best_score:
                best_score = score
                best_node = cand_node
 
        if best_node is None:
            break
                          font_color='black',
        if worker_args.graph_type == "undirected":t(facecolor='white', edgecolor='lightgray', 
            frontier = sorted(list(((set(frontier) | set(graph.neighbors(best_node))) - visited) - {best_node}))alpha=0.8, boxstyle='round,pad=0.1'))
        elif worker_args.graph_type == "directed":
            frontier = sorted(list(((set(frontier) | set(graph.successors(best_node))) - visited) - {best_node}))        graph_type = "Directed" if pattern.is_directed() else "Undirected"
              tern.nodes[n].get('anchor', 0) == 1 for n in pattern.nodes())
        visited.add(best_node)fo = " (Red squares = anchor nodes)" if has_anchors else ""
        neigh.append(best_node)        
tern.nodes[n].keys() 
        if len(neigh) >= worker_args.min_pattern_size:
            neigh_g = graph.subgraph(neigh).copy()s())
            neigh_g.remove_edges_from(nx.selfloop_edges(neigh_g))
            for v_idx, v in enumerate(neigh_g.nodes):
                neigh_g.nodes[v]["anchor"] = 1 if worker_args.node_anchored and v == neigh[0] else 0ty: {edge_density:.2f}"

            trial_patterns[len(neigh_g)].append((best_score, neigh_g))            density_info += " (Very Dense)"
            trial_counts[len(neigh_g)][utils.wl_hash(neigh_g, node_anchored=worker_args.node_anchored)].append(neigh_g)
            
    return trial_patterns, trial_counts


        title = f"{graph_type} Pattern Graph{anchor_info}\n"
class GreedySearchAgent(SearchAgent):fo}, {density_info})"
    def __init__(self, min_pattern_size, max_pattern_size, model, dataset,
        embs, node_anchored=False, analyze=False, rank_method="counts",title(title, fontsize=14, fontweight='bold')
        model_type="order", out_batch_size=20, n_beams=1, n_workers=4):
        super().__init__(min_pattern_size, max_pattern_size, model, dataset,
            embs, node_anchored=node_anchored, analyze=analyze,        if unique_edge_types and len(unique_edge_types) > 1:
            model_type=model_type, out_batch_size=out_batch_size)            x_pos = 1.2  
        self.rank_method = rank_method
        self.n_beams = n_beams
        self.n_workers = n_workers
        print("Rank Method:", rank_method)
        if self.n_workers > 1:
            print(f"Using {self.n_workers} worker processes for parallel search.")

    def run_search(self, n_trials=1000):n edge_color_map.items()
        """
        Overridden run_search that uses an initializer to avoid repetitive data transfer.
        """
        self.cand_patterns = defaultdict(list)egend_elements,
        self.counts = defaultdict(lambda: defaultdict(list))
        self.n_trials = n_trials                bbox_to_anchor=(x_pos, y_pos),

        init_args = (self.model, self.dataset, self.embs, self.args)     framealpha=0.9,
        
        args_for_pool = range(n_trials)     fontsize=9

        print(f"Starting {n_trials} search trials on {self.n_workers} cores...")
        with mp.Pool(processes=self.n_workers, initializer=init_greedy_worker, initargs=init_args) as pool:
            results = list(tqdm(pool.imap_unordered(run_greedy_trial, args_for_pool), total=n_trials))            plt.tight_layout(rect=[0, 0, 0.85, 1])

        print("Aggregating results from all worker processes...")    plt.tight_layout()
        for trial_patterns, trial_counts in results:
            for size, scored_patterns in trial_patterns.items():        pattern_info = [f"{num_nodes}-{count_by_size[num_nodes]}"]
                self.cand_patterns[size].extend(scored_patterns)
            for size, hashed_patterns in trial_counts.items():
                for h, graphs in hashed_patterns.items():
                    self.counts[size][h].extend(graphs)            pattern_info.append('nodes-' + '-'.join(node_types))

        return self.finish_search()('type', '') for e in pattern.edges()))

    def finish_search(self):
        """
        Processes the aggregated results from all trials to find the most frequent patterns.
        This method remains unchanged.
        """
        if self.analyze:
            pass            pattern_info.append(f'{total_node_attrs}attrs')

        cand_patterns_uniq = []edge_density > 0.5:
        for pattern_size in range(self.min_pattern_size, self.max_pattern_size + 1):
            if self.rank_method == "hybrid":
                if self.counts[pattern_size]: pattern_info.append('dense')
                    cur_rank_method = "margin" if len(max(
                        self.counts[pattern_size].values(), key=len, default=[])) < 3 else "counts"ern_info.append('sparse')
                else:
                    cur_rank_method = "margin"" if pattern.is_directed() else "undir"
            else:
                cur_rank_method = self.rank_method
}.png", bbox_inches='tight', dpi=300)
            print(f"Ranking patterns of size {pattern_size} using method: '{cur_rank_method}'")inches='tight')

            if cur_rank_method == "margin":
                wl_hashes = set()
                cands = self.cand_patterns[pattern_size]ion as e:
                cand_patterns_uniq_size = []e}")
                for score, pattern in sorted(cands, key=lambda x: x[0]):        return False
                    wl_hash = utils.wl_hash(pattern, node_anchored=self.node_anchored)
                    if wl_hash not in wl_hashes:def pattern_growth(dataset, task, args):
                        wl_hashes.add(wl_hash)pattern growth
                        cand_patterns_uniq_size.append(pattern)
                        if len(cand_patterns_uniq_size) >= self.out_batch_size:
                            break
                cand_patterns_uniq.extend(cand_patterns_uniq_size)
                
            elif cur_rank_method == "counts":
                sorted_counts = sorted(self.counts[pattern_size].items(), key=lambda x: len(x[1]), reverse=True)den_dim, args)
                for _, neighs in sorted_counts[:self.out_batch_size]:
                    cand_patterns_uniq.append(random.choice(neighs))
            else:
                print("Unrecognized rank method")
                tate_dict(torch.load(args.model_path,
        return cand_patterns_uniq

class MemoryEfficientGreedyAgent(GreedySearchAgent):
    def __init__(self, min_pattern_size, max_pattern_size, model, dataset,
        embs, node_anchored=False, analyze=False, rank_method="counts",
        model_type="order", out_batch_size=20, batch_size=64):
        super().__init__(min_pattern_size, max_pattern_size, model, dataset,taset), "graphs")
            embs, node_anchored=node_anchored, analyze=analyze,s.search_strategy)
            rank_method=rank_method, model_type=model_type,    print("graph type:", args.graph_type)
            out_batch_size=out_batch_size)")
        self.batch_size = batch_size
        self.use_fp16 = torch.cuda.is_available()
        
    def _grow_pattern(self, graph, start_node):
        neigh = [start_node]
        visited = {start_node}== nx.DiGraph:
        frontier = set(graph.neighbors(start_node))graph).to_undirected()
    ):
        while frontier and len(neigh) < self.max_pattern_size:e]:
            best_score = float('inf')            graph.nodes[node]['label'] = str(node)
            best_node = None]:
        des[node]['id'] = str(node)
            for i in range(0, len(frontier), self.batch_size):
                batch_nodes = list(frontier)[i:i+self.batch_size]
                cand_neighs = [graph.subgraph(neigh + [n]) for n in batch_nodes]if args.use_whole_graphs:
                anchors = [neigh[0]] * len(cand_neighs) if self.node_anchored else None
            
                with torch.no_grad():
                    cand_embs = self.model.emb_model(utils.batch_nx_graphs(if args.sample_method == "radial":
                        cand_neighs, anchors=anchors))
                
                    if self.use_fp16:
                        cand_embs = self._half_tensor(cand_embs)
                for i, graph in enumerate(graphs):
                    for node, emb in zip(batch_nodes, cand_embs):ode iteration
                        score = 0
                        for emb_batch in self.embs:
                            if self.use_fp16:    if len(dataset) <= 10 and j % 100 == 0: 
                                emb_batch = self._half_tensor(emb_batch)
                            
                            if self.model_type == "order":    # Set seed based on position for determinism
                                pred = self.model.predict((
                                    emb_batch.to(utils.get_device()),d(args.seed + i * 10000 + j)
                                    emb)).unsqueeze(1)
                                if self.use_fp16:
                                    pred = pred.float()
                                score -= torch.sum(torch.argmax(
                                    self.model.clf_model(pred), axis=1)).item()test_path_length(graph,
                            elif self.model_type == "mlp":))
                                pred = self.model(
                                    emb_batch.to(utils.get_device()),h, min(len(neigh),
                                    emb.unsqueeze(0).expand(len(emb_batch), -1)ple_size))
                                    )
                                if self.use_fp16:
                                    pred = pred.float()
                                score += torch.sum(pred[:,0]).item()ax(
                                nents(subgraph), key=len))
                        if score < best_score:
                            best_score = scoreraph.nodes()}
                            best_node = node{(u,v): subgraph.edges[u,v].copy() 
        bgraph.edges()}
            if best_node is None:
                breake(subgraph.nodes())}
             = nx.relabel_nodes(subgraph, mapping)
            neigh.append(best_node)
            visited.add(best_node)g.items():
            frontier = set((frontier | set(graph.neighbors(best_node))) - ew].update(orig_attrs[old])
                     visited - {best_node})                
            u, old_v), attrs in edge_attrs.items():
        if len(neigh) >= self.min_pattern_size:       subgraph.edges[mapping[old_u], mapping[old_v]].update(attrs)
            pattern = graph.subgraph(neigh).copy()            
            pattern.remove_edges_from(nx.selfloop_edges(pattern))d_edge(0, 0)
            for v in pattern.nodes:end(subgraph)
                pattern.nodes[v]["anchor"] = 1 if v == neigh[0] else 0
            0)
            if self.analyze: args.sample_method == "tree":
                emb = self.model.emb_model(utils.batch_nx_graphs(
                    [pattern], anchors=[neigh[0]] if self.node_anchored else None)).squeeze(0)
                self.analyze_embs.append([emb.detach().cpu().numpy()])
             2000)
            self.cand_patterns[len(pattern)].append((best_score, pattern))
            if self.rank_method in ["counts", "hybrid"]:
                self.counts[len(pattern)][utils.wl_hash(pattern,nge(args.n_neighborhoods)):
                    node_anchored=self.node_anchored)].append(pattern)
            
            return pattern
        return None    

    def step(self):_size,
        if torch.cuda.is_available():type)
            torch.cuda.empty_cache()
        neigh = nx.convert_node_labels_to_integers(neigh)
        new_beam_sets = []edge(0, 0)
        processed_graphs = set()ghs.append(neigh)
                        if args.node_anchored:
        for beam_set in tqdm(self.beam_sets): anchors.append(0)
            if isinstance(beam_set, (list, tuple)) and len(beam_set) > 0:
                if isinstance(beam_set[0], (list, tuple)):
                    graph_idx = beam_set[0][-1]if len(neighs) % args.batch_size != 0:
                else:mber of graphs not multiple of batch size")
                    graph_idx = beam_set[-1]/ args.batch_size):
                processed_graphs.add(graph_idx)top = (i+1)*args.batch_size
            
            patterns = []
            try: None)
                states = [beam_set] if not isinstance(beam_set[0], (list, tuple)) else beam_set
                for state in states:.to(torch.device("cpu"))
                    if len(state) >= 5:
                        _, neigh, frontier, visited, graph_idx = state
                        graph = self.dataset[graph_idx]analyze:
                        tack(embs).numpy()
                        for node in list(frontier)[:self.batch_size]:ter(embs_np[:,0], embs_np[:,1], label="node neighborhood")
                            pattern = self._grow_pattern(graph, node)
                            if pattern is not None:):
                                patterns.append(pattern)
                
                if patterns:
                    patterns.sort(key=len, reverse=True)egy == "mcts":
                    new_beam_sets.append(patterns[:self.n_beams])
                    
            except Exception as e:in_pattern_size, args.max_pattern_size,
                print(f"Error processing beam: {e}")node_anchored,
                continueanalyze=args.analyze, out_batch_size=args.out_batch_size)

        print(f"Processing beams from {len(processed_graphs)} distinct graphs")e, args.max_pattern_size,
        self.beam_sets = [b for b in new_beam_sets if b]ored,
yze=args.analyze, out_batch_size=args.out_batch_size)
class MemoryEfficientMCTSAgent(MCTSSearchAgent):greedy":
    """Memory-efficient MCTS implementation with legacy AMP support"""
    ryEfficientGreedyAgent(args.min_pattern_size, args.max_pattern_size,
    def __init__(self, min_pattern_size, max_pattern_size, model, dataset,                model, graphs, embs, node_anchored=args.node_anchored,
        embs, node_anchored=False, analyze=False, model_type="order",
        out_batch_size=20, c_uct=0.7, memory_limit=1000000):
        super().__init__(min_pattern_size, max_pattern_size, model, dataset,        else:
            embs, node_anchored=node_anchored, analyze=analyze,attern_size, args.max_pattern_size,
            model_type=model_type, out_batch_size=out_batch_size, c_uct=c_uct)
        self.memory_limit = memory_limit            analyze=args.analyze, model_type=args.method_type,
        self.wl_hash_to_graphs = self._create_lru_cache(maxsize=10000)
        self.use_fp16 = torch.cuda.is_available()
        
    def _half_tensor(self, tensor):
        """Helper to convert tensor to FP16 if CUDA is available"""_pattern_size,
        return tensor.half() if self.use_fp16 else tensor
        type=args.method_type,
    def _create_lru_cache(self, maxsize):idth)
        """Create a size-limited LRU cache for storing graph patterns"""
        from functools import lru_cachen search
        return lru_cache(maxsize=maxsize)rgs.n_trials)
        
    def _stream_neighborhood(self, graph, start_node, max_nodes=1000):
        """Stream neighborhoods instead of loading all at once"""int(time.time() - start_time)
        visited = {start_node})
        frontier = set(graph.neighbors(start_node))
        while frontier and len(visited) < max_nodes:
            node = frontier.pop()
            if node not in visited:ings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
                visited.add(node)
                frontier.update(n for n in graph.neighbors(node) 
                              if n not in visited)
                yield nodes, count_by_size):
                
    def _batch_embeddings(self, cand_neighs, batch_size=64):)] += 1
        """Process embeddings in batches with FP16 support"""
        for i in range(0, len(cand_neighs), batch_size):ed {successful_visualizations}/{len(out_graphs)} patterns")
            batch = cand_neighs[i:i+batch_size]
            # Filter out graphs with no edges
            valid_batch = [g for g in batch if g.number_of_edges() > 0]"results"):
        irs("results")
            # Skip if no valid graphs in this batch
            if not valid_batch:
                continue
            
            anchors = None
            if self.node_anchored:
                anchors = [list(g.nodes)[0] for g in valid_batch]et seeds for all random number generators to ensure reproducibility."""
        
            with torch.no_grad():
                embs = self.model.emb_model(utils.batch_nx_graphs(
                    valid_batch, anchors=anchors))eed(seed)
                if self.use_fp16:
                    embs = self._half_tensor(embs)
                for emb in embs:
                    yield emb
    def step(self):
        """Memory-efficient implementation of the MCTS step with FP16 support"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()ed)  # for multi-GPU
            
        ps = np.array([len(g) for g in self.dataset], dtype=np.float32)ms (may affect performance)
        ps /= np.sum(ps)rministic = True
        graph_dist = stats.rv_discrete(values=(np.arange(len(self.dataset)), ps))cudnn.benchmark = False

        print("Size", self.max_size)sh randomization
        print(len(self.visited_seed_nodes), "distinct seeds")tr(seed)
        
        for simulation_n in tqdm(range(self.n_trials // 
            (self.max_pattern_size+1-self.min_pattern_size))):
            
            if simulation_n % 100 == 0 and torch.cuda.is_available():    if not os.path.exists("plots/cluster"):
                torch.cuda.empty_cache()
            
            graph_idx = np.arange(len(self.dataset))[graph_dist.rvs()]er = argparse.ArgumentParser(description='Decoder arguments')
            graph = self.dataset[graph_idx] 
            
            seed_scores = []
            for _ in range(min(10, graph.number_of_nodes())):
                start_node = random.choice(list(graph.nodes))
                n_reachable = sum(1 for _ in self._stream_neighborhood(producibility seeds FIRST
                    graph, start_node, max_nodes=self.min_pattern_size))
                seed_scores.append((start_node, n_reachable))
            start_node = max(seed_scores, key=lambda x: x[1])[0]sing dataset {}".format(args.dataset))
            ormat(args.graph_type))
            neigh = [start_node]
            visited = {start_node}
            frontier = set()
            
            for next_node in self._stream_neighborhood(graph, start_node):
                if len(neigh) >= self.max_size:
                    break
                    (nx.Graph, nx.DiGraph)):
                cand_neigh = graph.subgraph(neigh + [next_node])
                if self.node_anchored:
                    for v in cand_neigh.nodes:    # Convert graph type if needed
                        cand_neigh.nodes[v]["anchor"] = 1 if v == neigh[0] else 0():
d graph to directed...")
                if cand_neigh.number_of_edges() > 0: = graph.to_directed()
                    try: args.graph_type == "undirected" and graph.is_directed():
                        cand_emb = next(self._batch_embeddings([cand_neigh]))ed...")
        ndirected()
                        score = 0
                        n_embs = 0
                        for emb_batch in self.embs:                print(f"Using NetworkX {graph_type} graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
                            if self.use_fp16:
                                emb_batch = self._half_tensor(emb_batch)dge direction information if available
                            pred = self.model.predict((
                                emb_batch.to(utils.get_device()), cand_emb))        if sample_edges:
                            if self.use_fp16: edge attributes:")
                                pred = pred.float()s in sample_edges:
                            score += torch.sum(pred).item()'direction', f"{u} -> {v}" if graph.is_directed() else f"{u} -- {v}")
                            n_embs += len(emb_batch)('type', 'unknown')
            
                        if n_embs > 0 and score/n_embs > 0.5:  
                            neigh.append(next_node)ata:
                            visited.add(next_node)fied type
                            frontier.update(n for n in graph.neighbors(next_node) 
                                if n not in visited)
                    except StopIteration:
                        pass        graph = nx.Graph()
                if len(neigh) >= self.min_pattern_size:
                    pattern = graph.subgraph(neigh).copy()
                    pattern_hash = utils.wl_hash(pattern,graph from dict format with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
                        node_anchored=self.node_anchored)
                    self.visit_counts[len(pattern)][pattern_hash] += 1rmat. Expected NetworkX graph or dict with 'nodes'/'edges' keys, got {type(data)}")
                    
            self.max_size += 1

class BeamSearchAgent(SearchAgent):
    """Beam Search implementation for subgraph pattern mining.ZYMES')
    
    Beam search maintains a fixed-size set of the most promising candidates at each step,
    providing a good balance between exploration quality and computational efficiency.Dataset(root='/tmp/cox2', name='COX2')
    """
        elif args.dataset == 'reddit-binary':
    def __init__(self, min_pattern_size, max_pattern_size, model, dataset,tmp/REDDIT-BINARY', name='REDDIT-BINARY')
        embs, node_anchored=False, analyze=False, model_type="order",
        out_batch_size=20, beam_width=5, batch_size=64):elif args.dataset == 'dblp':
        """Initialize the beam search agent.
        
        Args:f args.dataset == 'coil':
            min_pattern_size: Minimum size of patterns to find.    dataset = TUDataset(root='/tmp/coil', name='COIL-DEL')
            max_pattern_size: Maximum size of patterns to find.
            model: Trained subgraph matching model.
            dataset: DeepSNAP dataset to mine for patterns.
            embs: Embeddings of sampled node neighborhoods.e == "undirected" else nx.DiGraph()
            node_anchored: Whether to identify node-anchored patterns.with open("data/{}.txt".format(args.dataset), "r") as f:
            analyze: Whether to enable analysis visualization.or row in f:
            model_type: Type of subgraph matching model.
            out_batch_size: Number of patterns to output for each size.
            beam_width: Number of candidates to maintain at each step.
            batch_size: Size of batches for processing embeddings.
        """
        super().__init__(min_pattern_size, max_pattern_size, model, dataset,
            embs, node_anchored=node_anchored, analyze=analyze,
            model_type=model_type, out_batch_size=out_batch_size)
        self.beam_width = beam_width:
        self.batch_size = batch_size
        self.use_fp16 = torch.cuda.is_available()
     "mn-roads": "mn-roads.mtx",
    def _half_tensor(self, tensor):
        """Convert tensor to half precision if CUDA is available."""
        return tensor.half() if self.use_fp16 else tensore nx.DiGraph()
    n[args.dataset]), "r") as f:
    def init_search(self):
        """Initialize search data structures."""
        self.pattern_beams = {size: [] for size in range(            a, b = line.strip().split(" ")
            self.min_pattern_size, self.max_pattern_size + 1)}a), int(b))
        self.cand_patterns = defaultdict(list)
        self.pattern_counts = defaultdict(lambda: defaultdict(list))
        self.trials_completed = 0elif args.dataset.startswith('plant-'):
        self.current_size = self.min_pattern_sizetaset.split("-")[-1])
        self.analyze_embs = [] if self.analyze else None
    
    def is_search_done(self):
        """Check if search is complete."""
        return self.trials_completed >= self.n_trials
    
    def _compute_pattern_score(self, pattern, anchor=None):
        """Compute score for a pattern using the trained model."""        if pattern.number_of_edges() == 0:            return float('inf')  # Invalid pattern                    with torch.no_grad():            anchors = [anchor] if self.node_anchored and anchor else None            emb = self.model.emb_model(utils.batch_nx_graphs([pattern], anchors=anchors)).squeeze(0)                        if self.use_fp16:                emb = self._half_tensor(emb)                            score = 0            n_embs = 0                        for emb_batch in self.embs:                n_embs += len(emb_batch)                if self.use_fp16:                    emb_batch = self._half_tensor(emb_batch)                                    if self.model_type == "order":                    pred = self.model.predict((emb_batch.to(utils.get_device()), emb)).unsqueeze(1)                    if self.use_fp16:                        pred = pred.float()                    score -= torch.sum(torch.argmax(self.model.clf_model(pred), axis=1)).item()                elif self.model_type == "mlp":                    pred = self.model(                        emb_batch.to(utils.get_device()),                        emb.unsqueeze(0).expand(len(emb_batch), -1))                    if self.use_fp16:                        pred = pred.float()                    score += torch.sum(pred[:,0]).item()                        return score / max(1, n_embs)  # Normalize by number of embeddings        def _sample_seed_node(self):        """Sample a seed node from the dataset."""        # Weight sampling by graph size        ps = np.array([len(g) for g in self.dataset], dtype=np.float32)        ps /= np.sum(ps)        graph_dist = stats.rv_discrete(values=(np.arange(len(self.dataset)), ps))                # Sample a graph        graph_idx = graph_dist.rvs()        graph = self.dataset[graph_idx]                # Sample node with enough neighbors        candidates = []        for _ in range(min(10, graph.number_of_nodes())):            node = random.choice(list(graph.nodes))            subgraph = graph.subgraph(list(nx.ego_graph(graph, node, radius=2)))            if subgraph.number_of_nodes() >= self.min_pattern_size:                candidates.append((node, subgraph.number_of_nodes()))                if not candidates:            # Fallback to random node            return graph_idx, random.choice(list(graph.nodes))                # Choose node with largest 2-hop neighborhood        node = max(candidates, key=lambda x: x[1])[0]        return graph_idx, node        def _grow_patterns(self, beam):        """Grow patterns in the current beam by one node."""        new_candidates = []                for score, pattern, graph_idx, seed_node in beam:            graph = self.dataset[graph_idx]                        # Find nodes that can be added to the pattern            pattern_nodes = set(pattern.nodes)            frontier = set()            for node in pattern_nodes:                frontier.update(n for n in graph.neighbors(node) if n not in pattern_nodes)                        # Process frontier nodes in batches            for i in range(0, len(frontier), self.batch_size):                batch_nodes = list(frontier)[i:i+self.batch_size]                                for node in batch_nodes:                    # Create new pattern with added node                    new_pattern_nodes = list(pattern_nodes) + [node]                    new_pattern = graph.subgraph(new_pattern_nodes).copy()                                        # Skip if no new edges were added                    if new_pattern.number_of_edges() <= pattern.number_of_edges():                        continue                                            # Set anchor if needed                    if self.node_anchored:                        for v in new_pattern.nodes:                            new_pattern.nodes[v]["anchor"] = 1 if v == seed_node else 0                                        # Compute pattern score                    new_score = self._compute_pattern_score(new_pattern, anchor=seed_node)                                        # Add to candidates                    new_candidates.append((new_score, new_pattern, graph_idx, seed_node))                # Return top-k candidates        return sorted(new_candidates, key=lambda x: x[0])[:self.beam_width]        def step(self):        """Execute one step of beam search."""        if torch.cuda.is_available():            torch.cuda.empty_cache()                # Initialize beam if needed        if not self.pattern_beams[self.current_size]:            # Sample seed nodes and create initial patterns            initial_beam = []            num_seeds = min(self.beam_width * 2, self.n_trials - self.trials_completed)                        for _ in range(num_seeds):                graph_idx, seed_node = self._sample_seed_node()                graph = self.dataset[graph_idx]                                # Create pattern from seed node and its 1-hop neighbors                neighbors = list(graph.neighbors(seed_node))                if not neighbors:                    continue                                    # Start with seed node and its first neighbor                initial_nodes = [seed_node, neighbors[0]]                pattern = graph.subgraph(initial_nodes).copy()                                if pattern.number_of_edges() == 0:                    continue                                    # Set anchor if needed                if self.node_anchored:                    for v in pattern.nodes:                        pattern.nodes[v]["anchor"] = 1 if v == seed_node else 0                                # Grow pattern to minimum size                current_pattern = pattern                current_nodes = set(initial_nodes)                                while len(current_nodes) < self.min_pattern_size and neighbors:                    # Add next neighbor                    next_node = neighbors.pop(0)                    if next_node in current_nodes:                        continue                                            current_nodes.add(next_node)                    current_pattern = graph.subgraph(list(current_nodes)).copy()                                        # Set anchor if needed                    if self.node_anchored:                        for v in current_pattern.nodes:                            current_pattern.nodes[v]["anchor"] = 1 if v == seed_node else 0                                if len(current_pattern) < self.min_pattern_size:                    continue                                    # Compute pattern score                score = self._compute_pattern_score(current_pattern, anchor=seed_node)                initial_beam.append((score, current_pattern, graph_idx, seed_node))                        # Sort and keep top beam_width patterns            self.pattern_beams[self.current_size] = sorted(                initial_beam, key=lambda x: x[0])[:self.beam_width]            self.trials_completed += len(initial_beam)                # Grow patterns        current_beam = self.pattern_beams[self.current_size]        if current_beam and self.current_size < self.max_pattern_size:            next_beam = self._grow_patterns(current_beam)                        if next_beam:                self.pattern_beams[self.current_size + 1] = next_beam                # Record patterns from current beam        for score, pattern, graph_idx, seed_node in current_beam:            # Add to candidate patterns            self.cand_patterns[len(pattern)].append((score, pattern))                        # Track pattern counts by WL hash            pattern_hash = utils.wl_hash(pattern, node_anchored=self.node_anchored)            self.pattern_counts[len(pattern)][pattern_hash].append(pattern)                        # Save embedding for analysis if needed            if self.analyze:                with torch.no_grad():                    anchors = [seed_node] if self.node_anchored else None                    emb = self.model.emb_model(utils.batch_nx_graphs(                        [pattern], anchors=anchors)).squeeze(0)                    self.analyze_embs.append(emb.detach().cpu().numpy())                # Move to next pattern size or wrap around        self.current_size += 1        if self.current_size > self.max_pattern_size:            self.current_size = self.min_pattern_size        def finish_search(self):        """Finish search and return identified patterns."""        if self.analyze:            print("Saving analysis info in results/analyze.p")            with open("results/analyze.p", "wb") as f:                pickle.dump((self.cand_patterns, self.analyze_embs), f)                        # Create visualization            analysis_data = np.array(self.analyze_embs)            if len(analysis_data) > 0 and len(analysis_data[0]) >= 2:                xs, ys = analysis_data[:, 0], analysis_data[:, 1]                plt.scatter(xs, ys, color="red", label="motif")                plt.legend()                plt.savefig("plots/analyze.png")                plt.close()                # Collect results        cand_patterns_uniq = []        for pattern_size in range(self.min_pattern_size, self.max_pattern_size + 1):            pattern_counts = [(h, len(ps)) for h, ps in self.pattern_counts[pattern_size].items()]                        # Take the most frequent patterns            for wl_hash, count in sorted(pattern_counts, key=lambda x: x[1], reverse=True)[:self.out_batch_size]:                patterns = self.pattern_counts[pattern_size][wl_hash]                if patterns:                    cand_patterns_uniq.append(random.choice(patterns))                    print(f"- outputting {count} motifs of size {pattern_size}")                return cand_patterns_uniq