import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import defaultdict, deque
import math
import traceback

def get_anchor_centered_layout(G):
    # Find the node with highest degree instead of anchor nodes
    degrees = dict(G.degree())
    highest_degree_node = max(degrees, key=degrees.get)
    center_nodes = [highest_degree_node]
    
    if not center_nodes:
        # Fallback to spring layout if no nodes
        return nx.spring_layout(G, k=3.0, seed=42, iterations=150)
    
    pos = {}
    
    # Position highest degree node at center
    pos[center_nodes[0]] = np.array([0.0, 0.0])
    
    distances = {}
    all_parents = {}  
    primary_parent = {} 
    visited = set()
    queue = deque()
    
    # Initialize with center node at distance 0
    for center_node in center_nodes:
        distances[center_node] = 0
        all_parents[center_node] = []
        primary_parent[center_node] = None
        queue.append((center_node, 0, None))  # (node, distance, parent)
        visited.add(center_node)
    
    # BFS to find shortest distance and parent relationships
    while queue:
        node, dist, parent = queue.popleft()
        
        # Get successors (outgoing edges) for directed positioning
        if G.is_directed():
            neighbors = list(G.successors(node))
        else:
            neighbors = list(G.neighbors(node))
        
        for neighbor in neighbors:
            if neighbor not in visited:
                distances[neighbor] = dist + 1
                all_parents[neighbor] = [node]
                primary_parent[neighbor] = node
                queue.append((neighbor, dist + 1, node))
                visited.add(neighbor)
            elif neighbor in distances and distances[neighbor] == dist + 1:
                # Same distance - this is an additional parent at the same layer
                if neighbor not in all_parents:
                    all_parents[neighbor] = []
                all_parents[neighbor].append(node)
    
    # Group nodes by distance (layers)
    layers = defaultdict(list)
    for node, dist in distances.items():
        layers[dist].append(node)
    
    # Position Layer 1 in a perfect circle around center node
    num_nodes = len(G)
    if num_nodes <= 8:
        base_radius = 6.0    # Increased for small graphs
        radius_increment = 5.5
    else:
        base_radius = 3.0
        radius_increment = 3.5
    if 1 in layers:
        layer_1_nodes = layers[1]
        num_layer1 = len(layer_1_nodes)
        
        # Sort layer 1 nodes for consistent arrangement
        def layer1_priority(node):
            degree = G.degree(node) if not G.is_directed() else G.in_degree(node) + G.out_degree(node)
            return (-degree, str(node))  # Sort by degree, then name for consistency
        
        layer_1_nodes.sort(key=layer1_priority)
        
        # Position in perfect circle
        for i, node in enumerate(layer_1_nodes):
            angle = 2 * math.pi * i / num_layer1
            pos[node] = np.array([
                base_radius * math.cos(angle),
                base_radius * math.sin(angle)
            ])
    
    for layer_dist in range(2, max(layers.keys()) + 1 if layers else 2):
        if layer_dist not in layers:
            continue
        
        nodes_in_layer = layers[layer_dist]
        
        for node in nodes_in_layer:
            parents_list = all_parents.get(node, [])
            
            if len(parents_list) == 1:
                # Single parent - use original radial positioning
                parent = parents_list[0]
                if parent in pos:
                    parent_pos = pos[parent]
                    
                    # Calculate the radial direction from center through parent
                    if np.linalg.norm(parent_pos) > 0:
                        radial_direction = parent_pos / np.linalg.norm(parent_pos)
                    else:
                        radial_direction = np.array([1.0, 0.0])
                    
                    # Calculate distance for this layer
                    layer_distance = base_radius + (layer_dist - 1) * radius_increment
                    base_position = radial_direction * layer_distance
                    
                    # Handle siblings
                    siblings = [n for n in nodes_in_layer if len(all_parents.get(n, [])) == 1 and all_parents.get(n, [])[0] == parent]
                    if len(siblings) > 1:
                        sibling_index = siblings.index(node)
                        total_siblings = len(siblings)
                        perp_direction = np.array([-radial_direction[1], radial_direction[0]])
                        if total_siblings > 1:
                            spread_factor = min(2.0, layer_distance * 0.3)
                            offset = (sibling_index - (total_siblings - 1) / 2) * (spread_factor / max(1, total_siblings - 1))
                            base_position += perp_direction * offset
                    
                    pos[node] = base_position
                    
            elif len(parents_list) > 1:
                # Multiple parents - position at centroid of parents, then move outward
                parent_positions = []
                for parent in parents_list:
                    if parent in pos:
                        parent_positions.append(pos[parent])
                
                if parent_positions:
                    # Calculate centroid of parent positions
                    centroid = np.mean(parent_positions, axis=0)
                    
                    # Direction from center through centroid
                    if np.linalg.norm(centroid) > 0:
                        direction = centroid / np.linalg.norm(centroid)
                    else:
                        direction = np.array([1.0, 0.0])
                    
                    # Position at appropriate distance
                    layer_distance = base_radius + (layer_dist - 1) * radius_increment
                    
                    # For diamond patterns, place slightly closer to maintain visual connection
                    if len(parents_list) == 2:
                        # Check if parents are roughly opposite each other (diamond pattern)
                        if len(parent_positions) == 2:
                            p1, p2 = parent_positions
                            # Calculate angle between parents from center
                            if np.linalg.norm(p1) > 0 and np.linalg.norm(p2) > 0:
                                dir1 = p1 / np.linalg.norm(p1)
                                dir2 = p2 / np.linalg.norm(p2)
                                dot_product = np.clip(np.dot(dir1, dir2), -1.0, 1.0)
                                angle_between = math.acos(dot_product)
                                
                                # If parents are far apart (> 90 degrees), bring node closer
                                if angle_between > math.pi / 2:
                                    layer_distance *= 0.8  # Bring closer for diamond pattern
                    
                    pos[node] = direction * layer_distance
    
    pos = optimize_radial_layout(G, pos, center_nodes, layers, primary_parent)
    
    return pos

def optimize_radial_layout(G, pos, center_nodes, layers, parents, max_iterations=50):
    # Convert to numpy arrays for easier manipulation
    nodes = list(pos.keys())
    positions = np.array([pos[node] for node in nodes])
    
    # Keep center nodes fixed at center
    center_indices = [i for i, node in enumerate(nodes) if node in center_nodes]
    
    for iteration in range(max_iterations):
        moved = False
        
        # For each non-center node, try small adjustments that preserve radial structure
        for i, node in enumerate(nodes):
            if i in center_indices:
                continue  # Don't move center nodes
            
            current_pos = positions[i].copy()
            best_pos = current_pos.copy()
            best_score = calculate_radial_score(G, positions, nodes, i, parents, center_nodes)
            
            parent = parents.get(node)
            if parent and parent in nodes:
                parent_idx = nodes.index(parent)
                parent_pos = positions[parent_idx]
                
                # For radial layout, allow movement mainly perpendicular to radial direction
                if np.linalg.norm(current_pos) > 0:
                    radial_dir = current_pos / np.linalg.norm(current_pos)
                    perp_dir = np.array([-radial_dir[1], radial_dir[0]])
                    
                    # Try small adjustments: radial (in/out) and perpendicular (around circle)
                    test_directions = [
                        perp_dir * 0.3,      
                        -perp_dir * 0.3,     
                        radial_dir * 0.2,    
                        -radial_dir * 0.1,   
                        (perp_dir + radial_dir) * 0.2,  
                        (perp_dir - radial_dir) * 0.2,
                    ]
                else:
                    # Fallback for nodes at origin
                    test_directions = []
                    for angle in [0, math.pi/3, 2*math.pi/3, math.pi, 4*math.pi/3, 5*math.pi/3]:
                        test_directions.append(0.2 * np.array([math.cos(angle), math.sin(angle)]))
            else:
                # No parent info, try small general perturbations
                test_directions = []
                for angle in [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4]:
                    test_directions.append(0.2 * np.array([math.cos(angle), math.sin(angle)]))
            
            for direction in test_directions:
                test_pos = current_pos + direction
                
                # Temporarily update position
                positions[i] = test_pos
                score = calculate_radial_score(G, positions, nodes, i, parents, center_nodes)
                
                if score < best_score:
                    best_score = score
                    best_pos = test_pos.copy()
                    moved = True
            
            # Apply best position
            positions[i] = best_pos
        
        if not moved:
            break
    
    # Convert back to position dictionary
    optimized_pos = {}
    for i, node in enumerate(nodes):
        optimized_pos[node] = positions[i]
    
    return optimized_pos

def calculate_radial_score(G, positions, nodes, node_idx, parents, center_nodes):
    """
    Calculate a score for the radial layout (lower is better)
    Emphasizes maintaining radial structure while avoiding overlaps
    """
    score = 0.0
    node = nodes[node_idx]
    node_pos = positions[node_idx]
    
    # Penalty for being too close to other nodes
    for i, other_node in enumerate(nodes):
        if i == node_idx:
            continue
        distance = np.linalg.norm(node_pos - positions[i])
        if distance < 1.0:  # Minimum desired distance
            score += (1.0 - distance) ** 2 * 20
    
    # Penalty for deviating too much from radial structure
    parent = parents.get(node)
    if parent and parent in nodes:
        parent_idx = nodes.index(parent)
        parent_pos = positions[parent_idx]
        
        # Check if node is roughly in line with parent (from center)
        if np.linalg.norm(parent_pos) > 0.1:  # Parent not at center
            parent_direction = parent_pos / np.linalg.norm(parent_pos)
            if np.linalg.norm(node_pos) > 0.1:  # Node not at center
                node_direction = node_pos / np.linalg.norm(node_pos)
                
                # Calculate angle deviation from parent's radial direction
                dot_product = np.dot(parent_direction, node_direction)
                dot_product = np.clip(dot_product, -1.0, 1.0)  # Ensure valid range
                angle_deviation = math.acos(dot_product)
                
                # Penalty for large angular deviation (allows some spread for siblings)
                max_allowed_angle = math.pi / 6  # 30 degrees
                if angle_deviation > max_allowed_angle:
                    score += (angle_deviation - max_allowed_angle) ** 2 * 5
        
        # Ensure child is farther from center than parent
        parent_dist = np.linalg.norm(parent_pos)
        node_dist = np.linalg.norm(node_pos)
        if node_dist < parent_dist + 0.5:  # Should be at least 0.5 units farther
            score += (parent_dist + 0.5 - node_dist) ** 2 * 10
    
    # Penalty for very long edges
    neighbors = set()
    if G.is_directed():
        neighbors.update(G.successors(node))
        neighbors.update(G.predecessors(node))
    else:
        neighbors.update(G.neighbors(node))
    
    for neighbor in neighbors:
        if neighbor in nodes:
            neighbor_idx = nodes.index(neighbor)
            edge_length = np.linalg.norm(node_pos - positions[neighbor_idx])
            if edge_length > 8.0:
                score += (edge_length - 8.0) ** 2 * 2
    
    return score

def visualize_pattern_graph_ext(pattern, args, count_by_size):
    try:
        num_nodes = len(pattern)
        num_edges = pattern.number_of_edges()
        edge_density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        base_size = max(12, min(20, num_nodes * 2))
        if edge_density > 0.3:
            figsize = (base_size * 1.2, base_size)
        else:
            figsize = (base_size, base_size * 0.8)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Node labels: all attributes, smart truncation
        node_labels = {}
        for n in pattern.nodes():
            node_data = pattern.nodes[n]
            label_parts = []
            for key, value in node_data.items():
                if key != "anchor":
                    if isinstance(value, float):
                        value = f"{value:.1f}" if abs(value) < 100 else f"{value:.0e}"
                    label_parts.append(f"{key}: {value}")
            node_labels[n] = "\n".join(label_parts)
        
        # Use modified layout that centers highest degree node
        pos_init = get_anchor_centered_layout(pattern)
        pos = nx.spring_layout(pattern, k=1.5, iterations=100, seed=42, pos=pos_init)
        
        print("pos:", pos)
        print("num_nodes:", num_nodes, "num_edges:", num_edges, "edge_density:", edge_density)
        
        # Color mapping
        unique_labels = sorted(set(pattern.nodes[n].get('label', 'unknown') for n in pattern.nodes()))
        label_color_map = {label: plt.cm.Set3(i) for i, label in enumerate(unique_labels)}
        unique_edge_types = sorted(set(data.get('type', 'default') for u, v, data in pattern.edges(data=True)))
        edge_color_map = {edge_type: plt.cm.tab20(i % 20) for i, edge_type in enumerate(unique_edge_types)}

        base_node_size = 9000  # Large for small graphs
        center_node_size = base_node_size * 1.2

        colors = []
        node_sizes = []
        shapes = []
        node_list = list(pattern.nodes())
        
        for i, node in enumerate(node_list):
            node_data = pattern.nodes[node]
            node_label = node_data.get('label', 'unknown')
            is_anchor = node_data.get('anchor', 0) == 1 

            if is_anchor:
                colors.append('red')
                node_sizes.append(base_node_size)
                shapes.append('s')
            else:
                colors.append(label_color_map[node_label])
                node_sizes.append(base_node_size)
                shapes.append('o')
        
        # Separate center, anchor and regular nodes for drawing
        anchor_nodes = []
        regular_nodes = []
        anchor_colors = []
        regular_colors = []
        anchor_sizes = []
        regular_sizes = []
        
        for i, node in enumerate(node_list):
            if shapes[i] == 's':
                anchor_nodes.append(node)
                anchor_colors.append(colors[i])
                anchor_sizes.append(node_sizes[i])
            else:
                regular_nodes.append(node)
                regular_colors.append(colors[i])
                regular_sizes.append(node_sizes[i])
        
        # Draw nodes
        if regular_nodes:
            nx.draw_networkx_nodes(pattern, pos, 
                    nodelist=regular_nodes,
                    node_color=regular_colors, 
                    node_size=regular_sizes, 
                    node_shape='o',
                    edgecolors='black', 
                    linewidths=2,
                    alpha=0.8)
        
        if anchor_nodes:
            nx.draw_networkx_nodes(pattern, pos, 
                    nodelist=anchor_nodes,
                    node_color=anchor_colors, 
                    node_size=anchor_sizes, 
                    node_shape='s',
                    edgecolors='darkred', 
                    linewidths=3,
                    alpha=0.9)
        
        # Adaptive edge styling
        if num_nodes > 30:
            edge_width = 1.0
            edge_alpha = 0.4
        elif num_nodes > 20 or edge_density > 0.5:
            edge_width = 1.5
            edge_alpha = 0.6
        elif edge_density > 0.3:
            edge_width = 2.0
            edge_alpha = 0.7
        else:
            edge_width = 2.5
            edge_alpha = 0.8
        
        # Draw edges with minimal curvature for cleaner directional layout
        if pattern.is_directed():
            # Simplified arrow sizing for directional layout
            if num_nodes >= 30:
                arrow_size = 15
            elif num_nodes >= 20:
                arrow_size = 20
            else:
                arrow_size = 25
            
            # Draw edges with minimal curvature to maintain directional structure
            for u, v, data in pattern.edges(data=True):
                edge_type = data.get('type', 'default')
                edge_color = edge_color_map[edge_type]
                
                # Use straight edges for most connections in directional layout
                connectionstyle = "arc3,rad=0.05" 
                
                nx.draw_networkx_edges(
                    pattern, pos,
                    edgelist=[(u, v)],
                    width=edge_width,
                    edge_color=[edge_color],
                    alpha=edge_alpha,
                    arrows=True,
                    arrowsize=arrow_size,
                    arrowstyle='-|>',
                    connectionstyle=connectionstyle,
                    node_size=max(node_sizes) * 1.2,
                    min_source_margin=15,
                    min_target_margin=15
                )
        else:
            # Undirected edges
            for u, v, data in pattern.edges(data=True):
                edge_type = data.get('type', 'default')
                edge_color = edge_color_map[edge_type]
                nx.draw_networkx_edges(
                    pattern, pos,
                    edgelist=[(u, v)],
                    width=edge_width,
                    edge_color=[edge_color],
                    alpha=edge_alpha,
                    arrows=False
                )
        
        # smaller fonts for smaller graphs to prevent overlap
        max_attrs_per_node = max(len([k for k in pattern.nodes[n].keys() 
                                    if k not in ['id', 'label', 'anchor'] and pattern.nodes[n][k] is not None]) 
                                for n in pattern.nodes())
        font_size = max(8, min(11, 80 // (max_attrs_per_node + 1)))

        for node, (x, y) in pos.items():
            label = node_labels[node]
            node_data = pattern.nodes[node]
            is_anchor = node_data.get('anchor', 0) == 1
            pad = 0.3 
        
            if is_anchor:
                bbox_props = dict(
                    facecolor='lightcoral',
                    edgecolor='darkred',
                    alpha=0.95,
                    boxstyle=f'round,pad={pad}',
                    linewidth=2
                )
            else:
                bbox_props = dict(
                    facecolor='lightblue',
                    edgecolor='navy',
                    alpha=0.85,
                    boxstyle=f'round,pad={pad}',
                    linewidth=1
                )
            
            plt.text(x, y, label, 
                    fontsize=font_size, 
                    fontweight='normal',
                    color='black',
                    ha='center', va='center',
                    bbox=bbox_props,
                    zorder=10)
        
        # Edge labels 
        edge_labels = {}
        for u, v, data in pattern.edges(data=True):
            edge_type = (data.get('type') or 
                        data.get('label') or 
                        data.get('input_label') or
                        data.get('relation') or
                        data.get('edge_type'))
            if edge_type and len(str(edge_type)) <= 20: 
                edge_labels[(u, v)] = str(edge_type)
        
        if edge_labels:
            edge_font_size = max(6, font_size - 2)
            nx.draw_networkx_edge_labels(pattern, pos, 
                        edge_labels=edge_labels, 
                        font_size=edge_font_size, 
                        font_color='darkblue',
                        bbox=dict(facecolor='white', edgecolor='lightgray', 
                                alpha=0.8, boxstyle='round,pad=0.1'))
        
        # Create comprehensive title
        graph_type = "Directed" if pattern.is_directed() else "Undirected"
        has_anchors = any(pattern.nodes[n].get('anchor', 0) == 1 for n in pattern.nodes())
        anchor_info = " (Red squares = anchor nodes)" if has_anchors else ""
        total_node_attrs = sum(len([k for k in pattern.nodes[n].keys() 
                                  if k not in ['id', 'label', 'anchor'] and pattern.nodes[n][k] is not None]) 
                             for n in pattern.nodes())
        attr_info = f", {total_node_attrs} total attrs" if total_node_attrs > 0 else ""
        density_info = f"Density: {edge_density:.2f}"
        if edge_density > 0.5:
            density_info += " (Very Dense)"
        elif edge_density > 0.3:
            density_info += " (Dense)"
        else:
            density_info += " (Sparse)"
        
        title = f"{graph_type} Pattern Graph{anchor_info}\n"
        title += f"({num_nodes} nodes, {num_edges} edges{attr_info}, {density_info})"
        plt.title(title, fontsize=max(12, min(16, 20 - num_nodes//10)), fontweight='bold')
        plt.axis('off')
        
        # Create legends
        legend_elements = []
        
        if len(unique_labels) > 1:
            for label, color in label_color_map.items():
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=color, markersize=10, 
                              label=f'Node: {label}', markeredgecolor='black')
                )
        
        if has_anchors:
            legend_elements.append(
                plt.Line2D([0], [0], marker='s', color='w', 
                          markerfacecolor='red', markersize=12, 
                          label='Anchor Node', markeredgecolor='darkred', linewidth=2)
            )
        
        if len(unique_edge_types) > 1:
            for edge_type, color in edge_color_map.items():
                if edge_type != 'default':
                    legend_elements.append(
                        plt.Line2D([0], [0], color=color, linewidth=3, 
                                  label=f'Edge: {edge_type}')
                    )
        
        # Position legend
        if legend_elements:
            if num_nodes >= 25:
                legend = plt.legend(
                    handles=legend_elements,
                    loc='center left',
                    bbox_to_anchor=(1.02, 0.5),
                    borderaxespad=0.,
                    framealpha=0.95,
                    title="Graph Elements",
                    fontsize=9)
                legend.get_title().set_fontsize(font_size + 1)
                plt.tight_layout(rect=[0, 0, 0.87, 1])
            else:
                legend = plt.legend(
                    handles=legend_elements,
                    loc='upper right',
                    bbox_to_anchor=(0.02, 0.98),
                    borderaxespad=0.,
                    framealpha=0.95,
                    title="Graph Elements",
                    fontsize=9
                )
                legend.get_title().set_fontsize(font_size + 1)
                plt.tight_layout()
        else:
            plt.tight_layout()
        
        # Generate filename
        pattern_info = [f"{num_nodes}-{count_by_size[num_nodes]}"]
        node_types = sorted(set(pattern.nodes[n].get('label', '') for n in pattern.nodes() if pattern.nodes[n].get('label', '')))
        if node_types:
            pattern_info.append('nodes-' + '-'.join(node_types))
        edge_types = sorted(set(pattern.edges[e].get('type', '') for e in pattern.edges() if pattern.edges[e].get('type', '')))
        if edge_types:
            pattern_info.append('edges-' + '-'.join(edge_types))
        if has_anchors:
            pattern_info.append('anchored')
        pattern_info.append('centered-highest-degree')  # New indicator
        if total_node_attrs > 0:
            pattern_info.append(f'{total_node_attrs}attrs')
        if edge_density > 0.5:
            pattern_info.append('very-dense')
        elif edge_density > 0.3:
            pattern_info.append('dense')
        else:
            pattern_info.append('sparse')
        
        graph_type_short = "dir" if pattern.is_directed() else "undir"
        filename = f"{graph_type_short}_{('_'.join(pattern_info))}"
        
        # Save plots
        plt.savefig(f"plots/cluster/{filename}.png", bbox_inches='tight', dpi=300)
        plt.savefig(f"plots/cluster/{filename}.pdf", bbox_inches='tight')
        plt.close()
        
        print(f"Successfully saved plot for pattern with {len(pattern)} nodes")
        return True
        
    except Exception as e:
        print(f"Error visualizing pattern with {len(pattern)} nodes: {e}")
        traceback.print_exc()
        return False