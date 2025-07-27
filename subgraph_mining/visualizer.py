import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import defaultdict, deque
import math

def get_anchor_centered_layout(G):
    """
    Create a directional layered layout with anchor nodes at the center
    and other nodes arranged in directions following their predecessors
    """
    # Find anchor nodes
    anchor_nodes = [n for n in G.nodes() if G.nodes[n].get('anchor', 0) == 1]
    
    if not anchor_nodes:
        # Fallback to spring layout if no anchors
        return nx.spring_layout(G, k=3.0, seed=42, iterations=150)
    
    pos = {}
    
    # Step 1: Position anchor nodes at center (possibly shifted down if needed)
    if len(anchor_nodes) == 1:
        # Will position anchor after calculating layer structure
        anchor_pos = np.array([0.0, 0.0])  # Temporary position
    else:
        # Multiple anchors in small circle at center
        anchor_radius = 0.5
        anchor_positions = {}
        for i, anchor in enumerate(anchor_nodes):
            angle = 2 * math.pi * i / len(anchor_nodes)
            anchor_positions[anchor] = np.array([
                anchor_radius * math.cos(angle),
                anchor_radius * math.sin(angle)
            ])
    
    # Step 2: Build parent-child relationships and calculate distances
    distances = {}
    parents = {}  # Track immediate parent for directional positioning
    visited = set()
    queue = deque()
    
    # Initialize with anchor nodes at distance 0
    for anchor in anchor_nodes:
        distances[anchor] = 0
        parents[anchor] = None
        queue.append((anchor, 0, None))  # (node, distance, parent)
        visited.add(anchor)
    
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
                parents[neighbor] = node
                queue.append((neighbor, dist + 1, node))
                visited.add(neighbor)
    
    # Step 3: Group nodes by distance (layers)
    layers = defaultdict(list)
    for node, dist in distances.items():
        layers[dist].append(node)
    
    # Step 4: Position nodes layer by layer following directional flow
    base_radius = 2.5
    radius_increment = 3.0
    
    # Position layer 1 (directly connected to anchors) in a circle
    if 1 in layers:
        layer_1_nodes = layers[1]
        num_layer1 = len(layer_1_nodes)
        
        # Sort by parent anchor and connection strength
        def layer1_priority(node):
            parent = parents[node]
            degree = G.degree(node) if not G.is_directed() else G.in_degree(node) + G.out_degree(node)
            return (-degree, str(node))  # Sort by degree, then name for consistency
        
        layer_1_nodes.sort(key=layer1_priority)
        
        for i, node in enumerate(layer_1_nodes):
            angle = 2 * math.pi * i / num_layer1
            pos[node] = np.array([
                base_radius * math.cos(angle),
                base_radius * math.sin(angle)
            ])
    
    # Step 5: Position subsequent layers directionally
    for layer_dist in range(2, max(layers.keys()) + 1 if layers else 2):
        if layer_dist not in layers:
            continue
        
        nodes_in_layer = layers[layer_dist]
        layer_radius = base_radius + (layer_dist - 1) * radius_increment
        
        for node in nodes_in_layer:
            parent = parents[node]
            if parent in pos:
                # Calculate direction from anchor to parent
                parent_pos = pos[parent]
                
                if len(anchor_nodes) == 1:
                    # Direction from anchor to parent
                    if np.linalg.norm(parent_pos) > 0:
                        direction = parent_pos / np.linalg.norm(parent_pos)
                    else:
                        # Fallback direction if parent is at origin
                        direction = np.array([1.0, 0.0])
                else:
                    # For multiple anchors, use parent position as reference
                    if np.linalg.norm(parent_pos) > 0:
                        direction = parent_pos / np.linalg.norm(parent_pos)
                    else:
                        direction = np.array([1.0, 0.0])
                
                # Position node in the same direction but further out
                base_position = direction * layer_radius
                
                # Add small perpendicular offset for multiple children of same parent
                siblings = [n for n in nodes_in_layer if parents[n] == parent]
                if len(siblings) > 1:
                    sibling_index = siblings.index(node)
                    total_siblings = len(siblings)
                    
                    # Create perpendicular vector
                    perp_direction = np.array([-direction[1], direction[0]])
                    
                    # Spread siblings around the base position
                    if total_siblings > 1:
                        offset_range = min(1.5, total_siblings * 0.4)
                        offset = (sibling_index - (total_siblings - 1) / 2) * (offset_range / max(1, total_siblings - 1))
                        base_position += perp_direction * offset
                
                pos[node] = base_position
    
    # Step 6: Position anchors (possibly shifted to accommodate upper branches)
    if len(anchor_nodes) == 1:
        anchor = anchor_nodes[0]
        # Check if we need to shift anchor down
        max_y = max([pos[node][1] for node in pos.values()] + [0])
        min_y = min([pos[node][1] for node in pos.values()] + [0])
        
        # If there are nodes with high y values, shift anchor down
        if max_y > 2.0:
            anchor_y = min_y - 1.0  # Position anchor below the lowest point
        else:
            anchor_y = 0.0
        
        pos[anchor] = np.array([0.0, anchor_y])
    else:
        # Multiple anchors - use pre-calculated positions
        for anchor in anchor_nodes:
            pos[anchor] = anchor_positions[anchor]
    
    # Step 7: Fine-tune positions to reduce overlaps
    pos = optimize_directional_layout(G, pos, anchor_nodes, layers, parents)
    
    return pos

def optimize_directional_layout(G, pos, anchor_nodes, layers, parents, max_iterations=50):
    """
    Fine-tune the directional layout to reduce overlaps while maintaining directional flow
    """
    # Convert to numpy arrays for easier manipulation
    nodes = list(pos.keys())
    positions = np.array([pos[node] for node in nodes])
    
    # Keep anchors fixed
    anchor_indices = [i for i, node in enumerate(nodes) if node in anchor_nodes]
    
    for iteration in range(max_iterations):
        moved = False
        
        # For each non-anchor node, try small adjustments
        for i, node in enumerate(nodes):
            if i in anchor_indices:
                continue  # Don't move anchor nodes
            
            current_pos = positions[i].copy()
            best_pos = current_pos.copy()
            best_score = calculate_directional_score(G, positions, nodes, i, parents)
            
            # Try small perturbations, but prefer to stay in the same general direction
            parent = parents.get(node)
            if parent and parent in nodes:
                parent_idx = nodes.index(parent)
                parent_pos = positions[parent_idx]
                
                # Preferred direction: away from parent
                if np.linalg.norm(current_pos - parent_pos) > 0:
                    preferred_dir = (current_pos - parent_pos) / np.linalg.norm(current_pos - parent_pos)
                else:
                    preferred_dir = np.array([1.0, 0.0])
                
                # Try perturbations mainly in preferred direction and perpendicular
                perp_dir = np.array([-preferred_dir[1], preferred_dir[0]])
                
                test_directions = [
                    preferred_dir * 0.3,  # Further from parent
                    -preferred_dir * 0.1,  # Slightly closer to parent
                    perp_dir * 0.2,  # Perpendicular directions
                    -perp_dir * 0.2,
                    (preferred_dir + perp_dir) * 0.2,  # Diagonal combinations
                    (preferred_dir - perp_dir) * 0.2,
                ]
            else:
                # No parent info, try general perturbations
                perturbation_size = 0.2
                test_directions = []
                for angle in [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4]:
                    test_directions.append(perturbation_size * np.array([math.cos(angle), math.sin(angle)]))
            
            for direction in test_directions:
                test_pos = current_pos + direction
                
                # Temporarily update position
                positions[i] = test_pos
                score = calculate_directional_score(G, positions, nodes, i, parents)
                
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

def calculate_directional_score(G, positions, nodes, node_idx, parents):
    """
    Calculate a score for the directional layout (lower is better)
    Considers node overlaps, edge lengths, and directional consistency
    """
    score = 0.0
    node = nodes[node_idx]
    node_pos = positions[node_idx]
    
    # Penalty for being too close to other nodes
    for i, other_node in enumerate(nodes):
        if i == node_idx:
            continue
        distance = np.linalg.norm(node_pos - positions[i])
        if distance < 1.2:  # Minimum desired distance
            score += (1.2 - distance) ** 2 * 15
    
    # Penalty for deviating from directional flow
    parent = parents.get(node)
    if parent and parent in nodes:
        parent_idx = nodes.index(parent)
        parent_pos = positions[parent_idx]
        
        # Check if we're moving in the right direction from parent
        current_dist = np.linalg.norm(node_pos - parent_pos)
        if current_dist < 1.5:  # Too close to parent
            score += (1.5 - current_dist) ** 2 * 10
    
    # Penalty for very long edges to connected nodes
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
            # Penalty for edges that are too long
            if edge_length > 8.0:
                score += (edge_length - 8.0) ** 2 * 2
    
    return score

def visualize_pattern_graph_ext(pattern, args, count_by_size):
    try:
        num_nodes = len(pattern)
        num_edges = pattern.number_of_edges()
        edge_density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        # Increased figure sizing for better spacing
        if num_nodes >= 14:
            base_size = max(14, min(24, num_nodes * 2.5))
            if edge_density > 0.3:
                figsize = (base_size * 1.4, base_size * 1.2)
            else:
                figsize = (base_size * 1.3, base_size * 1.1)
        else:
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
                if isinstance(value, float):
                    value = f"{value:.1f}" if abs(value) < 100 else f"{value:.0e}"
                label_parts.append(f"{key}: {value}")
            node_labels[n] = "\n".join(label_parts)
        
        # Use anchor-centered layout
        pos = get_anchor_centered_layout(pattern)
        
        print("pos:", pos)
        print("num_nodes:", num_nodes, "num_edges:", num_edges, "edge_density:", edge_density)
        
        # Color mapping
        unique_labels = sorted(set(pattern.nodes[n].get('label', 'unknown') for n in pattern.nodes()))
        label_color_map = {label: plt.cm.Set3(i) for i, label in enumerate(unique_labels)}
        unique_edge_types = sorted(set(data.get('type', 'default') for u, v, data in pattern.edges(data=True)))
        edge_color_map = {edge_type: plt.cm.tab20(i % 20) for i, edge_type in enumerate(unique_edge_types)}
        
        # Adaptive node sizing
        if num_nodes > 30:
            base_node_size = 3000
            anchor_node_size = base_node_size * 1.5
        elif num_nodes > 20 or edge_density > 0.5:
            base_node_size = 3500
            anchor_node_size = base_node_size * 1.5
        elif edge_density > 0.3:
            base_node_size = 5000
            anchor_node_size = base_node_size * 1.5
        else:
            base_node_size = 7000
            anchor_node_size = base_node_size * 1.5
        
        # Prepare node attributes
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
                node_sizes.append(anchor_node_size)
                shapes.append('s')
            else:
                colors.append(label_color_map[node_label])
                node_sizes.append(base_node_size)
                shapes.append('o')
        
        # Separate anchor and regular nodes for drawing
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
                connectionstyle = "arc3,rad=0.05"  # Very slight curvature for all edges
                
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
        
        # Continue with the rest of your original code for labels, legends, etc.
        # (The rest remains the same as your original implementation)
        
        # Adaptive font sizing
        max_attrs_per_node = max(len([k for k in pattern.nodes[n].keys() 
                                     if k not in ['id', 'label', 'anchor'] and pattern.nodes[n][k] is not None]) 
                                for n in pattern.nodes())
        if num_nodes > 30:
            font_size = max(6, min(8, 120 // (num_nodes + max_attrs_per_node * 3)))
        elif num_nodes > 20:
            font_size = max(7, min(9, 160 // (num_nodes + max_attrs_per_node * 4)))
        elif edge_density > 0.5:
            font_size = max(8, min(10, 200 // (num_nodes + max_attrs_per_node * 5)))
        else:
            font_size = max(12, min(14, 250 // (num_nodes + max_attrs_per_node * 2)))
        
        # Draw node labels with improved positioning
        for node, (x, y) in pos.items():
            label = node_labels[node]
            node_data = pattern.nodes[node]
            is_anchor = node_data.get('anchor', 0) == 1
            
            if num_nodes > 25 or edge_density > 0.5:
                pad = 0.15
            elif num_nodes > 15:
                pad = 0.25
            else:
                pad = 0.3
            
            bbox_props = dict(
                facecolor='lightcoral' if is_anchor else 'lightblue',
                edgecolor='darkred' if is_anchor else 'navy',
                alpha=0.95 if is_anchor else 0.85,
                boxstyle=f'round,pad={pad}',
                linewidth=2 if is_anchor else 1
            )
            
            plt.text(x, y, label, 
                    fontsize=font_size, 
                    fontweight='bold' if is_anchor else 'normal',
                    color='black',
                    ha='center', va='center',
                    bbox=bbox_props,
                    zorder=10)
        
        # Edge labels - now applied to all layers including second layer and beyond
        if num_nodes <= 30 and edge_density < 0.6 and num_edges < 50:  # More permissive for directional layout
            edge_labels = {}
            for u, v, data in pattern.edges(data=True):
                edge_type = (data.get('type') or 
                           data.get('label') or 
                           data.get('input_label') or
                           data.get('relation') or
                           data.get('edge_type'))
                if edge_type and len(str(edge_type)) <= 20:  # Slightly longer labels allowed
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
        
        # Create legends (keeping your original legend code)
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
                plt.tight_layout(rect=[0, 0, 0.85, 1])
            elif num_nodes >= 14:
                legend = plt.legend(
                    handles=legend_elements,
                    loc='center left',
                    bbox_to_anchor=(1.01, 0.5),
                    borderaxespad=0.,
                    framealpha=0.95,
                    title="Graph Elements",
                    fontsize=9
                )
                legend.get_title().set_fontsize(font_size + 1)
                plt.tight_layout(rect=[0, 0, 0.87, 1])
            else:
                legend = plt.legend(
                    handles=legend_elements,
                    loc='upper right',
                    bbox_to_anchor=(0.98, 0.98),
                    borderaxespad=0.,
                    framealpha=0.95,
                    title="Graph Elements",
                    fontsize=9
                )
                legend.get_title().set_fontsize(font_size + 1)
                plt.tight_layout()
        else:
            plt.tight_layout()
        
        # Generate filename (keeping your original filename generation)
        pattern_info = [f"{num_nodes}-{count_by_size[num_nodes]}"]
        node_types = sorted(set(pattern.nodes[n].get('label', '') for n in pattern.nodes() if pattern.nodes[n].get('label', '')))
        if node_types:
            pattern_info.append('nodes-' + '-'.join(node_types))
        edge_types = sorted(set(pattern.edges[e].get('type', '') for e in pattern.edges() if pattern.edges[e].get('type', '')))
        if edge_types:
            pattern_info.append('edges-' + '-'.join(edge_types))
        if has_anchors:
            pattern_info.append('anchored')
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
        import traceback
        traceback.print_exc()
        return False