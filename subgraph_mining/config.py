import argparse
from common.config_utils import validate_label_config_compatibility, check_backward_compatibility

def parse_decoder(parser):
    dec_parser = parser.add_argument_group()
    
    # Sampling parameters
    dec_parser.add_argument('--chunk_size', type=int, default=10000,
                        help='Chunk size for processing large graphs')
    dec_parser.add_argument('--sample_method', type=str,
        help='"tree" or "radial" sampling method')
    dec_parser.add_argument('--radius', type=int,
        help='radius of node neighborhoods')
    dec_parser.add_argument('--subgraph_sample_size', type=int,
        help='number of nodes to take from each neighborhood')
    dec_parser.add_argument('--use_whole_graphs', action="store_true",
        help="whether to cluster whole graphs or sampled node neighborhoods")
        
    # Pattern search parameters
    dec_parser.add_argument('--min_pattern_size', type=int,
        help='minimum size of patterns to find')
    dec_parser.add_argument('--max_pattern_size', type=int,
        help='maximum size of patterns to find')
    dec_parser.add_argument('--min_neighborhood_size', type=int,
        help='minimum neighborhood size to consider')
    dec_parser.add_argument('--max_neighborhood_size', type=int,
        help='maximum neighborhood size to consider')
    dec_parser.add_argument('--n_neighborhoods', type=int,
        help='number of neighborhoods to sample')
    
    # Search strategy parameters
    dec_parser.add_argument('--search_strategy', type=str,
        help='"greedy" or "mcts" search strategy')
    dec_parser.add_argument('--n_trials', type=int,
        help='number of search trials to run')
    dec_parser.add_argument('--out_batch_size', type=int,
        help='number of motifs to output per graph size')
    
    # Memory efficiency parameters
    dec_parser.add_argument('--memory_efficient', action='store_true',
        help='Use memory efficient search for large graphs')
    # Beam search parameter
    parser.add_argument('--beam_width', type=int, default=5,
                        help='Width of beam for beam search')
    # Output and analysis
    dec_parser.add_argument('--out_path', type=str,
        help='path to output candidate motifs')
    dec_parser.add_argument('--analyze', action="store_true",
        help='enable analysis mode')
    dec_parser.add_argument('--motif_dataset', type=str,
        help='Motif dataset to use')
    dec_parser.add_argument('--n_clusters', type=int,
        help='number of clusters for analysis')

    # Graph type selection
    dec_parser.add_argument('--graph_type', type=str,
        help='"directed" or "undirected" graph type')
    
    # Node label configuration arguments
    dec_parser.add_argument('--use_node_labels', action="store_true",
                        help='Enable node label features for content-aware pattern differentiation')
    dec_parser.add_argument('--node_label_key', type=str, default="label",
                        help='Node attribute key to use as label (default: "label")')
    dec_parser.add_argument('--max_label_vocab_size', type=int, default=1000,
                        help='Maximum label vocabulary size (default: 1000)')
    dec_parser.add_argument('--label_encoding_dim', type=int, default=None,
                        help='Dimension for label encoding (auto-detected if None)')
    
    # Set default values
    parser.set_defaults(
        # Dataset defaults
        dataset="enzymes",
        batch_size=1000,
        
        # Decoder defaults
        out_path="results/out-patterns.p",
        n_neighborhoods=2000,
        n_trials=1000,
        decode_thresh=0.5,
        radius=3,
        subgraph_sample_size=0,
        sample_method="tree",
        skip="learnable",
        graph_type="undirected",
        min_pattern_size=5,
        max_pattern_size=10,
        min_neighborhood_size=5,
        max_neighborhood_size=10,
        search_strategy="greedy",
        out_batch_size=10,
        node_anchored=True,
        memory_limit=1000000,
        use_node_labels=False,
        node_label_key="label",
        max_label_vocab_size=1000,
        label_encoding_dim=None
    )


def parse_and_validate_decoder(arg_str=None):
    """
    Parse decoder arguments and validate configuration.
    
    Args:
        arg_str: Optional argument string for testing
        
    Returns:
        Parsed and validated arguments
    """
    parser = argparse.ArgumentParser()
    parse_decoder(parser)
    args = parser.parse_args(arg_str)
    
    # Validate label configuration
    if not validate_label_config_compatibility(args):
        raise ValueError("Invalid label configuration")
    
    # Check backward compatibility
    if not check_backward_compatibility(args):
        raise ValueError("Configuration breaks backward compatibility")
    
    return args