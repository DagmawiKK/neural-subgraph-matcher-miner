"""
Configuration utilities for node label support and validation.
"""
import warnings
from typing import Dict, Any, Optional


class LabelConfig:
    """Configuration class for node label settings."""
    
    def __init__(self, use_node_labels: bool = False, 
                 node_label_key: str = "label",
                 max_label_vocab_size: int = 1000,
                 label_encoding_dim: Optional[int] = None):
        self.use_node_labels = use_node_labels
        self.node_label_key = node_label_key
        self.max_label_vocab_size = max_label_vocab_size
        self.label_encoding_dim = label_encoding_dim
    
    @classmethod
    def from_args(cls, args):
        """Create LabelConfig from parsed arguments."""
        return cls(
            use_node_labels=getattr(args, 'use_node_labels', False),
            node_label_key=getattr(args, 'node_label_key', "label"),
            max_label_vocab_size=getattr(args, 'max_label_vocab_size', 1000),
            label_encoding_dim=getattr(args, 'label_encoding_dim', None)
        )
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        valid = True
        
        if self.use_node_labels:
            if not isinstance(self.node_label_key, str) or not self.node_label_key.strip():
                warnings.warn("node_label_key must be a non-empty string. Using default 'label'.")
                self.node_label_key = "label"
                valid = False
            
            if self.max_label_vocab_size <= 0:
                warnings.warn("max_label_vocab_size must be positive. Using default 1000.")
                self.max_label_vocab_size = 1000
                valid = False
            
            if self.label_encoding_dim is not None and self.label_encoding_dim <= 0:
                warnings.warn("label_encoding_dim must be positive or None. Setting to None for auto-detection.")
                self.label_encoding_dim = None
                valid = False
        
        return valid
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'use_node_labels': self.use_node_labels,
            'node_label_key': self.node_label_key,
            'max_label_vocab_size': self.max_label_vocab_size,
            'label_encoding_dim': self.label_encoding_dim
        }


def validate_label_config_compatibility(args) -> bool:
    """
    Validate that label configuration is compatible with other settings.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        bool: True if configuration is valid and compatible
    """
    label_config = LabelConfig.from_args(args)
    
    # Validate the label configuration itself
    if not label_config.validate():
        return False
    
    # Check compatibility with existing features
    if label_config.use_node_labels:
        # Ensure node_anchored mode is compatible
        if hasattr(args, 'node_anchored') and not args.node_anchored:
            warnings.warn(
                "Node labels work best with node_anchored=True. "
                "Consider enabling node anchoring for better pattern differentiation."
            )
        
        # Check dataset compatibility
        if hasattr(args, 'dataset') and args.dataset == 'syn':
            warnings.warn(
                "Synthetic datasets may not have meaningful node labels. "
                "Ensure your synthetic data generation includes label assignment."
            )
    
    return True


def propagate_label_config(args, target_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Propagate label configuration from args to a target dictionary.
    
    Args:
        args: Parsed command line arguments
        target_dict: Dictionary to update with label configuration
        
    Returns:
        Dict[str, Any]: Updated dictionary with label configuration
    """
    label_config = LabelConfig.from_args(args)
    target_dict.update(label_config.to_dict())
    return target_dict


def check_backward_compatibility(args) -> bool:
    """
    Check that new label configuration maintains backward compatibility.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        bool: True if backward compatible
    """
    # When labels are disabled, behavior should be identical to original system
    if not getattr(args, 'use_node_labels', False):
        return True
    
    # Check that required attributes exist for label processing
    required_attrs = ['node_label_key', 'max_label_vocab_size']
    for attr in required_attrs:
        if not hasattr(args, attr):
            warnings.warn(f"Missing required attribute {attr} for label processing")
            return False
    
    return True