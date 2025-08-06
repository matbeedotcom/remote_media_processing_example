"""
Basic custom pipelines for RemoteMedia Processing.

This file demonstrates how custom pipelines are automatically discovered
when placed in the pipelines/ directory.
"""

from remotemedia.core.pipeline import Pipeline

# Import our custom nodes (assuming they're discoverable)
# In practice, these would be imported from the nodes package
try:
    from nodes.timestamp_node import TimestampNode
    from nodes.math_processor_node import MathProcessorNode
    from nodes.data_aggregator_node import DataAggregatorNode
except ImportError:
    # Fallback for when running discovery
    TimestampNode = None
    MathProcessorNode = None
    DataAggregatorNode = None


def create_timestamped_math_pipeline() -> Pipeline:
    """
    Create a pipeline that performs math operations and adds timestamps.
    
    Flow: Input -> Math Processing -> Timestamp -> Output
    """
    if not (MathProcessorNode and TimestampNode):
        raise ImportError("Required nodes not available for pipeline")
        
    pipeline = Pipeline(name="timestamped_math")
    
    # Add math processor
    math_node = MathProcessorNode(
        operations=["square", "double"],
        handle_lists=True,
        name="math_processor"
    )
    pipeline.add_node(math_node)
    
    # Add timestamp node
    timestamp_node = TimestampNode(
        format="iso",
        include_metadata=True,
        name="timestamper"
    )
    pipeline.add_node(timestamp_node)
    
    # Connect nodes
    pipeline.connect("math_processor", "timestamper")
    
    return pipeline


def create_aggregation_pipeline() -> Pipeline:
    """
    Create a pipeline that aggregates data and adds timestamps.
    
    Flow: Input -> Aggregation -> Timestamp -> Output
    """
    if not (DataAggregatorNode and TimestampNode):
        raise ImportError("Required nodes not available for pipeline")
        
    pipeline = Pipeline(name="aggregation")
    
    # Add aggregator
    aggregator_node = DataAggregatorNode(
        window_size=3,
        aggregation_type="collect",
        name="aggregator"
    )
    pipeline.add_node(aggregator_node)
    
    # Add timestamp node
    timestamp_node = TimestampNode(
        format="readable",
        include_metadata=True,
        name="timestamper"
    )
    pipeline.add_node(timestamp_node)
    
    # Connect nodes
    pipeline.connect("aggregator", "timestamper")
    
    return pipeline


# Pipeline registry for automatic discovery
PIPELINE_REGISTRY = {
    "timestamped_math": {
        "factory": create_timestamped_math_pipeline,
        "description": "Mathematical operations with timestamps",
        "category": "math",
        "tags": ["math", "timestamp", "basic"],
        "version": "1.0.0",
        "author": "RemoteMedia Example"
    },
    "aggregation": {
        "factory": create_aggregation_pipeline,
        "description": "Data aggregation with timestamps",
        "category": "aggregation",
        "tags": ["aggregation", "timestamp", "collection"],
        "version": "1.0.0",
        "author": "RemoteMedia Example"
    }
}