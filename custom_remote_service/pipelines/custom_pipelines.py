"""
Custom Pipelines for RemoteMedia Processing Example

This file demonstrates how to create custom pipelines that combine
both built-in and custom nodes.
"""

from remotemedia.core.pipeline import Pipeline
from remotemedia.nodes import AudioTransform, AudioBuffer
from custom_nodes import (
    TimestampNode, 
    DataAggregatorNode, 
    TextProcessorNode,
    MathProcessorNode,
    CustomValidatorNode,
    SimpleStreamingNode
)

# Pipeline metadata constants
PIPELINE_VERSION = "1.0.0"
PIPELINE_AUTHOR = "RemoteMedia Example"


def create_timestamped_processing_pipeline() -> Pipeline:
    """
    Create a pipeline that processes data and adds timestamps.
    
    Flow: Input -> Timestamp -> Validation -> Output
    """
    pipeline = Pipeline(name="timestamped_processing")
    
    # Add timestamp node
    timestamp_node = TimestampNode(
        format="iso",
        include_metadata=True,
        name="timestamper"
    )
    pipeline.add_node(timestamp_node)
    
    # Add validation node
    validator = CustomValidatorNode(
        rules={
            "required_fields": ["data", "timestamp"],
            "allowed_types": ["dict"]
        },
        strict_mode=False,
        name="validator"
    )
    pipeline.add_node(validator)
    
    # Connect nodes
    pipeline.connect("timestamper", "validator")
    
    return pipeline


def create_text_analysis_pipeline() -> Pipeline:
    """
    Create a pipeline for text analysis and aggregation.
    
    Flow: Input -> Text Processor -> Aggregator -> Timestamp -> Output
    """
    pipeline = Pipeline(name="text_analysis")
    
    # Add text processor
    text_processor = TextProcessorNode(
        operations=["lowercase", "word_count", "char_count"],
        extract_field="text",
        name="text_analyzer"
    )
    pipeline.add_node(text_processor)
    
    # Add aggregator
    aggregator = DataAggregatorNode(
        window_size=3,
        aggregation_type="collect",
        name="text_aggregator"
    )
    pipeline.add_node(aggregator)
    
    # Add timestamp
    timestamp_node = TimestampNode(
        format="readable",
        include_metadata=True,
        name="final_timestamper"
    )
    pipeline.add_node(timestamp_node)
    
    # Connect nodes
    pipeline.connect("text_analyzer", "text_aggregator")
    pipeline.connect("text_aggregator", "final_timestamper")
    
    return pipeline


def create_math_processing_pipeline() -> Pipeline:
    """
    Create a pipeline for mathematical data processing.
    
    Flow: Input -> Math Processor -> Aggregator -> Validator -> Output
    """
    pipeline = Pipeline(name="math_processing")
    
    # Add math processor
    math_processor = MathProcessorNode(
        operations=["square", "sqrt", "double"],
        handle_lists=True,
        name="math_calculator"
    )
    pipeline.add_node(math_processor)
    
    # Add aggregator for collecting results
    aggregator = DataAggregatorNode(
        window_size=2,
        aggregation_type="collect",
        name="math_aggregator"
    )
    pipeline.add_node(aggregator)
    
    # Add validator to ensure results are valid
    validator = CustomValidatorNode(
        rules={
            "required_fields": ["original", "processed"],
            "allowed_types": ["dict"]
        },
        strict_mode=False,
        name="math_validator"
    )
    pipeline.add_node(validator)
    
    # Connect nodes
    pipeline.connect("math_calculator", "math_aggregator")
    pipeline.connect("math_aggregator", "math_validator")
    
    return pipeline


def create_streaming_aggregation_pipeline() -> Pipeline:
    """
    Create a streaming pipeline that processes data in chunks.
    
    Flow: Input Stream -> Streaming Node -> Timestamp -> Output Stream
    """
    pipeline = Pipeline(name="streaming_aggregation")
    
    # Add streaming processor
    streaming_node = SimpleStreamingNode(
        chunk_size=5,
        delay=0.05,  # Small delay for demonstration
        name="stream_processor"
    )
    pipeline.add_node(streaming_node)
    
    # Add timestamp to each chunk
    timestamp_node = TimestampNode(
        format="unix",
        include_metadata=True,
        name="chunk_timestamper"
    )
    pipeline.add_node(timestamp_node)
    
    # Connect nodes
    pipeline.connect("stream_processor", "chunk_timestamper")
    
    return pipeline


def create_audio_with_metadata_pipeline() -> Pipeline:
    """
    Create a pipeline that processes audio and adds metadata.
    
    Combines built-in audio nodes with custom metadata nodes.
    Flow: Audio Input -> Audio Transform -> Audio Buffer -> Timestamp -> Validation
    """
    pipeline = Pipeline(name="audio_with_metadata")
    
    # Add audio preprocessing
    audio_transform = AudioTransform(
        sample_rate=16000,
        channels=1,
        format="float32",
        name="audio_preprocessor"
    )
    pipeline.add_node(audio_transform)
    
    # Add audio buffering
    audio_buffer = AudioBuffer(
        buffer_size=1024,
        overlap=0.5,
        name="audio_buffer"
    )
    pipeline.add_node(audio_buffer)
    
    # Add timestamp for metadata
    timestamp_node = TimestampNode(
        format="iso",
        include_metadata=True,
        name="audio_timestamper"
    )
    pipeline.add_node(timestamp_node)
    
    # Add validation
    validator = CustomValidatorNode(
        rules={
            "required_fields": ["data", "timestamp"],
            "allowed_types": ["dict"]
        },
        strict_mode=False,
        name="audio_validator"
    )
    pipeline.add_node(validator)
    
    # Connect nodes
    pipeline.connect("audio_preprocessor", "audio_buffer")
    pipeline.connect("audio_buffer", "audio_timestamper") 
    pipeline.connect("audio_timestamper", "audio_validator")
    
    return pipeline


def create_validation_chain_pipeline() -> Pipeline:
    """
    Create a pipeline with multiple validation stages.
    
    Flow: Input -> Validator1 -> Math Processor -> Validator2 -> Timestamp
    """
    pipeline = Pipeline(name="validation_chain")
    
    # First validation - check input format
    input_validator = CustomValidatorNode(
        rules={
            "allowed_types": ["int", "float", "list", "dict"]
        },
        strict_mode=True,
        name="input_validator"
    )
    pipeline.add_node(input_validator)
    
    # Process with math operations
    math_processor = MathProcessorNode(
        operations=["square", "abs"],
        handle_lists=True,
        name="safe_math_processor"
    )
    pipeline.add_node(math_processor)
    
    # Second validation - check processing results
    output_validator = CustomValidatorNode(
        rules={
            "required_fields": ["original", "processed", "operations_applied"],
            "allowed_types": ["dict"]
        },
        strict_mode=False,
        name="output_validator"
    )
    pipeline.add_node(output_validator)
    
    # Final timestamp
    timestamp_node = TimestampNode(
        format="iso",
        include_metadata=True,
        name="final_timestamp"
    )
    pipeline.add_node(timestamp_node)
    
    # Connect nodes
    pipeline.connect("input_validator", "safe_math_processor")
    pipeline.connect("safe_math_processor", "output_validator")
    pipeline.connect("output_validator", "final_timestamp")
    
    return pipeline


def create_simple_demo_pipeline() -> Pipeline:
    """
    Create a simple demonstration pipeline.
    
    Flow: Input -> Timestamp -> Output
    """
    pipeline = Pipeline(name="simple_demo")
    
    # Just add a timestamp - simplest possible custom pipeline
    timestamp_node = TimestampNode(
        format="readable",
        include_metadata=True,
        name="demo_timestamper"
    )
    pipeline.add_node(timestamp_node)
    
    return pipeline


# Pipeline registry information
CUSTOM_PIPELINES = {
    "timestamped_processing": {
        "factory": create_timestamped_processing_pipeline,
        "description": "Adds timestamps and validates data",
        "category": "utility",
        "tags": ["timestamp", "validation", "basic"]
    },
    "text_analysis": {
        "factory": create_text_analysis_pipeline,
        "description": "Analyzes text with aggregation",
        "category": "text",
        "tags": ["text", "analysis", "aggregation"]
    },
    "math_processing": {
        "factory": create_math_processing_pipeline,
        "description": "Mathematical operations with validation",
        "category": "math",
        "tags": ["math", "calculation", "validation"]
    },
    "streaming_aggregation": {
        "factory": create_streaming_aggregation_pipeline,
        "description": "Streaming data processing in chunks",
        "category": "streaming",
        "tags": ["streaming", "chunks", "async"]
    },
    "audio_with_metadata": {
        "factory": create_audio_with_metadata_pipeline,
        "description": "Audio processing with custom metadata",
        "category": "audio",
        "tags": ["audio", "metadata", "hybrid"]
    },
    "validation_chain": {
        "factory": create_validation_chain_pipeline,
        "description": "Multi-stage validation pipeline",
        "category": "validation",
        "tags": ["validation", "chain", "strict"]
    },
    "simple_demo": {
        "factory": create_simple_demo_pipeline,
        "description": "Simple demonstration pipeline",
        "category": "demo",
        "tags": ["demo", "simple", "example"]
    }
}


def get_pipeline_info(pipeline_name: str) -> dict:
    """Get information about a specific pipeline."""
    if pipeline_name in CUSTOM_PIPELINES:
        info = CUSTOM_PIPELINES[pipeline_name].copy()
        info.update({
            "name": pipeline_name,
            "version": PIPELINE_VERSION,
            "author": PIPELINE_AUTHOR
        })
        return info
    return None


def list_available_pipelines() -> list:
    """List all available custom pipelines."""
    pipelines = []
    for name, info in CUSTOM_PIPELINES.items():
        pipeline_info = info.copy()
        pipeline_info.update({
            "name": name,
            "version": PIPELINE_VERSION,
            "author": PIPELINE_AUTHOR
        })
        pipelines.append(pipeline_info)
    return pipelines