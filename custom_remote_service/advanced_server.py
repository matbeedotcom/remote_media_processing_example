#!/usr/bin/env python3
"""
Advanced Custom RemoteMedia Execution Service Example

This example demonstrates an alternative approach where developers can create
their own TaskExecutor subclass for more advanced customization while still
using the existing infrastructure.

Usage:
    python advanced_server.py

This approach gives developers full control over executor initialization
and custom logic while maintaining all existing server features.
"""

import sys
import os
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List

# Add the remote_service src directory to path so we can import the core components
# In a real deployment, this would be imported from the installed package
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "remote_media_processing" / "remote_service" / "src"))

from server import serve
from executor import TaskExecutor
from config import ServiceConfig

# Import our custom nodes
from custom_nodes import (
    TimestampNode,
    DataAggregatorNode,
    TextProcessorNode,
    MathProcessorNode,
    CustomValidatorNode,
    SimpleStreamingNode
)


class CustomTaskExecutor(TaskExecutor):
    """
    Custom TaskExecutor subclass demonstrating advanced customization.
    
    This approach allows developers to:
    - Override executor methods
    - Add custom initialization logic
    - Implement specialized node handling
    - Add custom metrics or logging
    """
    
    def __init__(self, config: ServiceConfig):
        # Create custom node registry
        custom_nodes = {
            'TimestampNode': TimestampNode,
            'DataAggregatorNode': DataAggregatorNode,
            'TextProcessorNode': TextProcessorNode,
            'MathProcessorNode': MathProcessorNode,
            'CustomValidatorNode': CustomValidatorNode,
            'SimpleStreamingNode': SimpleStreamingNode,
        }
        
        # Initialize parent with custom nodes
        super().__init__(config, custom_nodes)
        
        # Add any custom initialization
        self.custom_metrics = {
            'custom_node_executions': 0,
            'total_processing_time': 0.0
        }
        
        self.logger.info(f"CustomTaskExecutor initialized with custom metrics tracking")
    
    async def execute_sdk_node(self, node_type: str, config: Dict[str, Any], 
                              input_data: bytes, serialization_format: str, 
                              options: Any):
        """Override to add custom metrics tracking."""
        import time
        start_time = time.time()
        
        # Track custom node executions
        custom_node_names = ['TimestampNode', 'DataAggregatorNode', 'TextProcessorNode',
                           'MathProcessorNode', 'CustomValidatorNode', 'SimpleStreamingNode']
        
        if node_type in custom_node_names:
            self.custom_metrics['custom_node_executions'] += 1
            self.logger.info(f"Executing custom node: {node_type} (#{self.custom_metrics['custom_node_executions']})")
        
        # Execute the node using parent implementation
        result = await super().execute_sdk_node(node_type, config, input_data, 
                                               serialization_format, options)
        
        # Track processing time
        processing_time = time.time() - start_time
        self.custom_metrics['total_processing_time'] += processing_time
        
        if node_type in custom_node_names:
            self.logger.info(f"Custom node {node_type} completed in {processing_time:.3f}s")
        
        return result
    
    async def get_available_nodes(self, category: str = None) -> List[Any]:
        """Override to add custom node metadata."""
        nodes = await super().get_available_nodes(category)
        
        # Add custom metrics to node info for custom nodes
        for node in nodes:
            if node['node_type'] in ['TimestampNode', 'DataAggregatorNode', 'TextProcessorNode',
                                   'MathProcessorNode', 'CustomValidatorNode', 'SimpleStreamingNode']:
                node['custom'] = True
                node['execution_count'] = self.custom_metrics['custom_node_executions']
        
        return nodes
    
    def get_custom_metrics(self) -> Dict[str, Any]:
        """Get custom metrics (could be exposed via gRPC extension)."""
        return self.custom_metrics.copy()


async def main():
    """Main entry point for the advanced custom remote service."""
    # Configure environment
    os.environ.setdefault('GRPC_PORT', '50053')  # Different port to avoid conflicts
    os.environ.setdefault('MAX_WORKERS', '4') 
    os.environ.setdefault('LOG_LEVEL', 'INFO')
    os.environ.setdefault('SANDBOX_ENABLED', 'true')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Advanced Custom RemoteMedia Execution Service")
    
    # Create configuration
    config = ServiceConfig()
    
    # Create custom executor instance
    custom_executor = CustomTaskExecutor(config)
    
    logger.info("Custom executor created with advanced features:")
    logger.info(f"  - Custom metrics tracking")
    logger.info(f"  - Enhanced logging")
    logger.info(f"  - Custom node metadata")
    logger.info(f"  - Total nodes: {len(custom_executor.node_registry)}")
    
    try:
        # Use the existing serve() function with custom executor instance
        await serve(custom_executor=custom_executor)
    except KeyboardInterrupt:
        logger.info("Service interrupted")
        logger.info(f"Final metrics: {custom_executor.get_custom_metrics()}")
    except Exception as e:
        logger.error(f"Service error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())