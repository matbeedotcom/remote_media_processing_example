#!/usr/bin/env python3
"""
Custom RemoteMedia Execution Service Example

This example demonstrates how to create a custom remote execution service
by extending the existing TaskExecutor to include custom nodes while
keeping ALL the sophisticated features from the base server.

Usage:
    python server.py

The server will start on port 50052 by default and include both the standard
RemoteMedia SDK nodes plus any custom nodes defined in this project.
"""

import sys
import os
import logging
import asyncio
from pathlib import Path

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


def create_custom_node_registry():
    """Create registry of custom nodes."""
    return {
        'TimestampNode': TimestampNode,
        'DataAggregatorNode': DataAggregatorNode,
        'TextProcessorNode': TextProcessorNode,
        'MathProcessorNode': MathProcessorNode,
        'CustomValidatorNode': CustomValidatorNode,
        'SimpleStreamingNode': SimpleStreamingNode,
    }


async def main():
    """Main entry point for the custom remote service."""
    # Configure environment
    os.environ.setdefault('GRPC_PORT', '50052')
    os.environ.setdefault('MAX_WORKERS', '4') 
    os.environ.setdefault('LOG_LEVEL', 'INFO')
    os.environ.setdefault('SANDBOX_ENABLED', 'true')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Custom RemoteMedia Execution Service")
    logger.info("Custom nodes will be available:")
    for node_name in create_custom_node_registry().keys():
        logger.info(f"  - {node_name}")
    
    # Create custom node registry
    custom_nodes = create_custom_node_registry()
    
    try:
        # Use the existing serve() function with custom node registry
        # This will create a TaskExecutor with the custom nodes registered
        await serve(custom_node_registry=custom_nodes)
    except KeyboardInterrupt:
        logger.info("Service interrupted")
    except Exception as e:
        logger.error(f"Service error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())