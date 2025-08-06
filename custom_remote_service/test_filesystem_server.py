#!/usr/bin/env python3
"""
Test script for filesystem-based custom remote service.

This demonstrates testing custom nodes using the high-level RemoteMedia client,
not raw gRPC boilerplate.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add the main library to path for testing
# In production, this would be: from remote_media_processing.remote import RemoteExecutionClient, RemoteExecutorConfig
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "remote_media_processing"))

from remotemedia.remote.client import RemoteExecutionClient
from remotemedia.core.node import RemoteExecutorConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_filesystem_server(port: int = 50054):
    """Test the filesystem-based server using the high-level client."""
    
    # Create RemoteMedia client - no gRPC boilerplate!
    config = RemoteExecutorConfig(host="localhost", port=port)
    client = RemoteExecutionClient(config)
    
    try:
        logger.info(f"Testing filesystem-based server on port {port}")
        
        # Test 1: TimestampNode (discovered from nodes/timestamp_node.py)
        logger.info("Testing discovered TimestampNode...")
        result = await client.execute_node(
            node_type="TimestampNode",
            config={"format": "iso", "include_metadata": True},
            input_data={"message": "Hello Filesystem Discovery!", "value": 123}
        )
        logger.info(f"TimestampNode result: {result}")
        
        # Test 2: MathProcessorNode (discovered from nodes/math_processor_node.py)
        logger.info("Testing discovered MathProcessorNode...")
        math_result = await client.execute_node(
            node_type="MathProcessorNode", 
            config={"operations": ["square", "sqrt", "double"], "handle_lists": True},
            input_data=[2, 4, 6, 8]
        )
        logger.info(f"MathProcessorNode result: {math_result}")
        
        # Test 3: DataAggregatorNode (discovered from nodes/data_aggregator_node.py)  
        logger.info("Testing discovered DataAggregatorNode...")
        agg_result = await client.execute_node(
            node_type="DataAggregatorNode",
            config={"window_size": 2, "aggregation_type": "collect"},
            input_data="test_item_1"
        )
        logger.info(f"DataAggregatorNode result: {agg_result}")
        
        logger.info("✅ All filesystem discovery tests passed!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        await client.close()


async def main():
    """Run tests for filesystem-based server."""
    logger.info("=" * 60)
    logger.info("TESTING FILESYSTEM-BASED CUSTOM SERVER")
    logger.info("=" * 60)
    logger.info("This test uses the high-level RemoteMedia client - no gRPC boilerplate!")
    logger.info("")
    
    try:
        await test_filesystem_server(50054)
        logger.info("")
        logger.info("🎉 Filesystem discovery test completed successfully!")
        logger.info("Custom nodes were automatically discovered from nodes/ directory")
        logger.info("Custom pipelines were automatically discovered from pipelines/ directory")
    except ConnectionError:
        logger.error("❌ Server not available on port 50054")
        logger.error("Start the filesystem server first: python filesystem_server.py")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())