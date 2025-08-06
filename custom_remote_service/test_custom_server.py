#!/usr/bin/env python3
"""
Test script for custom remote service.

This script tests both approaches:
1. Simple custom node registry approach
2. Advanced custom executor approach

Usage:
    # Start simple server: python server.py
    # Start advanced server: python advanced_server.py
    # Run this test: python test_custom_server.py
"""

import asyncio
import sys
import json
from pathlib import Path

# Add the remote_service src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "remote_media_processing" / "remote_service" / "src"))

# For this test, we'll import the client components directly
# In a real deployment, this would be: from remote_media_processing.remote import ...
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "remote_media_processing"))

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_custom_nodes(port: int = 50052):
    """Test custom nodes through direct gRPC calls."""
    import grpc
    import execution_pb2
    import execution_pb2_grpc
    from remotemedia.serialization import JSONSerializer
    
    # Create gRPC channel and client
    channel = grpc.aio.insecure_channel(f'localhost:{port}')
    client = execution_pb2_grpc.RemoteExecutionServiceStub(channel)
    
    serializer = JSONSerializer()
    
    try:
        logger.info(f"Testing custom remote service on port {port}")
        
        # Test 1: TimestampNode
        logger.info("Testing TimestampNode...")
        test_data = {"message": "Hello World", "value": 42}
        input_data = serializer.serialize(test_data)
        
        request = execution_pb2.ExecuteNodeRequest(
            node_type="TimestampNode",
            config={
                "format": "iso",
                "include_metadata": "true"
            },
            input_data=input_data,
            serialization_format="json"
        )
        
        response = await client.ExecuteNode(request)
        if response.status == 1:  # EXECUTION_STATUS_SUCCESS
            result = serializer.deserialize(response.output_data)
            logger.info(f"TimestampNode result: {json.dumps(result, indent=2)}")
        else:
            logger.error(f"TimestampNode failed: {response.error_message}")
        
        # Test 2: MathProcessorNode with numbers
        logger.info("Testing MathProcessorNode with numbers...")
        math_data = [1, 2, 3, 4, 5]
        input_data = serializer.serialize(math_data)
        
        request = execution_pb2.ExecuteNodeRequest(
            node_type="MathProcessorNode",
            config={
                "operations": json.dumps(["square", "double", "sqrt"]),
                "handle_lists": "true"
            },
            input_data=input_data,
            serialization_format="json"
        )
        
        response = await client.ExecuteNode(request)
        if response.status == 1:  # EXECUTION_STATUS_SUCCESS
            result = serializer.deserialize(response.output_data)
            logger.info(f"MathProcessorNode result: {json.dumps(result, indent=2)}")
        else:
            logger.error(f"MathProcessorNode failed: {response.error_message}")
        
        # Test 3: Get available nodes (should include our custom nodes)
        logger.info("Getting available nodes...")
        nodes_request = execution_pb2.ListNodesRequest()
        nodes_response = await client.ListNodes(nodes_request)
        
        custom_node_count = 0
        for node in nodes_response.available_nodes:
            node_type = node.node_type
            if node_type in ['TimestampNode', 'MathProcessorNode']:
                custom_node_count += 1
                logger.info(f"Found custom node: {node_type} (category: {node.category})")
        
        logger.info(f"Total nodes available: {len(nodes_response.available_nodes)}")
        logger.info(f"Custom nodes found: {custom_node_count}")
        
        if custom_node_count >= 2:
            logger.info("✅ All tests passed! Custom nodes are working correctly.")
        else:
            logger.error("❌ Tests failed! Custom nodes not found.")
        
    except grpc.RpcError as e:
        logger.error(f"gRPC error: {e.code()}: {e.details()}")
        if e.code() == grpc.StatusCode.UNAVAILABLE:
            logger.error(f"Server not available on port {port}. Make sure to start the server first:")
            logger.error("  Simple server: python server.py")
            logger.error("  Advanced server: python advanced_server.py")
    except Exception as e:
        logger.error(f"Test error: {e}", exc_info=True)
    finally:
        await channel.close()


async def main():
    """Run tests for both server types."""
    
    # Test simple server (port 50052)
    logger.info("=" * 60)
    logger.info("TESTING SIMPLE CUSTOM SERVER (port 50052)")
    logger.info("=" * 60)
    await test_custom_nodes(50052)
    
    logger.info("\n" + "=" * 60)
    logger.info("TESTING ADVANCED CUSTOM SERVER (port 50053)")
    logger.info("=" * 60)
    await test_custom_nodes(50053)
    
    logger.info("\n" + "=" * 60)
    logger.info("TESTING COMPLETE")
    logger.info("=" * 60)
    logger.info("To run servers:")
    logger.info("  Simple server:   python server.py")
    logger.info("  Advanced server: python advanced_server.py")


if __name__ == "__main__":
    asyncio.run(main())