#!/usr/bin/env python3
"""
Filesystem-Based Custom RemoteMedia Execution Service

This example demonstrates how to create a custom remote execution service
that automatically discovers custom nodes and pipelines from filesystem
directories (nodes/ and pipelines/).

Directory Structure:
    nodes/          # Custom node files
    pipelines/      # Custom pipeline files
    server.py       # This server file

Usage:
    python filesystem_server.py

The server will automatically discover and register:
- All Node subclasses from *.py files in nodes/
- All pipeline factory functions from *.py files in pipelines/
"""

import sys
import os
import logging
import asyncio
from pathlib import Path

# Add the remote_service src directory to path
# In a real deployment, this would be: from remote_media_processing.remote_service import serve
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "remote_media_processing" / "remote_service" / "src"))

from server import serve
from discovery import create_discovery_server, list_discovered_components


async def main():
    """Main entry point for the filesystem-based custom remote service."""
    # Configure environment
    os.environ.setdefault('GRPC_PORT', '50054')  # Different port to avoid conflicts
    os.environ.setdefault('MAX_WORKERS', '4') 
    os.environ.setdefault('LOG_LEVEL', 'INFO')
    os.environ.setdefault('SANDBOX_ENABLED', 'true')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Filesystem-Based Custom RemoteMedia Execution Service")
    
    # Discover custom components from filesystem
    logger.info("Discovering custom components from filesystem...")
    nodes_registry, pipelines_registry = create_discovery_server()
    
    # Display discovered components
    list_discovered_components(nodes_registry, pipelines_registry)
    
    if not nodes_registry and not pipelines_registry:
        logger.warning("No custom components discovered!")
        logger.info("To add custom components:")
        logger.info("  1. Add .py files with Node subclasses to the 'nodes/' directory")
        logger.info("  2. Add .py files with pipeline factories to the 'pipelines/' directory")
        logger.info("  3. Restart the server")
    
    try:
        # Use the existing serve() function with discovered nodes
        logger.info(f"Starting server on port {os.environ.get('GRPC_PORT', '50054')}...")
        await serve(custom_node_registry=nodes_registry)
    except KeyboardInterrupt:
        logger.info("Service interrupted")
    except Exception as e:
        logger.error(f"Service error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())