"""
Automatic discovery system for custom nodes and pipelines.

This module provides functionality to automatically discover and register
custom nodes and pipelines from filesystem directories.
"""

import os
import sys
import importlib
import inspect
import logging
from typing import Dict, Type, Any, List, Callable
from pathlib import Path
from remotemedia.core.node import Node
from remotemedia.core.pipeline import Pipeline


logger = logging.getLogger(__name__)


def discover_nodes(nodes_dir: str = "nodes") -> Dict[str, Type[Node]]:
    """
    Automatically discover custom nodes from a directory.
    
    Args:
        nodes_dir: Directory containing node modules
        
    Returns:
        Dictionary mapping node names to node classes
    """
    nodes_registry = {}
    nodes_path = Path(nodes_dir)
    
    if not nodes_path.exists():
        logger.info(f"Nodes directory '{nodes_dir}' does not exist, skipping node discovery")
        return nodes_registry
    
    logger.info(f"Discovering custom nodes in '{nodes_dir}' directory...")
    
    # Add nodes directory to Python path for imports
    if str(nodes_path.absolute()) not in sys.path:
        sys.path.insert(0, str(nodes_path.absolute().parent))
    
    try:
        # Iterate through all Python files in the nodes directory
        for py_file in nodes_path.glob("*.py"):
            if py_file.name.startswith("__"):
                continue  # Skip __init__.py and __pycache__
                
            module_name = f"{nodes_dir}.{py_file.stem}"
            
            try:
                # Import the module
                module = importlib.import_module(module_name)
                
                # Find all Node subclasses in the module
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, Node) and 
                        obj != Node and  # Don't include the base Node class
                        obj.__module__ == module.__name__):  # Only include classes defined in this module
                        
                        node_name = name
                        nodes_registry[node_name] = obj
                        logger.info(f"Discovered node: {node_name} from {py_file.name}")
                        
            except Exception as e:
                logger.error(f"Failed to import node module {module_name}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error during node discovery: {e}")
    
    logger.info(f"Discovered {len(nodes_registry)} custom nodes")
    return nodes_registry


def discover_pipelines(pipelines_dir: str = "pipelines") -> Dict[str, Dict[str, Any]]:
    """
    Automatically discover custom pipelines from a directory.
    
    Args:
        pipelines_dir: Directory containing pipeline modules
        
    Returns:
        Dictionary mapping pipeline names to pipeline info (factory function, metadata)
    """
    pipelines_registry = {}
    pipelines_path = Path(pipelines_dir)
    
    if not pipelines_path.exists():
        logger.info(f"Pipelines directory '{pipelines_dir}' does not exist, skipping pipeline discovery")
        return pipelines_registry
    
    logger.info(f"Discovering custom pipelines in '{pipelines_dir}' directory...")
    
    # Add pipelines directory to Python path for imports
    if str(pipelines_path.absolute()) not in sys.path:
        sys.path.insert(0, str(pipelines_path.absolute().parent))
    
    try:
        # Iterate through all Python files in the pipelines directory
        for py_file in pipelines_path.glob("*.py"):
            if py_file.name.startswith("__"):
                continue  # Skip __init__.py and __pycache__
                
            module_name = f"{pipelines_dir}.{py_file.stem}"
            
            try:
                # Import the module
                module = importlib.import_module(module_name)
                
                # Look for pipeline factory functions and PIPELINE_REGISTRY
                if hasattr(module, 'PIPELINE_REGISTRY'):
                    # Module defines a pipeline registry
                    registry = getattr(module, 'PIPELINE_REGISTRY')
                    if isinstance(registry, dict):
                        for name, info in registry.items():
                            pipelines_registry[name] = info
                            logger.info(f"Discovered pipeline: {name} from {py_file.name}")
                
                # Also look for functions that return Pipeline objects
                for name, obj in inspect.getmembers(module):
                    if (inspect.isfunction(obj) and 
                        name.startswith("create_") and 
                        name.endswith("_pipeline")):
                        
                        pipeline_name = name.replace("create_", "").replace("_pipeline", "")
                        if pipeline_name not in pipelines_registry:
                            pipelines_registry[pipeline_name] = {
                                "factory": obj,
                                "description": obj.__doc__ or f"{pipeline_name} pipeline",
                                "category": "custom",
                                "source_file": py_file.name
                            }
                            logger.info(f"Discovered pipeline function: {name} -> {pipeline_name} from {py_file.name}")
                        
            except Exception as e:
                logger.error(f"Failed to import pipeline module {module_name}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error during pipeline discovery: {e}")
    
    logger.info(f"Discovered {len(pipelines_registry)} custom pipelines")
    return pipelines_registry


def create_discovery_server(base_dir: str = None, 
                          nodes_dir: str = "nodes", 
                          pipelines_dir: str = "pipelines"):
    """
    Create a server configuration with automatic discovery.
    
    Args:
        base_dir: Base directory for discovery (defaults to current directory)
        nodes_dir: Subdirectory name for nodes
        pipelines_dir: Subdirectory name for pipelines
        
    Returns:
        Tuple of (nodes_registry, pipelines_registry)
    """
    if base_dir:
        original_cwd = os.getcwd()
        os.chdir(base_dir)
    
    try:
        # Discover nodes and pipelines
        nodes_registry = discover_nodes(nodes_dir)
        pipelines_registry = discover_pipelines(pipelines_dir)
        
        return nodes_registry, pipelines_registry
        
    finally:
        if base_dir:
            os.chdir(original_cwd)


def list_discovered_components(nodes_registry: Dict[str, Type[Node]], 
                             pipelines_registry: Dict[str, Dict[str, Any]]) -> None:
    """
    Print a summary of discovered components.
    
    Args:
        nodes_registry: Discovered nodes
        pipelines_registry: Discovered pipelines
    """
    print(f"\n{'=' * 60}")
    print("DISCOVERED CUSTOM COMPONENTS")
    print(f"{'=' * 60}")
    
    print(f"\n📦 Custom Nodes ({len(nodes_registry)}):")
    if nodes_registry:
        for name, node_class in nodes_registry.items():
            category = getattr(node_class, 'CATEGORY', 'unknown')
            doc = node_class.__doc__ or 'No description'
            print(f"  ✓ {name} (category: {category})")
            print(f"    └─ {doc.split('.')[0]}")
    else:
        print("  └─ No custom nodes found")
    
    print(f"\n🔄 Custom Pipelines ({len(pipelines_registry)}):")
    if pipelines_registry:
        for name, info in pipelines_registry.items():
            description = info.get('description', 'No description')
            category = info.get('category', 'unknown')
            source_file = info.get('source_file', 'unknown')
            print(f"  ✓ {name} (category: {category})")
            print(f"    └─ {description.split('.')[0]} [{source_file}]")
    else:
        print("  └─ No custom pipelines found")
    
    print(f"\n{'=' * 60}")
    print(f"Total: {len(nodes_registry)} nodes + {len(pipelines_registry)} pipelines")
    print(f"{'=' * 60}\n")