"""
Custom Nodes for RemoteMedia Processing Example

This file demonstrates how to create custom nodes that can be used
in a remote execution service.
"""

import time
import asyncio
import numpy as np
from typing import Any, Dict, List, AsyncGenerator
from datetime import datetime

from remotemedia.core.node import Node


class TimestampNode(Node):
    """Adds timestamp information to incoming data."""
    
    CATEGORY = "utility"
    INPUT_TYPES = ["Any"]
    OUTPUT_TYPES = ["Dict"]
    CONFIG_SCHEMA = {
        "format": {
            "type": "string",
            "enum": ["iso", "unix", "readable"],
            "default": "iso",
            "description": "Timestamp format"
        },
        "include_metadata": {
            "type": "boolean", 
            "default": True,
            "description": "Include processing metadata"
        }
    }
    
    def __init__(self, format: str = "iso", include_metadata: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.format = format
        self.include_metadata = include_metadata
        self.process_count = 0
    
    async def process(self, data: Any) -> Dict[str, Any]:
        """Add timestamp to incoming data."""
        self.process_count += 1
        
        # Generate timestamp in requested format
        if self.format == "iso":
            timestamp = datetime.now().isoformat()
        elif self.format == "unix":
            timestamp = time.time()
        elif self.format == "readable":
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        else:
            timestamp = datetime.now().isoformat()
        
        result = {
            "data": data,
            "timestamp": timestamp
        }
        
        if self.include_metadata:
            result["metadata"] = {
                "processed_by": self.name or "TimestampNode",
                "process_count": self.process_count,
                "node_type": "TimestampNode"
            }
        
        return result


class DataAggregatorNode(Node):
    """Aggregates data over time windows."""
    
    CATEGORY = "aggregation"
    INPUT_TYPES = ["Any"]
    OUTPUT_TYPES = ["List"]
    CONFIG_SCHEMA = {
        "window_size": {
            "type": "integer",
            "minimum": 1,
            "default": 5,
            "description": "Number of items to aggregate"
        },
        "aggregation_type": {
            "type": "string",
            "enum": ["collect", "sum", "average", "count"],
            "default": "collect",
            "description": "Type of aggregation to perform"
        }
    }
    
    def __init__(self, window_size: int = 5, aggregation_type: str = "collect", **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.aggregation_type = aggregation_type
        self.buffer: List[Any] = []
    
    async def process(self, data: Any) -> Any:
        """Aggregate data based on window size."""
        self.buffer.append(data)
        
        if len(self.buffer) >= self.window_size:
            # Process the window
            window_data = self.buffer[:self.window_size]
            self.buffer = self.buffer[self.window_size:]  # Slide window
            
            if self.aggregation_type == "collect":
                return window_data
            elif self.aggregation_type == "sum":
                try:
                    return sum(window_data)
                except TypeError:
                    return window_data  # Fallback if sum not applicable
            elif self.aggregation_type == "average":
                try:
                    return sum(window_data) / len(window_data)
                except TypeError:
                    return window_data  # Fallback if average not applicable
            elif self.aggregation_type == "count":
                return len(window_data)
            else:
                return window_data
        
        return None  # Don't output until window is full
    
    async def flush(self) -> Any:
        """Flush remaining buffer contents."""
        if self.buffer:
            remaining = self.buffer.copy()
            self.buffer.clear()
            
            if self.aggregation_type == "collect":
                return remaining
            elif self.aggregation_type == "count":
                return len(remaining)
            else:
                return remaining
        return None


class TextProcessorNode(Node):
    """Processes text data with various operations."""
    
    CATEGORY = "text"
    INPUT_TYPES = ["str", "Dict"]
    OUTPUT_TYPES = ["Dict"]
    CONFIG_SCHEMA = {
        "operations": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["lowercase", "uppercase", "word_count", "char_count", "reverse"]
            },
            "default": ["lowercase", "word_count"],
            "description": "Text processing operations to perform"
        },
        "extract_field": {
            "type": "string",
            "default": None,
            "description": "Field to extract from dict input (if applicable)"
        }
    }
    
    def __init__(self, operations: List[str] = None, extract_field: str = None, **kwargs):
        super().__init__(**kwargs)
        self.operations = operations or ["lowercase", "word_count"]
        self.extract_field = extract_field
    
    async def process(self, data: Any) -> Dict[str, Any]:
        """Process text data with configured operations."""
        # Extract text from input
        if isinstance(data, str):
            text = data
            original = data
        elif isinstance(data, dict):
            if self.extract_field and self.extract_field in data:
                text = str(data[self.extract_field])
                original = data
            else:
                text = str(data)
                original = data
        else:
            text = str(data)
            original = data
        
        result = {
            "original": original,
            "text": text,
            "processed": {}
        }
        
        # Apply operations
        for operation in self.operations:
            if operation == "lowercase":
                result["processed"]["lowercase"] = text.lower()
            elif operation == "uppercase":
                result["processed"]["uppercase"] = text.upper()
            elif operation == "word_count":
                result["processed"]["word_count"] = len(text.split())
            elif operation == "char_count":
                result["processed"]["char_count"] = len(text)
            elif operation == "reverse":
                result["processed"]["reverse"] = text[::-1]
        
        return result


class SimpleStreamingNode(Node):
    """Demonstrates streaming processing capabilities."""
    
    CATEGORY = "streaming"
    INPUT_TYPES = ["Any"]
    OUTPUT_TYPES = ["Dict"]
    is_streaming = True  # Mark as streaming node
    
    CONFIG_SCHEMA = {
        "chunk_size": {
            "type": "integer",
            "minimum": 1,
            "default": 3,
            "description": "Number of items to process per output chunk"
        },
        "delay": {
            "type": "number",
            "minimum": 0,
            "default": 0.1,
            "description": "Delay between chunks (seconds)"
        }
    }
    
    def __init__(self, chunk_size: int = 3, delay: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.delay = delay
        self.buffer: List[Any] = []
        self.chunk_counter = 0
    
    async def process(self, data_stream: AsyncGenerator[Any, None]) -> AsyncGenerator[Dict[str, Any], None]:
        """Process streaming data in chunks."""
        async for item in data_stream:
            self.buffer.append(item)
            
            if len(self.buffer) >= self.chunk_size:
                # Process a chunk
                chunk_data = self.buffer[:self.chunk_size]
                self.buffer = self.buffer[self.chunk_size:]
                
                self.chunk_counter += 1
                
                result = {
                    "chunk_id": self.chunk_counter,
                    "chunk_size": len(chunk_data),
                    "data": chunk_data,
                    "timestamp": time.time(),
                    "processed_by": self.name or "SimpleStreamingNode"
                }
                
                if self.delay > 0:
                    await asyncio.sleep(self.delay)
                
                yield result
    
    async def flush(self) -> Dict[str, Any]:
        """Flush remaining buffer contents."""
        if self.buffer:
            self.chunk_counter += 1
            result = {
                "chunk_id": self.chunk_counter,
                "chunk_size": len(self.buffer),
                "data": self.buffer.copy(),
                "timestamp": time.time(),
                "processed_by": self.name or "SimpleStreamingNode",
                "final_chunk": True
            }
            self.buffer.clear()
            return result
        return None


class MathProcessorNode(Node):
    """Performs mathematical operations on numeric data."""
    
    CATEGORY = "math"
    INPUT_TYPES = ["int", "float", "List", "Dict"]
    OUTPUT_TYPES = ["Dict"]
    CONFIG_SCHEMA = {
        "operations": {
            "type": "array",
            "items": {
                "type": "string", 
                "enum": ["square", "sqrt", "double", "half", "abs", "sin", "cos", "log"]
            },
            "default": ["square", "double"],
            "description": "Mathematical operations to perform"
        },
        "handle_lists": {
            "type": "boolean",
            "default": True,
            "description": "Apply operations to list elements"
        }
    }
    
    def __init__(self, operations: List[str] = None, handle_lists: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.operations = operations or ["square", "double"]
        self.handle_lists = handle_lists
    
    async def process(self, data: Any) -> Dict[str, Any]:
        """Apply mathematical operations to numeric data."""
        result = {
            "original": data,
            "processed": {},
            "operations_applied": self.operations
        }
        
        # Handle different input types
        if isinstance(data, (int, float)):
            numbers = [data]
        elif isinstance(data, list) and self.handle_lists:
            numbers = [x for x in data if isinstance(x, (int, float))]
        elif isinstance(data, dict):
            # Try to extract numeric values
            numbers = [v for v in data.values() if isinstance(v, (int, float))]
        else:
            numbers = []
        
        if not numbers:
            result["error"] = "No numeric values found in input"
            return result
        
        # Apply operations
        for operation in self.operations:
            try:
                if operation == "square":
                    result["processed"]["square"] = [x ** 2 for x in numbers]
                elif operation == "sqrt":
                    result["processed"]["sqrt"] = [np.sqrt(abs(x)) for x in numbers]
                elif operation == "double":
                    result["processed"]["double"] = [x * 2 for x in numbers]
                elif operation == "half":
                    result["processed"]["half"] = [x / 2 for x in numbers]
                elif operation == "abs":
                    result["processed"]["abs"] = [abs(x) for x in numbers]
                elif operation == "sin":
                    result["processed"]["sin"] = [np.sin(x) for x in numbers]
                elif operation == "cos":
                    result["processed"]["cos"] = [np.cos(x) for x in numbers]
                elif operation == "log":
                    result["processed"]["log"] = [np.log(x) if x > 0 else float('-inf') for x in numbers]
            except Exception as e:
                result["processed"][f"{operation}_error"] = str(e)
        
        return result


class CustomValidatorNode(Node):
    """Validates data against custom rules."""
    
    CATEGORY = "validation"
    INPUT_TYPES = ["Any"]
    OUTPUT_TYPES = ["Dict"]
    CONFIG_SCHEMA = {
        "rules": {
            "type": "object",
            "properties": {
                "required_fields": {"type": "array", "items": {"type": "string"}},
                "min_value": {"type": "number"},
                "max_value": {"type": "number"},
                "allowed_types": {"type": "array", "items": {"type": "string"}}
            },
            "default": {},
            "description": "Validation rules to apply"
        },
        "strict_mode": {
            "type": "boolean",
            "default": False,
            "description": "Fail validation on first error"
        }
    }
    
    def __init__(self, rules: Dict[str, Any] = None, strict_mode: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.rules = rules or {}
        self.strict_mode = strict_mode
        self.validation_count = 0
    
    async def process(self, data: Any) -> Dict[str, Any]:
        """Validate data against configured rules."""
        self.validation_count += 1
        
        result = {
            "original": data,
            "validation_id": self.validation_count,
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required fields (for dict input)
        if "required_fields" in self.rules and isinstance(data, dict):
            for field in self.rules["required_fields"]:
                if field not in data:
                    error = f"Missing required field: {field}"
                    result["errors"].append(error)
                    result["is_valid"] = False
                    if self.strict_mode:
                        return result
        
        # Check value ranges (for numeric input)
        if isinstance(data, (int, float)):
            if "min_value" in self.rules and data < self.rules["min_value"]:
                error = f"Value {data} is below minimum {self.rules['min_value']}"
                result["errors"].append(error)
                result["is_valid"] = False
                if self.strict_mode:
                    return result
            
            if "max_value" in self.rules and data > self.rules["max_value"]:
                error = f"Value {data} is above maximum {self.rules['max_value']}"
                result["errors"].append(error)
                result["is_valid"] = False
                if self.strict_mode:
                    return result
        
        # Check allowed types
        if "allowed_types" in self.rules:
            data_type = type(data).__name__
            if data_type not in self.rules["allowed_types"]:
                error = f"Type {data_type} not in allowed types: {self.rules['allowed_types']}"
                result["errors"].append(error)
                result["is_valid"] = False
                if self.strict_mode:
                    return result
        
        return result