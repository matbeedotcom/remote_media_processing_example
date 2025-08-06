"""
MathProcessorNode - Performs mathematical operations on numeric data.
"""

import numpy as np
from typing import Any, Dict, List
from remotemedia.core.node import Node


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