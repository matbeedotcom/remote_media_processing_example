"""
TimestampNode - Adds timestamp information to incoming data.
"""

import time
from typing import Any, Dict
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