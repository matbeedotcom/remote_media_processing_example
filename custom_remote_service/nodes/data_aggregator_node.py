"""
DataAggregatorNode - Aggregates data over time windows.
"""

from typing import Any, List
from remotemedia.core.node import Node


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