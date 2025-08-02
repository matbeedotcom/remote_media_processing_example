"""
Sample classes to demonstrate remote execution capabilities.
These classes showcase different types of methods and operations that can be executed remotely.
"""

import time
import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime


class DataProcessor:
    """A sample data processing class with various method types."""
    
    def __init__(self, name: str = "DataProcessor", buffer_size: int = 1000):
        self.name = name
        self.buffer_size = buffer_size
        self.data_buffer = []
        self.processing_count = 0
        self.start_time = datetime.now()
    
    def process_simple(self, value: float) -> float:
        """Simple synchronous processing method."""
        self.processing_count += 1
        return value * 2.5 + 10
    
    def process_batch(self, values: List[float]) -> List[float]:
        """Process a batch of values."""
        self.processing_count += len(values)
        return [self.process_simple(v) for v in values]
    
    async def async_process(self, value: float, delay: float = 0.1) -> Dict[str, Any]:
        """Asynchronous processing with simulated delay."""
        await asyncio.sleep(delay)
        self.processing_count += 1
        return {
            "input": value,
            "output": value ** 2,
            "timestamp": datetime.now().isoformat(),
            "processor": self.name,
            "total_processed": self.processing_count
        }
    
    def add_to_buffer(self, data: Any) -> bool:
        """Add data to internal buffer."""
        if len(self.data_buffer) < self.buffer_size:
            self.data_buffer.append(data)
            return True
        return False
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get statistics about the buffer."""
        return {
            "buffer_size": len(self.data_buffer),
            "max_size": self.buffer_size,
            "processing_count": self.processing_count,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
        }
    
    def clear_buffer(self) -> int:
        """Clear the buffer and return number of items cleared."""
        count = len(self.data_buffer)
        self.data_buffer.clear()
        return count


class ScientificCalculator:
    """A calculator class with various mathematical operations."""
    
    def __init__(self, precision: int = 6):
        self.precision = precision
        self.history = []
    
    def calculate(self, operation: str, *args) -> float:
        """Perform a calculation and store in history."""
        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else float('inf'),
            "power": lambda x, y: x ** y,
            "sqrt": lambda x: x ** 0.5,
            "log": lambda x: np.log(x) if x > 0 else float('-inf')
        }
        
        if operation not in operations:
            raise ValueError(f"Unknown operation: {operation}")
        
        result = operations[operation](*args)
        self.history.append({
            "operation": operation,
            "args": args,
            "result": round(result, self.precision),
            "timestamp": datetime.now().isoformat()
        })
        
        return round(result, self.precision)
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get calculation history."""
        if limit:
            return self.history[-limit:]
        return self.history.copy()
    
    def matrix_multiply(self, matrix_a: List[List[float]], matrix_b: List[List[float]]) -> List[List[float]]:
        """Perform matrix multiplication."""
        a = np.array(matrix_a)
        b = np.array(matrix_b)
        result = np.matmul(a, b)
        return result.tolist()
    
    def statistical_analysis(self, data: List[float]) -> Dict[str, float]:
        """Perform statistical analysis on data."""
        arr = np.array(data)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "variance": float(np.var(arr))
        }


class StreamingService:
    """A class that demonstrates streaming capabilities."""
    
    def __init__(self, chunk_size: int = 1024):
        self.chunk_size = chunk_size
        self.active_streams = 0
    
    def generate_data_stream(self, count: int = 10):
        """Generate a stream of data chunks."""
        self.active_streams += 1
        try:
            for i in range(count):
                time.sleep(0.1)  # Simulate data generation
                yield {
                    "chunk_id": i,
                    "data": f"Data chunk {i} of size {self.chunk_size}",
                    "timestamp": time.time()
                }
        finally:
            self.active_streams -= 1
    
    async def async_generate_stream(self, count: int = 10):
        """Asynchronously generate a stream of data."""
        self.active_streams += 1
        try:
            for i in range(count):
                await asyncio.sleep(0.1)
                yield {
                    "chunk_id": i,
                    "data": f"Async chunk {i}",
                    "value": np.random.random(),
                    "timestamp": time.time()
                }
        finally:
            self.active_streams -= 1
    
    def process_stream(self, input_stream):
        """Process an input stream and yield results."""
        for item in input_stream:
            # Simulate processing
            processed = {
                "original": item,
                "processed_at": time.time(),
                "result": f"Processed: {item}"
            }
            yield processed


class StatefulService:
    """A service that maintains state across method calls."""
    
    def __init__(self, service_id: str):
        self.service_id = service_id
        self.state = {}
        self.counters = {}
        self.locked = False
    
    def set_state(self, key: str, value: Any) -> None:
        """Set a state value."""
        if self.locked:
            raise RuntimeError("Service is locked")
        self.state[key] = value
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a state value."""
        return self.state.get(key, default)
    
    def increment_counter(self, counter_name: str) -> int:
        """Increment and return a counter value."""
        if counter_name not in self.counters:
            self.counters[counter_name] = 0
        self.counters[counter_name] += 1
        return self.counters[counter_name]
    
    def lock(self) -> bool:
        """Lock the service to prevent state changes."""
        if not self.locked:
            self.locked = True
            return True
        return False
    
    def unlock(self) -> bool:
        """Unlock the service."""
        if self.locked:
            self.locked = False
            return True
        return False
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the service."""
        return {
            "service_id": self.service_id,
            "state_keys": list(self.state.keys()),
            "counters": self.counters.copy(),
            "locked": self.locked,
            "state_size": len(self.state)
        }