"""
Practical streaming solution using the current RemoteProxyClient.
"""

import asyncio
import time
from remotemedia.core.node import RemoteExecutorConfig
from remotemedia.remote.proxy_client import RemoteProxyClient
from remotemedia.remote.streaming_methods import (
    StreamingDataProcessor, stream_from_remote
)


async def test_streaming_solution():
    """Test the practical streaming solution."""
    config = RemoteExecutorConfig(host="localhost", port=50052, ssl_enabled=False)
    
    async with RemoteProxyClient(config) as client:
        # Create a processor with streaming capabilities
        processor = StreamingDataProcessor()
        remote = await client.create_proxy(processor)
        
        print("=== Streaming Solution Using Batched Fetching ===\n")
        
        # Example 1: Stream file chunks
        print("1. Streaming file chunks (10 at a time):")
        start_time = time.time()
        chunks_processed = 0
        
        async for chunk in stream_from_remote(remote, 'read_large_file', 'bigdata.bin', chunks=30, batch_size=10):
            chunks_processed += 1
            if chunks_processed <= 3:  # Show first 3
                print(f"   Processing: {chunk[:50]}...")
            elif chunks_processed == 4:
                print("   ... (processing more chunks) ...")
            
            # Simulate processing each chunk
            await asyncio.sleep(0.01)
        
        print(f"   Total chunks processed: {chunks_processed}")
        print(f"   Time taken: {time.time() - start_time:.2f}s")
        
        # Example 2: Stream data processing results
        print("\n2. Streaming processed data (5 at a time):")
        items_processed = 0
        
        async for item in stream_from_remote(remote, 'process_data_stream', count=20, batch_size=5):
            items_processed += 1
            if items_processed <= 3:
                print(f"   Received: {item}")
            
            # Early termination example
            if item['value'] > 30:
                print(f"   Stopping early at value {item['value']}")
                break
        
        print(f"   Items processed before stopping: {items_processed}")
        
        print("\n✅ Benefits of this approach:")
        print("   - Works with current RemoteProxyClient")
        print("   - Allows processing items in batches")
        print("   - Supports early termination")
        print("   - Memory efficient for large datasets")
        print("   - Simple to implement")


async def test_manual_streaming():
    """Show manual streaming control."""
    config = RemoteExecutorConfig(host="localhost", port=50052, ssl_enabled=False)
    
    async with RemoteProxyClient(config) as client:
        processor = StreamingDataProcessor()
        remote = await client.create_proxy(processor)
        
        print("\n\n=== Manual Streaming Control ===\n")
        
        # Initialize a stream
        stream_info = await remote.stream_init("my_stream", "read_large_file", "data.txt", chunks=50)
        print(f"Stream initialized: {stream_info}")
        
        # Fetch batches manually
        print("\nFetching batches manually:")
        total_items = 0
        
        while True:
            batch_result = await remote.stream_next_batch("my_stream", 15)
            items = batch_result["items"]
            
            if not items:
                break
            
            total_items += len(items)
            print(f"   Got batch of {len(items)} items (total so far: {total_items})")
            
            # Process the batch
            for item in items[:2]:  # Show first 2 of each batch
                print(f"     - {item[:40]}...")
            
            if not batch_result["has_more"]:
                break
        
        # Clean up
        await remote.stream_close("my_stream")
        print(f"\nStream closed. Total items processed: {total_items}")


async def demonstrate_real_world_use_case():
    """Demonstrate a real-world use case."""
    print("\n\n=== Real-World Use Case: Log Processing ===\n")
    
    class LogProcessor(StreamingDataProcessor):
        """Process log files in chunks."""
        
        def __init__(self):
            super().__init__()
            self.error_count = 0
            self.warning_count = 0
        
        def analyze_logs(self, log_file: str, lines: int = 1000):
            """Simulate analyzing log file lines."""
            for i in range(lines):
                severity = "INFO"
                if i % 50 == 0:
                    severity = "WARNING"
                    self.warning_count += 1
                if i % 100 == 0:
                    severity = "ERROR"
                    self.error_count += 1
                
                yield {
                    "line": i + 1,
                    "severity": severity,
                    "message": f"Log entry {i+1}: {severity} - Something happened",
                    "timestamp": time.time() + i
                }
        
        def get_stats(self):
            """Get current statistics."""
            return {
                "errors": self.error_count,
                "warnings": self.warning_count
            }
    
    config = RemoteExecutorConfig(host="localhost", port=50052, ssl_enabled=False)
    
    async with RemoteProxyClient(config) as client:
        processor = LogProcessor()
        remote = await client.create_proxy(processor)
        
        print("Processing log file in chunks...")
        error_lines = []
        line_count = 0
        
        # Process logs in batches of 50
        async for log_entry in stream_from_remote(remote, 'analyze_logs', 'system.log', lines=500, batch_size=50):
            line_count += 1
            
            # Collect errors
            if log_entry["severity"] == "ERROR":
                error_lines.append(log_entry["line"])
                print(f"   ERROR found at line {log_entry['line']}")
            
            # Show progress every 100 lines
            if line_count % 100 == 0:
                print(f"   Processed {line_count} lines...")
        
        # Get final stats
        stats = await remote.get_stats()
        print(f"\nLog analysis complete:")
        print(f"   Total lines: {line_count}")
        print(f"   Errors: {stats['errors']}")
        print(f"   Warnings: {stats['warnings']}")
        print(f"   Error lines: {error_lines}")


async def main():
    """Run all examples."""
    await test_streaming_solution()
    await test_manual_streaming()
    await demonstrate_real_world_use_case()


if __name__ == "__main__":
    asyncio.run(main())