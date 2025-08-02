"""
Remote Class Execution Demo Client

This script demonstrates how to execute any class/instance remotely using the remote_media framework.
It shows various patterns including:
- Simple method calls
- Async method execution
- State management across calls
- Streaming data handling
- Batch processing
"""

import asyncio
import time
from typing import List
import sys
import os

# Add the parent directory to path so we can import remotemedia
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from remotemedia.core.node import RemoteExecutorConfig
from remotemedia.remote.proxy_client import RemoteProxyClient
from sample_classes import DataProcessor, ScientificCalculator, StreamingService, StatefulService


async def demo_data_processor():
    """Demonstrate remote execution of DataProcessor class."""
    print("\n" + "="*60)
    print("DEMO: Remote DataProcessor Execution")
    print("="*60)
    
    config = RemoteExecutorConfig(host="localhost", port=50052, ssl_enabled=False)
    
    async with RemoteProxyClient(config) as client:
        # Create a remote instance of DataProcessor
        processor = DataProcessor(name="RemoteProcessor", buffer_size=100)
        remote_processor = await client.create_proxy(processor)
        
        # Test simple processing
        print("\n1. Simple Processing:")
        result = await remote_processor.process_simple(42.0)
        print(f"   Input: 42.0, Output: {result}")
        
        # Test batch processing
        print("\n2. Batch Processing:")
        batch_input = [1.0, 2.0, 3.0, 4.0, 5.0]
        batch_result = await remote_processor.process_batch(batch_input)
        print(f"   Input: {batch_input}")
        print(f"   Output: {batch_result}")
        
        # Test async processing
        print("\n3. Async Processing:")
        async_result = await remote_processor.async_process(10.0, delay=0.2)
        print(f"   Result: {async_result}")
        
        # Test buffer operations
        print("\n4. Buffer Operations:")
        for i in range(5):
            success = await remote_processor.add_to_buffer(f"Item {i}")
            print(f"   Added item {i}: {success}")
        
        stats = await remote_processor.get_buffer_stats()
        print(f"   Buffer stats: {stats}")
        
        cleared = await remote_processor.clear_buffer()
        print(f"   Cleared {cleared} items from buffer")


async def demo_scientific_calculator():
    """Demonstrate remote execution of ScientificCalculator class."""
    print("\n" + "="*60)
    print("DEMO: Remote ScientificCalculator Execution")
    print("="*60)
    
    config = RemoteExecutorConfig(host="localhost", port=50052, ssl_enabled=False)
    
    async with RemoteProxyClient(config) as client:
        # Create a remote calculator instance
        calculator = ScientificCalculator(precision=4)
        remote_calc = await client.create_proxy(calculator)
        
        # Perform various calculations
        print("\n1. Basic Operations:")
        operations = [
            ("add", 10, 5),
            ("multiply", 3, 7),
            ("power", 2, 8),
            ("sqrt", 16),
        ]
        
        for op in operations:
            result = await remote_calc.calculate(*op)
            print(f"   {op[0]}({', '.join(map(str, op[1:]))})) = {result}")
        
        # Matrix multiplication
        print("\n2. Matrix Multiplication:")
        matrix_a = [[1, 2], [3, 4]]
        matrix_b = [[5, 6], [7, 8]]
        matrix_result = await remote_calc.matrix_multiply(matrix_a, matrix_b)
        print(f"   A = {matrix_a}")
        print(f"   B = {matrix_b}")
        print(f"   A × B = {matrix_result}")
        
        # Statistical analysis
        print("\n3. Statistical Analysis:")
        data = [1.5, 2.3, 3.1, 4.7, 5.2, 6.8, 7.4, 8.9, 9.2, 10.1]
        stats = await remote_calc.statistical_analysis(data)
        print(f"   Data: {data}")
        for key, value in stats.items():
            print(f"   {key}: {value:.4f}")
        
        # Get calculation history
        print("\n4. Calculation History:")
        history = await remote_calc.get_history(limit=3)
        for entry in history:
            print(f"   {entry}")


async def demo_streaming_service():
    """Demonstrate streaming capabilities with remote execution."""
    print("\n" + "="*60)
    print("DEMO: Remote StreamingService Execution")
    print("="*60)
    
    config = RemoteExecutorConfig(host="localhost", port=50052, ssl_enabled=False)
    
    async with RemoteProxyClient(config) as client:
        # Create a remote streaming service
        service = StreamingService(chunk_size=2048)
        remote_service = await client.create_proxy(service)
        
        # Note: Current implementation materializes generators to lists
        print("\n1. Data Stream Generation (currently materialized):")
        stream_data = await remote_service.generate_data_stream(5)
        print(f"   Received {len(stream_data)} chunks")
        for chunk in stream_data[:2]:  # Show first 2 chunks
            print(f"   Chunk {chunk['chunk_id']}: {chunk['data']}")
        
        print("\n2. Async Stream Generation:")
        async_stream = await remote_service.async_generate_stream(5)
        print(f"   Received {len(async_stream)} async chunks")
        for chunk in async_stream[:2]:  # Show first 2 chunks
            print(f"   Chunk {chunk['chunk_id']}: value={chunk['value']:.4f}")
        
        # Process a stream
        print("\n3. Stream Processing:")
        input_data = ["Item A", "Item B", "Item C"]
        processed = await remote_service.process_stream(input_data)
        print(f"   Processed {len(processed)} items")
        for item in processed:
            print(f"   {item['result']}")


async def demo_stateful_service():
    """Demonstrate stateful service with persistent state across calls."""
    print("\n" + "="*60)
    print("DEMO: Remote StatefulService Execution")
    print("="*60)
    
    config = RemoteExecutorConfig(host="localhost", port=50052, ssl_enabled=False)
    
    async with RemoteProxyClient(config) as client:
        # Create a remote stateful service
        service = StatefulService(service_id="demo-service-001")
        remote_service = await client.create_proxy(service)
        
        # Set and get state
        print("\n1. State Management:")
        await remote_service.set_state("user", "Alice")
        await remote_service.set_state("session_id", "xyz123")
        await remote_service.set_state("preferences", {"theme": "dark", "language": "en"})
        
        user = await remote_service.get_state("user")
        prefs = await remote_service.get_state("preferences")
        print(f"   User: {user}")
        print(f"   Preferences: {prefs}")
        
        # Use counters
        print("\n2. Counter Operations:")
        for i in range(3):
            count = await remote_service.increment_counter("api_calls")
            print(f"   API calls: {count}")
        
        for i in range(2):
            count = await remote_service.increment_counter("errors")
            print(f"   Errors: {count}")
        
        # Lock/unlock operations
        print("\n3. Lock/Unlock Operations:")
        locked = await remote_service.lock()
        print(f"   Lock acquired: {locked}")
        
        try:
            await remote_service.set_state("locked_update", "should fail")
        except Exception as e:
            print(f"   Expected error when locked: {type(e).__name__}")
        
        unlocked = await remote_service.unlock()
        print(f"   Unlocked: {unlocked}")
        
        # Get service info
        print("\n4. Service Information:")
        info = await remote_service.get_service_info()
        for key, value in info.items():
            print(f"   {key}: {value}")


async def demo_multiple_instances():
    """Demonstrate working with multiple remote instances simultaneously."""
    print("\n" + "="*60)
    print("DEMO: Multiple Remote Instances")
    print("="*60)
    
    config = RemoteExecutorConfig(host="localhost", port=50052, ssl_enabled=False)
    
    async with RemoteProxyClient(config) as client:
        # Create multiple remote instances
        processor1 = DataProcessor(name="Processor1")
        processor2 = DataProcessor(name="Processor2")
        calculator = ScientificCalculator()
        
        remote_p1 = await client.create_proxy(processor1)
        remote_p2 = await client.create_proxy(processor2)
        remote_calc = await client.create_proxy(calculator)
        
        print("\n1. Parallel Processing:")
        # Process data in parallel using multiple processors
        tasks = []
        for i in range(5):
            if i % 2 == 0:
                tasks.append(remote_p1.async_process(i, delay=0.1))
            else:
                tasks.append(remote_p2.async_process(i, delay=0.1))
        
        results = await asyncio.gather(*tasks)
        for result in results:
            print(f"   {result['processor']}: processed {result['input']} → {result['output']}")
        
        # Get stats from both processors
        print("\n2. Instance Statistics:")
        stats1 = await remote_p1.get_buffer_stats()
        stats2 = await remote_p2.get_buffer_stats()
        print(f"   Processor1: {stats1['processing_count']} items processed")
        print(f"   Processor2: {stats2['processing_count']} items processed")
        
        # Use calculator for aggregate statistics
        print("\n3. Aggregate Calculations:")
        values = [r['output'] for r in results]
        stats = await remote_calc.statistical_analysis(values)
        print(f"   Mean of outputs: {stats['mean']:.4f}")
        print(f"   Std deviation: {stats['std']:.4f}")


async def main():
    """Run all demonstrations."""
    demos = [
        ("Data Processor", demo_data_processor),
        ("Scientific Calculator", demo_scientific_calculator),
        ("Streaming Service", demo_streaming_service),
        ("Stateful Service", demo_stateful_service),
        ("Multiple Instances", demo_multiple_instances),
    ]
    
    print("\n" + "="*80)
    print("REMOTE CLASS EXECUTION DEMONSTRATION")
    print("="*80)
    print("\nThis demo shows how to execute any Python class/instance remotely")
    print("using the remote_media framework.")
    
    print("\n⚠️  Prerequisites:")
    print("1. Ensure the remote execution server is running:")
    print("   cd ../../remote_service")
    print("   python src/server.py")
    print("\n2. Or use Docker:")
    print("   docker-compose up")
    
    input("\nPress Enter to start the demonstrations...")
    
    for name, demo_func in demos:
        try:
            await demo_func()
        except Exception as e:
            print(f"\n❌ Error in {name} demo: {e}")
            print("   Make sure the remote server is running!")
        
        if demo_func != demos[-1][1]:  # Don't wait after last demo
            input("\nPress Enter to continue to next demo...")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("- Any Python class can be executed remotely")
    print("- Method calls are transparently proxied to remote server")
    print("- State is maintained on the remote server")
    print("- Supports sync and async methods")
    print("- Multiple instances can run simultaneously")
    print("\nNote: Generator methods are currently materialized to lists")
    print("(see test_streaming_generators.py for details)")


if __name__ == "__main__":
    asyncio.run(main())