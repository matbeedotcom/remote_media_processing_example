"""
Example demonstrating transparent remote execution using proxy client.
"""

import asyncio
from remotemedia.core.node import RemoteExecutorConfig
from remotemedia.remote.proxy_client import RemoteProxyClient, remote_class
from remotemedia.examples.test_classes import Counter, DataProcessor


async def main():
    # Configure remote execution
    config = RemoteExecutorConfig(
        host="localhost",
        port=50052,
        ssl_enabled=False
    )
    
    async with RemoteProxyClient(config) as client:
        print("=== Example 1: Manual Proxy Creation ===")
        
        # Create a local counter object
        counter = Counter(initial_value=10)
        
        # Create a remote proxy - the counter now lives on the remote server
        remote_counter = await client.create_proxy(counter)
        
        # All method calls are transparently executed remotely
        print(f"Initial value: {await remote_counter.get_value()}")  # Returns 10
        
        new_value = await remote_counter.increment()
        print(f"After increment: {new_value}")  # Returns 11
        
        final_value = await remote_counter.add(5)
        print(f"After adding 5: {final_value}")  # Returns 16
        
        print("\n=== Example 2: Direct Remote Object Creation ===")
        
        # Create a remote processor directly
        processor = DataProcessor(mode="ultra-fast")
        remote_processor = await client.create_proxy(processor)
        
        # Process some data remotely
        result1 = await remote_processor.process("Hello World")
        print(f"Result 1: {result1}")
        
        result2 = await remote_processor.process("Remote Execution")
        print(f"Result 2: {result2}")
        
        # Get statistics
        stats = await remote_processor.get_stats()
        print(f"Stats: {stats}")
        
        print("\n=== Example 3: Chained Method Calls ===")
        
        # You can chain multiple operations
        remote_counter2 = await client.create_proxy(Counter(100))
        
        # Each of these is a separate remote call
        await remote_counter2.increment()
        await remote_counter2.increment()
        await remote_counter2.add(10)
        
        final = await remote_counter2.get_value()
        print(f"Final counter value: {final}")  # Should be 112


async def alternative_syntax_example():
    """
    Example showing an alternative syntax using __getattr__ capture.
    """
    config = RemoteExecutorConfig(
        host="localhost",
        port=50052,
        ssl_enabled=False
    )
    
    # This shows how you could implement a more dynamic client
    # that captures all calls, similar to your example:
    # client.initialize(arg1, arg2, test=True)
    
    class DynamicRemoteClient:
        def __init__(self, config):
            self.config = config
            self.proxy_client = RemoteProxyClient(config)
            self._connected = False
        
        async def __aenter__(self):
            await self.proxy_client.connect()
            self._connected = True
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            await self.proxy_client.disconnect()
        
        def __getattr__(self, name):
            """
            Capture any method call and execute it remotely.
            This allows syntax like: client.initialize(arg1, arg2)
            """
            async def remote_call(*args, **kwargs):
                if not self._connected:
                    raise RuntimeError("Client not connected. Use 'async with' context.")
                
                # Create a temporary object that has this method
                class DynamicObject:
                    pass
                
                obj = DynamicObject()
                # Add the method dynamically
                setattr(obj, name, lambda *a, **k: (a, k))
                
                # Execute remotely
                proxy = await self.proxy_client.create_proxy(obj)
                method = getattr(proxy, name)
                return await method(*args, **kwargs)
            
            return remote_call
    
    # Usage
    async with DynamicRemoteClient(config) as client:
        # These calls will be executed remotely
        result = await client.initialize("arg1", "arg2", test=True)
        data = await client.process_data([1, 2, 3, 4, 5])
        status = await client.get_status()


if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())
    
    # Uncomment to run the alternative syntax example
    # asyncio.run(alternative_syntax_example())