"""
Ultra-simple remote proxy example - works with ANY object.
"""

import asyncio
from remotemedia.core.node import RemoteExecutorConfig
from remotemedia.remote.proxy_client import RemoteProxyClient


async def main():
    # Import example objects to avoid pickling issues with __main__
    from remotemedia.examples.objects import Calculator, StringProcessor
    
    # Connect to remote server
    config = RemoteExecutorConfig(host="localhost", port=50052, ssl_enabled=False)
    
    async with RemoteProxyClient(config) as client:
        # Example 1: Calculator
        calc = Calculator()
        remote_calc = await client.create_proxy(calc)
        
        print(f"5 + 10 = {await remote_calc.add(5, 10)}")
        print(f"7 * 3 = {await remote_calc.multiply(7, 3)}")
        await remote_calc.reset()
        print(f"Reset: {await remote_calc.history()}")
        
        # Example 2: String Processor
        processor = StringProcessor()
        remote_proc = await client.create_proxy(processor)
        
        print(f"\nReverse 'hello': {await remote_proc.reverse('hello')}")
        print(f"Uppercase 'world': {await remote_proc.uppercase('world')}")
        print(f"Process 'python': {await remote_proc.process('python', reverse=True, uppercase=True)}")


if __name__ == "__main__":
    # Import these here to avoid pickling issues
    from remotemedia.examples.objects import SimpleCounter
    
    async def quick_demo():
        config = RemoteExecutorConfig(host="localhost", port=50052, ssl_enabled=False)
        async with RemoteProxyClient(config) as client:
            # ONE LINE to make any object remote!
            counter = await client.create_proxy(SimpleCounter())
            
            # Use it normally (just remember to await)
            print(f"Counter: {await counter.add(42)}")
    
    print("=== Quick Demo ===")
    asyncio.run(quick_demo())
    
    print("\n=== Main Examples ===")
    asyncio.run(main())