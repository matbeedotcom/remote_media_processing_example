"""
Minimal example - RemoteProxyClient works with ANY object!
"""

import asyncio
from remotemedia.core.node import RemoteExecutorConfig
from remotemedia.remote.proxy_client import RemoteProxyClient
from remotemedia.examples.objects import Calculator, SimpleCounter


async def main():
    # Setup connection
    config = RemoteExecutorConfig(host="localhost", port=50052, ssl_enabled=False)
    
    async with RemoteProxyClient(config) as client:
        # Example 1: Simple counter
        print("=== Any object becomes remote with ONE line ===")
        
        counter = SimpleCounter()
        remote = await client.create_proxy(counter)  # <-- That's it!
        
        print(f"Add 10: {await remote.add(10)}")
        print(f"Add 5: {await remote.add(5)}")
        
        # Example 2: Calculator  
        print("\n=== Works with any Python object ===")
        
        calc = Calculator()
        remote_calc = await client.create_proxy(calc)  # <-- One line!
        
        print(f"2 + 3 = {await remote_calc.add(2, 3)}")
        print(f"10 * 5 = {await remote_calc.multiply(10, 5)}")
        
        # The remote object maintains state
        await remote_calc.add(100)
        await remote_calc.add(50)
        print(f"History: {await remote_calc.history()}")
        
        # Example 3: More examples
        print("\n=== Works with any imported class ===")
        
        # You can also use any class from the examples
        from remotemedia.examples.objects import TodoList
        
        todos = TodoList()
        remote_todos = await client.create_proxy(todos)
        
        await remote_todos.add_todo("Learn RemoteProxyClient")
        await remote_todos.add_todo("Build amazing apps")
        await remote_todos.complete_todo(0)
        
        status = await remote_todos.get_status()
        print(f"Remote todo list: {status}")


if __name__ == "__main__":
    asyncio.run(main())