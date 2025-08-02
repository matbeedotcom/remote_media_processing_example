"""
The simplest possible remote proxy example.
ANY object can be executed remotely with just ONE line!
"""

import asyncio
from remotemedia.core.node import RemoteExecutorConfig
from remotemedia.remote.proxy_client import RemoteProxyClient

# Import some example objects (can be ANY Python object)
from remotemedia.examples.objects import Calculator, TodoList, StringProcessor


async def main():
    # Connect to remote server
    config = RemoteExecutorConfig(host="localhost", port=50052, ssl_enabled=False)
    
    async with RemoteProxyClient(config) as client:
        
        # ========================================
        # THAT'S IT! Any object → Remote object
        # ========================================
        
        # Example 1: Calculator
        calc = Calculator()
        remote_calc = await client.create_proxy(calc)  # ← ONE LINE!
        
        print(f"5 + 3 = {await remote_calc.add(5, 3)}")
        print(f"7 * 8 = {await remote_calc.multiply(7, 8)}")
        
        # Example 2: Todo List  
        todos = TodoList()
        remote_todos = await client.create_proxy(todos)  # ← ONE LINE!
        
        await remote_todos.add_todo("Buy milk")
        await remote_todos.add_todo("Walk dog")
        await remote_todos.complete_todo(0)
        
        print(f"\nTodos: {await remote_todos.get_status()}")
        
        # Example 3: String Processor
        processor = StringProcessor()
        remote_proc = await client.create_proxy(processor)  # ← ONE LINE!
        
        print(f"\nReverse 'hello': {await remote_proc.reverse('hello')}")
        print(f"Process 'world': {await remote_proc.process('world', uppercase=True)}")


# Run it
asyncio.run(main())