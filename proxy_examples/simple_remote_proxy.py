"""
Simple example demonstrating transparent remote execution for any object.
"""

import asyncio
import numpy as np
from remotemedia.core.node import RemoteExecutorConfig
from remotemedia.remote.proxy_client import RemoteProxyClient
from remotemedia.examples.objects import (
    SimpleCounter, StringProcessor, MathOperations, 
    TodoList, Calculator
)


async def main():
    # Configure remote execution
    config = RemoteExecutorConfig(
        host="localhost",
        port=50052,
        ssl_enabled=False
    )
    
    async with RemoteProxyClient(config) as client:
        print("=== Example 1: Simple Counter ===")
        
        # Any object can be made remote
        counter = SimpleCounter()
        remote_counter = await client.create_proxy(counter)
        
        # Use it like normal
        print(f"Add 5: {await remote_counter.add(5)}")
        print(f"Add 10: {await remote_counter.add(10)}")
        
        print("\n=== Example 2: String Processor ===")
        
        # Make it remote
        processor = StringProcessor()
        remote_processor = await client.create_proxy(processor)
        
        # Call methods remotely
        print(f"Reverse 'hello': {await remote_processor.reverse('hello')}")
        print(f"Uppercase 'world': {await remote_processor.uppercase('world')}")
        print(f"Process 'python' (reverse + uppercase): {await remote_processor.process('python', reverse=True, uppercase=True)}")
        
        print("\n=== Example 3: Math Operations ===")
        
        # Make it remote
        math_ops = MathOperations()
        remote_math = await client.create_proxy(math_ops)
        
        # Execute calculations remotely
        print(f"Calculate '2 + 2': {await remote_math.calculate('2 + 2')}")
        print(f"Calculate '10 * 5': {await remote_math.calculate('10 * 5')}")
        print(f"Calculate '100 / 4': {await remote_math.calculate('100 / 4')}")
        
        # Get history
        history = await remote_math.get_history()
        print(f"Calculation history: {history}")
        
        # Even works with numpy arrays
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = await remote_math.matrix_multiply(a, b)
        print(f"Matrix multiplication result:\n{result}")
        
        print("\n=== Example 4: Stateful Object ===")
        
        # Make it remote
        todo_list = TodoList()
        remote_todos = await client.create_proxy(todo_list)
        
        # Use it remotely
        print(await remote_todos.add_todo("Write documentation"))
        print(await remote_todos.add_todo("Fix bugs"))
        print(await remote_todos.add_todo("Add tests"))
        
        print(await remote_todos.complete_todo(0))
        
        status = await remote_todos.get_status()
        print(f"Todo status: {status}")
        
        print("\n=== Example 5: Any Python Object ===")
        
        # Even built-in Python objects work
        import collections
        
        # Create a deque remotely
        deque = collections.deque(maxlen=5)
        remote_deque = await client.create_proxy(deque)
        
        # Use it remotely
        await remote_deque.append(1)
        await remote_deque.append(2)
        await remote_deque.append(3)
        await remote_deque.appendleft(0)
        
        # Can't directly print the deque, but we can convert to list
        # This shows we can even work with built-in Python objects
        
        # Or use the Calculator
        calc = Calculator()
        remote_calc = await client.create_proxy(calc)
        
        print(f"5 + 3 = {await remote_calc.add(5, 3)}")
        print(f"10 * 4 = {await remote_calc.multiply(10, 4)}")
        await remote_calc.add(100)  # Adds to internal result
        await remote_calc.add(50)   # Adds to internal result
        
        history = await remote_calc.history()
        print(f"Calculator history: {history}")


if __name__ == "__main__":
    asyncio.run(main())