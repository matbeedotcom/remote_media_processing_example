# Remote Class Execution Demo

This example demonstrates how to execute any Python class or instance remotely using the remote_media framework.

## Overview

The remote_media framework allows you to take any Python class and execute its methods on a remote server transparently. This is useful for:

- Offloading compute-intensive operations to powerful servers
- Running ML models on GPU-equipped machines
- Distributed processing across multiple servers
- Isolating untrusted code execution
- Scaling applications horizontally

## Files

- `sample_classes.py` - Example classes that demonstrate various patterns:
  - `DataProcessor` - Shows sync/async methods and state management
  - `ScientificCalculator` - Demonstrates complex computations and history tracking
  - `StreamingService` - Shows generator methods (currently materialized)
  - `StatefulService` - Demonstrates persistent state across method calls

- `remote_execution_client.py` - Client script that demonstrates:
  - Creating remote proxy instances
  - Calling methods remotely
  - Managing state across calls
  - Working with multiple instances
  - Handling async operations

- `demo_with_pip_packages.py` - Comprehensive demo of pip package dependencies:
  - Installing and using numpy, pandas, scipy, etc.
  - Multiple packages working together
  - Error handling and validation

- `simple_pip_example.py` - Minimal example showing pip package usage

- `test_*.py` - Various test files for edge cases and validation

## Prerequisites

1. Install the remote_media package:
   ```bash
   cd ../..
   pip install -e .
   ```

2. Install demo requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the remote execution server:
   ```bash
   cd ../../remote_service
   python src/server.py
   ```

   Or using Docker:
   ```bash
   cd ../../remote_service
   docker-compose up
   ```

## Running the Demos

### Basic Demo (without pip packages)
```bash
python remote_execution_client.py
```

The demo will walk you through various examples:

1. **DataProcessor Demo** - Simple and async method execution
2. **ScientificCalculator Demo** - Complex calculations and matrix operations
3. **StreamingService Demo** - Data streaming (currently materialized to lists)
4. **StatefulService Demo** - Persistent state management
5. **Multiple Instances Demo** - Parallel processing with multiple remote objects

### Pip Package Dependencies Demo
```bash
# Simple example
python simple_pip_example.py

# Comprehensive demo with numpy, pandas, scipy, etc.
python demo_with_pip_packages.py

# Test external packages not in conda
python test_external_packages.py

# Test complex packages with dependencies
python test_complex_packages.py
```

These demos show:
- Automatic package installation on remote server
- Using scientific computing libraries remotely
- Web scraping with beautifulsoup4 and requests
- Image processing with PIL/Pillow
- Data visualization with matplotlib
- And much more!

## Key Concepts

### Remote Proxy Creation

```python
from remotemedia.core.node import RemoteExecutorConfig
from remotemedia.remote.proxy_client import RemoteProxyClient

# Configure connection
config = RemoteExecutorConfig(host="localhost", port=50052, ssl_enabled=False)

# Create proxy client
async with RemoteProxyClient(config) as client:
    # Create local instance
    my_object = MyClass()
    
    # Create remote proxy
    remote_object = await client.create_proxy(my_object)
    
    # Use remote object like local one
    result = await remote_object.my_method(args)
```

### Using Pip Package Dependencies

```python
# Specify packages your remote code needs
config = RemoteExecutorConfig(
    host="localhost", 
    port=50052,
    pip_packages=["numpy", "pandas", "requests"]  # Automatically installed!
)

class DataAnalyzer:
    def analyze(self, data):
        import pandas as pd  # This works because pandas was installed
        import numpy as np   # This too!
        
        df = pd.DataFrame(data)
        return {
            "mean": df.mean().to_dict(),
            "correlation": df.corr().to_dict()
        }

async with RemoteProxyClient(config) as client:
    analyzer = DataAnalyzer()
    remote_analyzer = await client.create_proxy(analyzer)
    
    # Packages are available on the remote server
    results = await remote_analyzer.analyze(my_data)
```

### Method Execution

- All methods are executed on the remote server
- Method calls are automatically serialized and sent over gRPC
- Results are deserialized and returned to the client
- Both sync and async methods are supported

### State Management

- Object state is maintained on the remote server
- Multiple method calls on the same proxy instance share state
- Each proxy instance is independent

### Current Limitations

- Generator methods are materialized to lists before returning
- Large data transfers may impact performance
- Network latency affects all method calls

## Extending the Example

To add your own classes:

1. Create your class in a new file or add to `sample_classes.py`
2. Import it in `remote_execution_client.py`
3. Create a demo function following the existing patterns
4. Add it to the `demos` list in `main()`

## Troubleshooting

- **Connection refused**: Make sure the remote server is running
- **Import errors**: Ensure remotemedia package is installed
- **Serialization errors**: Check that all method arguments and return values are serializable