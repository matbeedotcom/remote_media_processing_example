# Custom RemoteMedia Execution Service Examples

This directory demonstrates multiple approaches for creating custom remote execution services using the RemoteMedia Processing SDK.

## 🚀 Quick Start

```bash
# Install dependencies (in production: pip install remote_media_processing)
pip install grpcio grpcio-tools cloudpickle numpy

# Choose your approach:
python filesystem_server.py    # 🌟 Recommended: Filesystem-based discovery
python server.py               # Simple: Manual node registry  
python advanced_server.py      # Advanced: Custom executor subclass
```

## 📁 Directory Structure

```
custom_remote_service/
├── nodes/                     # 🌟 Custom nodes (auto-discovered)
│   ├── timestamp_node.py      #     TimestampNode implementation
│   ├── math_processor_node.py #     MathProcessorNode implementation  
│   └── data_aggregator_node.py#     DataAggregatorNode implementation
├── pipelines/                 # 🌟 Custom pipelines (auto-discovered)
│   └── basic_pipelines.py     #     Example pipeline definitions
├── filesystem_server.py       # 🌟 Filesystem-based discovery server
├── server.py                  #     Simple manual registry server
├── advanced_server.py         #     Advanced custom executor server
├── test_filesystem_server.py  #     High-level client tests
├── discovery.py               #     Auto-discovery system
└── README.md                  #     This file
```

## 🌟 Approach 1: Filesystem-Based Discovery (Recommended)

**Most developer-friendly approach** - just drop files in directories!

### How it works:
- **Nodes**: Add `.py` files with `Node` subclasses to `nodes/`
- **Pipelines**: Add `.py` files with pipeline factories to `pipelines/`
- **Server**: Automatically discovers and registers everything

### Example:

```python
# nodes/my_custom_node.py
from remotemedia.core.node import Node

class MyCustomNode(Node):
    async def process(self, data):
        return {"processed": data, "by": "MyCustomNode"}
```

```bash
# Start server (auto-discovers MyCustomNode)
python filesystem_server.py
```

```python
# Use with high-level client
from remotemedia.remote import RemoteExecutionClient, RemoteExecutorConfig

config = RemoteExecutorConfig(host="localhost", port=50054)
client = RemoteExecutionClient(config)

result = await client.execute_node(
    node_type="MyCustomNode",
    input_data={"hello": "world"}
)
```

## 🔧 Approach 2: Manual Registry (Simple)

**Explicit control** - manually specify which nodes to register.

```python
# server.py
from custom_nodes import TimestampNode, MathProcessorNode

custom_nodes = {
    'TimestampNode': TimestampNode,
    'MathProcessorNode': MathProcessorNode,
}

await serve(custom_node_registry=custom_nodes)
```

## ⚡ Approach 3: Custom Executor (Advanced)

**Maximum control** - subclass TaskExecutor for advanced customization.

```python
# advanced_server.py  
class CustomTaskExecutor(TaskExecutor):
    def __init__(self, config):
        super().__init__(config, custom_nodes)
        self.custom_metrics = {}  # Add custom features
    
    async def execute_sdk_node(self, ...):
        # Add custom logic (metrics, logging, etc.)
        return await super().execute_sdk_node(...)

await serve(custom_executor=CustomTaskExecutor(config))
```

## 🧪 Testing Your Custom Server

```python
# High-level client (no gRPC boilerplate!)
from remotemedia.remote import RemoteExecutionClient, RemoteExecutorConfig

config = RemoteExecutorConfig(host="localhost", port=50054)
client = RemoteExecutionClient(config)

# Test your custom node
result = await client.execute_node(
    node_type="TimestampNode",
    config={"format": "iso"},
    input_data={"message": "Hello World"}
)

print(result)  # Clean, simple!
```

## 🎯 Key Features

### ✅ Clean Extension Mechanism
- **No monkey patching** - proper library extension points
- **No core modification** - never touch library files  
- **Backward compatible** - existing code continues to work

### ✅ Multiple Deployment Options
- **Filesystem discovery**: Drop files in directories (recommended)
- **Manual registry**: Explicit control over registered nodes
- **Custom executor**: Advanced customization and metrics

### ✅ Full Feature Preservation  
- **Complete gRPC API** - all endpoints work with custom nodes
- **Error handling** - comprehensive error handling and recovery
- **Session management** - automatic cleanup and resource management
- **Performance** - full optimization and sandboxing support
- **Monitoring** - built-in health checks and metrics

### ✅ Developer Experience
- **High-level client** - no gRPC boilerplate needed
- **Automatic discovery** - just drop files and go
- **Rich logging** - comprehensive development feedback
- **Easy testing** - clean client API for testing

## 🚢 Production Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 50054
CMD ["python", "filesystem_server.py"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  custom-remote-service:
    build: .
    ports: ["50054:50054"]
    environment:
      - GRPC_PORT=50054
      - LOG_LEVEL=INFO
```

## 🔍 Architecture Benefits

### Filesystem-Based Discovery
- **Convention over configuration** - standard directory layout
- **Hot reload potential** - could watch filesystem for changes  
- **Modular organization** - each node/pipeline in its own file
- **Easy collaboration** - team members add files independently

### Clean Extension Points
- **Library-native** - uses built-in extension mechanisms
- **Maintainable** - easy to update library versions
- **Testable** - clear separation of custom and core code
- **Flexible** - multiple approaches for different needs

### Production Ready
- **Docker support** - complete containerization examples
- **Monitoring** - comprehensive logging and metrics
- **Error handling** - graceful failure and recovery
- **Security** - sandboxed execution environment

## 📚 Examples in This Directory

| File | Description | Port | Approach |
|------|-------------|------|----------|
| `filesystem_server.py` | 🌟 Auto-discovery from directories | 50054 | Filesystem |
| `server.py` | Manual node registry | 50052 | Simple |  
| `advanced_server.py` | Custom executor subclass | 50053 | Advanced |

## 🎉 Success!

The RemoteMedia Processing library now provides a **clean, production-ready extension mechanism** for custom remote services. No monkey patching, no core modifications - just clean, maintainable code that preserves all existing functionality while enabling powerful customization.

Choose the approach that best fits your needs and start building! 🚀