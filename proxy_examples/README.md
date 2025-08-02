# Remote Proxy Examples

This directory contains examples demonstrating the RemoteProxyClient for transparent remote execution of Python objects.

## Examples

### Basic Proxy Usage
- `minimal_proxy.py` - Minimal example showing basic proxy usage
- `simplest_proxy.py` - Simplest possible proxy implementation
- `ultra_simple_proxy.py` - Ultra-simple proxy demonstration

### Advanced Proxy Features
- `simple_remote_proxy.py` - More complete proxy example with various object types
- `remote_proxy_example.py` - Full-featured proxy example with counters and data processors

### Generator and Streaming
- `generator_streaming_comparison.py` - Comparison of different generator streaming approaches
- `streaming_solution.py` - Complete streaming solution with generators

## Running the Examples

Make sure the remote service is running:
```bash
cd ../remote_service
docker-compose up
```

Then run any example:
```bash
python minimal_proxy.py
```

## Key Concepts

The RemoteProxyClient allows you to:
- Execute any Python object remotely without modification
- Maintain object state across method calls
- Stream generator results efficiently
- Handle both sync and async methods transparently