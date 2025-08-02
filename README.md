# RemoteMedia Processing Examples

A comprehensive collection of examples demonstrating the RemoteMedia Processing SDK capabilities, including distributed processing, real-time audio/video handling, and transparent remote execution.

## Directory Structure

```
remote_media_processing_example/
├── proxy_examples/          # Remote proxy and transparent execution examples
├── audio_examples/          # Audio processing, speech recognition, and TTS
├── webrtc_examples/         # Real-time WebRTC communication examples
├── remote_class_execution_demo/  # Advanced remote class execution demos
├── assets/                  # Resource files
│   ├── audio/              # Sample audio files
│   └── generated/          # Generated output files
└── requirements.txt        # Python dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
# Basic installation
pip install -r requirements.txt

# With ML features (speech recognition, TTS)
pip install -e ".[ml]"
```

### 2. Start Remote Service (Required for Remote Examples)

```bash
cd ../remote_media_processing/remote_service
docker-compose up
```

### 3. Run Examples

Choose from the categorized examples below.

## Example Categories

### 🔌 Remote Proxy Examples (`proxy_examples/`)

Demonstrate transparent remote execution of Python objects without modification.

**Basic Examples:**
- `minimal_proxy.py` - Minimal proxy usage
- `simplest_proxy.py` - Simplest implementation
- `ultra_simple_proxy.py` - Ultra-simple demonstration

**Advanced Examples:**
- `simple_remote_proxy.py` - Various object types
- `remote_proxy_example.py` - Full-featured with counters and processors
- `generator_streaming_comparison.py` - Generator streaming approaches
- `streaming_solution.py` - Complete streaming with generators

```bash
cd proxy_examples
python minimal_proxy.py
```

### 🎤 Audio Processing Examples (`audio_examples/`)

Speech recognition, text-to-speech, and audio processing pipelines.

**Examples:**
- `kokoro_tts.py` - Text-to-Speech synthesis
- `vad_ultravox_nodes.py` - Voice Activity Detection + Speech Recognition
- `vad_ultravox_kokoro_streaming.py` - Complete speech-to-speech pipeline

```bash
cd audio_examples
python vad_ultravox_kokoro_streaming.py
```

### 🌐 WebRTC Examples (`webrtc_examples/`)

Real-time audio/video communication with WebRTC.

**Components:**
- `webrtc_pipeline_server.py` - WebRTC server with audio pipeline
- `webrtc_client.html` - Browser-based client

```bash
cd webrtc_examples
# Basic server
python webrtc_pipeline_server.py

# With ML features
USE_ML=true python webrtc_pipeline_server.py
```

Open browser: `http://localhost:8080/webrtc_client.html`

### 🚀 Advanced Remote Execution (`remote_class_execution_demo/`)

Advanced demonstrations of remote class execution with pip package installation.

**Examples:**
- `demo_with_pip_packages.py` - Remote execution with automatic pip installs
- `simple_pip_example.py` - Basic pip package usage
- Various test scenarios for edge cases

```bash
cd remote_class_execution_demo
./run_demo.sh
```

## Key Features Demonstrated

### 1. **Transparent Remote Execution**
- Execute any Python object remotely without modification
- Maintain object state across method calls
- Support for sync/async methods and generators

### 2. **Audio/Speech Processing**
- Real-time Voice Activity Detection (VAD)
- Speech-to-Text with Ultravox
- Text-to-Speech with Kokoro TTS
- Streaming audio pipelines

### 3. **WebRTC Integration**
- Low-latency audio/video streaming
- Browser-based clients
- Real-time processing pipelines

### 4. **Package Management**
- Automatic pip package installation
- Virtual environment isolation
- Dependency resolution

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `USE_ML` | Enable ML features (requires ML deps) | `false` |
| `REMOTE_HOST` | Remote service host | `localhost` |
| `REMOTE_PORT` | Remote service port | `50052` |
| `SERVER_HOST` | WebRTC server host | `0.0.0.0` |
| `SERVER_PORT` | WebRTC server port | `8080` |

## Requirements

### System Requirements
- Python 3.8+
- Linux/macOS (for audio processing)
- Docker (for remote service)

### Audio Processing Requirements
```bash
# Ubuntu/Debian
sudo apt-get install espeak-ng

# macOS
brew install espeak
```

## Troubleshooting

### Remote Service Issues
- Ensure Docker is running
- Check port 50052 is available
- Verify service logs: `docker-compose logs -f`

### Audio/ML Issues
- Install ML dependencies: `pip install -e ".[ml]"`
- Ensure espeak is installed for TTS
- Check GPU availability for ML models

### WebRTC Issues
- Use Chrome/Firefox for best compatibility
- Check firewall settings for ports 8080
- Enable microphone permissions in browser

## License

See the main project LICENSE file for details.

## Contributing

These examples are designed to demonstrate SDK capabilities. Feel free to adapt them for your use cases.