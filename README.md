# Remote Media Processing Test Project

A real-time speech-to-speech pipeline using WebRTC, voice activity detection (VAD), speech recognition (Ultravox), and text-to-speech synthesis (Kokoro TTS).

## Overview

This project demonstrates a complete audio processing pipeline that:
- Receives audio streams via WebRTC
- Detects speech segments using VAD
- Transcribes speech to text using Ultravox
- Synthesizes responses using Kokoro TTS
- Streams audio back to the client in real-time

## Features

- **Real-time WebRTC streaming** - Low-latency audio/video communication
- **Voice Activity Detection** - Intelligent speech/silence detection
- **Speech Recognition** - Ultravox model for accurate transcription
- **Text-to-Speech** - High-quality Kokoro TTS synthesis
- **Streaming architecture** - Processes audio in chunks for responsiveness
- **Remote execution** - Supports distributed processing via RemoteMedia framework

## Installation

### Prerequisites

- Python 3.8+
- Linux/macOS (for espeak-ng support)
- RemoteMedia framework

### Basic Installation

```bash
# Clone the repository
git clone <repository-url>
cd remote_media_processing_test_project

# Install WebRTC dependencies
pip install aiortc aiohttp aiohttp-cors

# Install TTS dependencies
pip install kokoro>=0.9.4 soundfile

# Install espeak (required for Kokoro TTS)
# Ubuntu/Debian:
sudo apt-get install espeak-ng
# macOS:
brew install espeak
```

### ML Dependencies (Optional)

For speech recognition features:
```bash
pip install -r requirements-ml.txt
```

## Quick Start

### 1. Start the Remote Service (if using remote execution)
```bash
PYTHONPATH=. python remote_service/src/server.py
```

### 2. Run the WebRTC Server
```bash
# Basic server
python webrtc_pipeline_server.py

# With ML features enabled
USE_ML=true python webrtc_pipeline_server.py
```

### 3. Connect a Client
Open your browser and navigate to:
```
http://localhost:8080/webrtc_client.html
```

## Usage Examples

### WebRTC Server with Custom Configuration
```bash
# Custom host and port
SERVER_HOST=0.0.0.0 SERVER_PORT=8081 python webrtc_pipeline_server.py

# Connect to remote ML service
REMOTE_HOST=192.168.1.100 USE_ML=true python webrtc_pipeline_server.py
```

### Standalone Pipeline Testing
```bash
# Test the speech-to-speech pipeline with a local audio file
python vad_ultravox_kokoro_streaming.py
```

## Architecture

### Pipeline Flow

```
Audio Input (WebRTC/File)
    ↓
Audio Transform (16kHz, mono)
    ↓
Voice Activity Detection
    ↓
VAD-Triggered Buffer
    ↓
Speech Recognition (Ultravox)
    ↓
Text-to-Speech (Kokoro TTS)
    ↓
Audio Output (WebRTC/File)
```

### Key Components

- **webrtc_pipeline_server.py** - WebRTC server with pipeline integration
- **vad_ultravox_nodes.py** - Custom pipeline nodes for VAD and buffering
- **kokoro_tts.py** - TTS synthesis node wrapper
- **vad_ultravox_kokoro_streaming.py** - Standalone pipeline example

### Configuration Parameters

- **VAD Settings**:
  - Frame duration: 30ms
  - Energy threshold: 0.02
  - Speech threshold: 0.3
  - Minimum speech duration: 1.0s
  - Silence duration for end detection: 0.5s

- **Audio Settings**:
  - Processing sample rate: 16kHz
  - TTS output sample rate: 24kHz
  - Pre-speech buffer: 1.0s

## API Endpoints

- WebSocket signaling: `ws://localhost:8080/ws`
- Health check: `http://localhost:8080/health`
- Active connections: `http://localhost:8080/connections`
- Web client: `http://localhost:8080/webrtc_client.html`

## Output

- **WebRTC mode**: Audio streamed directly to connected clients
- **Standalone mode**: Audio files saved to `generated_responses/` directory

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `USE_ML` | Enable ML features | `false` |
| `REMOTE_HOST` | Remote service host | `127.0.0.1` |
| `SERVER_HOST` | WebRTC server host | `0.0.0.0` |
| `SERVER_PORT` | WebRTC server port | `8080` |

## Troubleshooting

### Common Issues

1. **ML dependencies not found**
   - Install with: `pip install -r requirements-ml.txt`
   - Set `USE_ML=true` when running

2. **Kokoro TTS errors**
   - Ensure espeak is installed: `sudo apt-get install espeak-ng`
   - Verify Kokoro version: `pip install kokoro>=0.9.4`

3. **Remote service connection failed**
   - Start remote service first: `PYTHONPATH=. python remote_service/src/server.py`
   - Check `REMOTE_HOST` environment variable

4. **WebRTC connection issues**
   - Ensure ports are not blocked by firewall
   - Try different STUN servers if needed

## Development

### Running Tests
```bash
# Run unit tests (if available)
python -m pytest tests/

# Test with example audio
python vad_ultravox_kokoro_streaming.py
```

### Debugging
Enable debug logging:
```bash
export PYTHONPATH=.
export LOG_LEVEL=DEBUG
python webrtc_pipeline_server.py
```

## License

This is a test project. Please refer to the license file for details.

## Contributing

This is a test/example project. Feel free to fork and modify for your own use cases.

## Acknowledgments

- RemoteMedia framework for distributed processing
- Ultravox for speech recognition
- Kokoro TTS for speech synthesis
- aiortc for WebRTC implementation