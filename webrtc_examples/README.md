# WebRTC Examples

This directory contains WebRTC real-time communication examples.

## Examples

### WebRTC Server
- `webrtc_pipeline_server.py` - WebRTC server with audio processing pipeline
- `webrtc_client.html` - Web client for testing WebRTC connections
- `vad_ultravox_nodes.py` - Voice Activity Detection with Ultravox speech recognition

## Running the WebRTC Server

1. Install dependencies:
```bash
pip install -e ".[ml]"  # For ML features
```

2. Start the server:
```bash
# Basic server (no ML)
python webrtc_pipeline_server.py

# With ML features (speech recognition & TTS)
USE_ML=true python webrtc_pipeline_server.py
```

3. Open the client in a browser:
```
http://localhost:8080/webrtc_client.html
```

## Features

The WebRTC pipeline supports:
- Real-time audio streaming
- Voice Activity Detection (VAD)
- Speech-to-text (Ultravox)
- Text-to-speech (Kokoro TTS)
- Full duplex communication