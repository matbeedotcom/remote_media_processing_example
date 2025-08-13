# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains examples demonstrating the RemoteMedia Processing SDK - a framework for distributed computing, real-time media processing, and transparent remote execution of Python objects. The SDK enables running any Python object remotely without modification.

## Key Commands

### Setup and Installation
```bash
# Install the RemoteMedia SDK and dependencies
pip install -e .

# For WebRTC examples with browser clients
pip install aiortc aiohttp

# For audio processing examples
pip install git+https://github.com/remoteMedia/ultravoxLib@nemo-asr
pip install git+https://github.com/remoteMedia/kokoroLib.git
```

### Running Examples

```bash
# Start the RemoteMedia service (required for all examples)
remotemedia_service

# Run proxy examples (transparent remote execution)
python proxy_examples/0_minimal_proxy.py
python proxy_examples/1_with_pip_packages.py

# Run audio processing pipelines
python audio_examples/vad.py
python audio_examples/transcribe.py
python audio_examples/tts.py

# Run WebRTC examples (requires browser client)
python webrtc_examples/pipeline.py
# Then open http://localhost:8080 in browser

# Run VLBI optical telescope system
python webrtc_examples/vlbi_2024/pipeline.py
# Access at http://localhost:8084

# Run ChAruco camera calibration (full pipeline)
python webrtc_examples/charuco/examples/full_calibration_example.py --cameras 0 1 2 3

# Generate ChAruco board for printing
python webrtc_examples/charuco/examples/full_calibration_example.py --generate-board

# Run WebRTC ChAruco calibration server
python webrtc_examples/charuco/examples/webrtc_server.py --port 8081
```

### Testing
```bash
# No test suite currently exists
# Examples serve as integration tests - run each to verify functionality
```

## Architecture

### Core Concepts

1. **RemoteMedia Service**: Central daemon that manages remote execution
   - Started via `remotemedia_service` command
   - Handles transparent proxying of Python objects
   - Manages node and pipeline discovery

2. **Node-Based Processing**: All processing units inherit from base Node class
   - Must implement async `process()` method
   - Nodes are automatically discovered via filesystem scanning
   - Located in `nodes/` directories within each example category

3. **Pipeline Composition**: Complex workflows built from connected nodes
   - Pipelines defined in `pipelines/` directories
   - Support for WebRTC integration and browser clients
   - Real-time processing with <100ms latency targets

### Directory Structure

```
├── proxy_examples/          # Transparent remote execution demos
│   ├── 0_minimal_proxy.py  # Simplest example
│   └── 1_with_pip_packages.py # With auto-installation
│
├── audio_examples/          # Audio/speech processing
│   ├── nodes/              # Audio processing nodes
│   ├── pipelines/          # Audio pipeline definitions
│   └── *.py                # Entry points
│
├── webrtc_examples/        # Real-time WebRTC
│   ├── charuco/           # ChAruco camera calibration system
│   │   ├── calibration/   # Calibration pipeline modules
│   │   ├── nodes/         # Processing nodes (detection, warping, etc.)
│   │   ├── utils/         # Utility modules and tools
│   │   ├── config/        # Configuration files
│   │   ├── examples/      # Example scripts and WebRTC server
│   │   ├── vlbi/          # VLBI interferometry modules
│   │   └── README.md      # Complete documentation
│   ├── nodes/             # WebRTC processing nodes
│   ├── utils/             # WebRTC utility modules
│   ├── debug/             # Debug scripts and tools
│   ├── tests/             # Test scripts
│   ├── examples/          # Example applications and HTML clients
│   ├── raspberrypi_webrtc_client/  # RaspberryPi client implementation
│   ├── vlbi_2024/         # Optical telescope interferometry
│   ├── webrtc_ultravox_pipeline_server.py  # Audio processing server
│   └── webrtc_vlbi_pipeline_server.py      # VLBI processing server
│
├── remote_class_execution_with_pip_packages/  # Advanced remote execution
│   └── nodes/             # Example nodes with dependencies
│
└── custom_remote_service/  # Three approaches to extend SDK
    ├── a_monkey_patch/    # Quick but not recommended
    ├── b_custom_container/# Docker-based extension
    └── c_inherited_local_service/ # Clean inheritance approach
```

### Key Files and Patterns

**Node Implementation Pattern**:
```python
# nodes/*/[name]_node.py
class MyNode:
    async def process(self, data):
        # Processing logic here
        return processed_data
```

**Pipeline Definition**:
```python
# pipelines/*/[name]_pipeline.py
class MyPipeline:
    def __init__(self):
        self.nodes = [Node1(), Node2()]
    
    async def process(self, data):
        for node in self.nodes:
            data = await node.process(data)
        return data
```

**WebRTC Integration** (webrtc_examples/clients/):
- Browser clients connect via WebSocket signaling
- Video/audio streams processed through node pipeline
- Results streamed back in real-time

## Important Environment Variables

```bash
# RemoteMedia service configuration
REMOTEMEDIA_HOST=localhost
REMOTEMEDIA_PORT=50051

# WebRTC server ports (per example)
WEBRTC_PORT=8080  # Default WebRTC example
VLBI_PORT=8084    # VLBI telescope system
```

## VLBI Optical Telescope System

The `webrtc_examples/vlbi_2024/` directory contains a research-grade Very Long Baseline Interferometry system:
- 4-camera array with ChAruco calibration
- Sub-milliarcsecond angular resolution
- Real-time interferometric processing
- Professional astrophotography capabilities

Key components:
- `calibration/charuco_calibration.py`: Camera calibration
- `pipeline.py`: Main VLBI processing pipeline
- `nodes/interferometry_node.py`: Core interferometric algorithms

## Extension Patterns

To add custom functionality to RemoteMedia service, use approach C (inheritance):

```python
# custom_remote_service/c_inherited_local_service/
class CustomLocalService(LocalService):
    def custom_method(self):
        # Your custom logic
        pass
```

Avoid monkey patching (approach A) in production code.

## Working with WebRTC Examples

1. Start the pipeline server: `python webrtc_examples/pipeline.py`
2. Open browser to `http://localhost:8080`
3. Browser client code is in `webrtc_examples/clients/`
4. Modify nodes in `webrtc_examples/nodes/` for custom processing

## ChAruco Camera Calibration System

The `webrtc_examples/charuco/` directory contains a comprehensive multi-camera calibration system:

### Features
- **Individual Camera Calibration**: Full-screen calibration for each camera independently
- **Multi-Camera Calibration**: Stereo and multi-view calibration for camera arrays
- **Real-time Visualization**: Live preview during calibration process
- **Physical Units**: Calibration with real-world measurements (mm, degrees, arcseconds)
- **VLBI Support**: Integration with optical Very Long Baseline Interferometry systems

### Quick Start
```bash
# Generate ChAruco board for printing
python webrtc_examples/charuco/examples/full_calibration_example.py --generate-board

# Run full calibration pipeline
python webrtc_examples/charuco/examples/full_calibration_example.py --cameras 0 1 2 3

# Or use the WebRTC server for real-time calibration
python webrtc_examples/charuco/examples/webrtc_server.py --port 8081
```

### Calibration Process
1. **Individual Phase**: Show ChAruco board to each camera individually for full-screen coverage
2. **Multi-Camera Phase**: Show board visible to multiple cameras simultaneously for stereo calibration

### Key Components
- `calibration/`: Pipeline modules (individual, multi-camera, full pipeline)
- `nodes/`: Processing nodes (detection, warping, preview, etc.)
- `utils/`: Utility modules (board generation, testing, sensor config)
- `config/`: JSON configuration files for boards and sensors
- `examples/`: Complete examples and WebRTC server
- `vlbi/`: VLBI-specific interferometry modules

See `webrtc_examples/charuco/README.md` for complete documentation.

## Audio Processing Pipeline Notes

- **Ultravox ASR**: Requires nemo-asr branch from remoteMedia fork
- **Kokoro TTS**: High-quality text-to-speech synthesis
- **VAD**: Voice activity detection for audio segmentation
- Pipelines can be chained: VAD → Transcribe → TTS

## Development Tips

- Always start `remotemedia_service` before running examples
- Nodes are auto-discovered - just place in `nodes/` directory
- Use async/await patterns consistently in node processing
- WebRTC examples require modern browser with getUserMedia support
- VLBI system requires 4 calibrated cameras for full functionality