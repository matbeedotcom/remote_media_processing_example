#!/usr/bin/env python3
"""
WebRTC Server with Pipeline Integration Example.

This example demonstrates how to create a WebRTC server that processes
audio and video streams through RemoteMedia pipelines in real-time.

**Features:**
- Real-time audio processing with VAD and speech recognition
- Video processing capabilities
- Data channel communication
- Multiple concurrent WebRTC connections
- Integration with remote execution services
- Pipeline Registry integration for discovery and reuse
- Persistent pipeline storage across server restarts
- JavaScript client access to registered pipelines

**TO RUN THIS EXAMPLE:**

1.  **Install WebRTC dependencies:**
    $ pip install aiortc aiohttp aiohttp-cors

2.  **Install ML dependencies (optional):**
    $ pip install -r requirements-ml.txt

3.  **Start the remote service (if using remote nodes):**
    $ PYTHONPATH=. python remote_service/src/server.py

4.  **Run the WebRTC server:**
    $ python examples/webrtc_pipeline_server.py

5.  **Connect with a WebRTC client:**
    - Open a WebRTC client application
    - Connect to ws://localhost:8080/ws for signaling
    - The server will process your audio/video streams through the pipeline

**Pipeline Registry Integration:**

The server automatically registers the WebRTC pipeline with the PipelineRegistry,
making it discoverable and executable by JavaScript clients:

    $ node nodejs-client/examples/discover-webrtc-pipeline.js

Environment variables:
- ENABLE_PERSISTENCE=true/false (default: true) - Enable pipeline persistence
- USE_ML=true/false (default: false) - Enable ML features  
- REMOTE_HOST=hostname (default: 127.0.0.1) - Remote execution host
- SERVER_HOST=hostname (default: 0.0.0.0) - WebRTC server host
- SERVER_PORT=port (default: 8080) - WebRTC server port

**Example Client HTML:**
Save this as 'client.html' and open in a browser:

```html
<!DOCTYPE html>
<html>
<head>
    <title>WebRTC Pipeline Client</title>
</head>
<body>
    <h1>WebRTC Pipeline Client</h1>
    <video id="localVideo" autoplay muted width="320" height="240"></video>
    <video id="remoteVideo" autoplay width="320" height="240"></video>
    <br>
    <button id="startBtn">Start Call</button>
    <button id="stopBtn" disabled>Stop Call</button>
    <br>
    <div id="status">Disconnected</div>
    <div id="messages"></div>

    <script>
        const localVideo = document.getElementById('localVideo');
        const remoteVideo = document.getElementById('remoteVideo');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const status = document.getElementById('status');
        const messages = document.getElementById('messages');

        let pc = null;
        let ws = null;
        let localStream = null;

        startBtn.onclick = start;
        stopBtn.onclick = stop;

        async function start() {
            // Get user media
            localStream = await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: true
            });
            localVideo.srcObject = localStream;

            // Create peer connection
            pc = new RTCPeerConnection({
                iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
            });

            // Add tracks
            localStream.getTracks().forEach(track => {
                pc.addTrack(track, localStream);
            });

            // Handle remote stream
            pc.ontrack = (event) => {
                remoteVideo.srcObject = event.streams[0];
            };

            // Create data channel
            const dataChannel = pc.createDataChannel('messages');
            dataChannel.onopen = () => {
                console.log('Data channel opened');
                dataChannel.send('Hello from client!');
            };
            dataChannel.onmessage = (event) => {
                messages.innerHTML += '<div>Received: ' + event.data + '</div>';
            };

            // Connect WebSocket
            ws = new WebSocket('ws://localhost:8080/ws');
            ws.onopen = () => {
                status.textContent = 'Connected';
            };
            ws.onmessage = async (event) => {
                const message = JSON.parse(event.data);
                if (message.type === 'answer') {
                    await pc.setRemoteDescription(new RTCSessionDescription(message));
                }
            };

            // Create offer
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);
            ws.send(JSON.stringify({
                type: 'offer',
                sdp: offer.sdp
            }));

            startBtn.disabled = true;
            stopBtn.disabled = false;
        }

        async function stop() {
            if (localStream) {
                localStream.getTracks().forEach(track => track.stop());
            }
            if (pc) {
                pc.close();
            }
            if (ws) {
                ws.close();
            }
            status.textContent = 'Disconnected';
            startBtn.disabled = false;
            stopBtn.disabled = true;
        }
    </script>
</body>
</html>
```
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional
import datetime
import json
import random
import math

sys.path.insert(0, str(Path(__file__).parent.parent))

from remotemedia.core.pipeline import Pipeline
from remotemedia.core.node import RemoteExecutorConfig, Node
from remotemedia.core.pipeline_registry import PipelineRegistry
from remotemedia.remote.client import RemoteExecutionClient
import grpc
from remotemedia.nodes.audio import AudioTransform, VoiceActivityDetector
from remotemedia.nodes.ml import UltravoxNode
from remotemedia.nodes.transform import TextTransformNode
from remotemedia.nodes.remote import RemoteObjectExecutionNode
from remotemedia.nodes import PassThroughNode
from remotemedia.webrtc import WebRTCServer, WebRTCConfig
from remotemedia.persistence import AccessLevel

# Import from parent directory audio_examples
sys.path.insert(0, str(Path(__file__).parent.parent / "audio_examples"))
from kokoro_tts import KokoroTTSNode

# Import from current directory
from vad_ultravox_nodes import (
    VADTriggeredBuffer, UltravoxMinDurationWrapper, AudioOutputNode, 
    TextLoggingNode, VADLoggingNode, UltravoxImmediateProcessor, MessageLoggingNode
)
import numpy as np
from typing import AsyncGenerator, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce noise from some loggers
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)
logging.getLogger("remotemedia.core.pipeline").setLevel(logging.WARNING)  # Reduce pipeline worker noise
logging.getLogger("Pipeline").setLevel(logging.WARNING)  # Class-based logger name

# Also suppress after import
import logging
logging.getLogger("Pipeline").setLevel(logging.ERROR)
logging.getLogger("remotemedia.core.pipeline").setLevel(logging.ERROR)


# Tool implementations
async def get_current_time(timezone: str = "UTC", format: str = "24h") -> str:
    """
    Get the current time in the specified timezone.
    
    Args:
        timezone: The timezone (e.g., "UTC", "EST", "PST", "GMT")
        format: Time format - "24h" or "12h"
    
    Returns:
        Current time as a string
    """
    now = datetime.datetime.now()
    if format == "12h":
        time_str = now.strftime("%I:%M:%S %p")
    else:
        time_str = now.strftime("%H:%M:%S")
    
    date_str = now.strftime("%Y-%m-%d")
    return f"The current time is {time_str} on {date_str} ({timezone})"


async def get_weather(location: str, unit: str = "celsius") -> str:
    """
    Get the current weather for a location.
    
    Args:
        location: City name or location
        unit: Temperature unit - "celsius" or "fahrenheit"
    
    Returns:
        Weather information as a string
    """
    # This is a mock implementation - in production, you'd call a real weather API
    weather_conditions = ["sunny", "partly cloudy", "cloudy", "rainy", "stormy"]
    condition = random.choice(weather_conditions)
    
    if unit == "celsius":
        temp = random.randint(15, 30)
        unit_symbol = "°C"
    else:
        temp = random.randint(60, 85)
        unit_symbol = "°F"
    
    humidity = random.randint(40, 80)
    wind_speed = random.randint(5, 25)
    
    return (f"Weather in {location}: {condition}, {temp}{unit_symbol}, "
            f"humidity {humidity}%, wind {wind_speed} km/h")


async def calculate(expression: str) -> str:
    """
    Perform a mathematical calculation.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2", "sqrt(16)", "pi * 4")
    
    Returns:
        The result of the calculation
    """
    try:
        # Create a safe namespace with math functions
        safe_dict = {
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'pi': math.pi,
            'e': math.e,
            'log': math.log,
            'log10': math.log10,
            'pow': pow,
            'abs': abs,
            'round': round,
            'floor': math.floor,
            'ceil': math.ceil
        }
        
        # Evaluate the expression safely
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


async def search_web(query: str, num_results: int = 3) -> str:
    """
    Search the web for information.
    
    Args:
        query: Search query
        num_results: Number of results to return (max 5)
    
    Returns:
        Search results as a string
    """
    # This is a mock implementation - in production, you'd use a real search API
    mock_results = {
        "python": "Python is a high-level programming language known for its simplicity and readability.",
        "machine learning": "Machine learning is a subset of AI that enables systems to learn from data.",
        "webrtc": "WebRTC is a technology that enables real-time communication in web browsers.",
        "weather": "Weather refers to atmospheric conditions including temperature, precipitation, and wind.",
        "news": "Latest news: Technology advances continue to shape the future of communication."
    }
    
    # Find relevant mock result or provide generic response
    query_lower = query.lower()
    for key, value in mock_results.items():
        if key in query_lower:
            return f"Search results for '{query}': {value}"
    
    return f"Search results for '{query}': Found various articles and information about this topic."


async def set_reminder(message: str, minutes_from_now: int) -> str:
    """
    Set a reminder for the future.
    
    Args:
        message: The reminder message
        minutes_from_now: How many minutes from now to set the reminder
    
    Returns:
        Confirmation message
    """
    future_time = datetime.datetime.now() + datetime.timedelta(minutes=minutes_from_now)
    time_str = future_time.strftime("%H:%M")
    return f"Reminder set for {time_str} ({minutes_from_now} minutes from now): '{message}'"


# Tool definitions for Ultravox
CONVERSATION_TOOLS = [
    {
        "name": "get_current_time",
        "description": "Get the current time in a specified timezone",
        "parameters": {
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "The timezone (e.g., UTC, EST, PST)",
                    "default": "UTC"
                },
                "format": {
                    "type": "string",
                    "description": "Time format - 24h or 12h",
                    "enum": ["24h", "12h"],
                    "default": "24h"
                }
            }
        }
    },
    {
        "name": "get_weather",
        "description": "Get current weather information for a location",
        "parameters": {
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or location"
                },
                "unit": {
                    "type": "string",
                    "description": "Temperature unit",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "calculate",
        "description": "Perform mathematical calculations",
        "parameters": {
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "set_reminder",
        "description": "Set a reminder for a future time",
        "parameters": {
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The reminder message"
                },
                "minutes_from_now": {
                    "type": "integer",
                    "description": "Minutes from now to set the reminder"
                }
            },
            "required": ["message", "minutes_from_now"]
        }
    }
]

# Tool executors mapping
TOOL_EXECUTORS = {
    "get_current_time": get_current_time,
    "get_weather": get_weather,
    "calculate": calculate,
    "search_web": search_web,
    "set_reminder": set_reminder
}



def create_speech_to_speech_pipeline(remote_host: str = "127.0.0.1") -> Pipeline:
    """
    Create a VAD-triggered Ultravox + Kokoro TTS pipeline for WebRTC.
    
    This pipeline implements a complete speech-to-speech system:
    1. Receives audio from WebRTC client
    2. Applies audio transforms (resampling, etc.)
    3. Performs voice activity detection with metadata
    4. Buffers speech segments until speech ends
    5. Processes complete utterances with Ultravox
    6. Synthesizes responses using Kokoro TTS
    7. Streams audio back to WebRTC client
    """
    pipeline = Pipeline()
    
    # Audio preprocessing - resample for VAD and Ultravox
    pipeline.add_node(AudioTransform(
        output_sample_rate=16000,
        output_channels=1,
        name="AudioTransform"
    ))
    
    # Voice Activity Detection with metadata
    vad = VoiceActivityDetector(
        frame_duration_ms=30,
        energy_threshold=0.02,
        speech_threshold=0.3,
        filter_mode=False,  # Keep metadata for buffering
        include_metadata=True,
        name="VAD"
    )
    vad.is_streaming = True
    pipeline.add_node(vad)
    
    # Add VAD logging for debugging microphone input
    pipeline.add_node(VADLoggingNode(name="VADLogger"))
    
    # VAD-triggered buffer that only outputs complete speech segments
    vad_buffer = VADTriggeredBuffer(
        min_speech_duration_s=1.0,    # Minimum 1.0s of speech before triggering
        silence_duration_s=0.5,       # 500ms of silence to confirm speech end
        pre_speech_buffer_s=1.0,      # 1s of pre-speech context
        max_silence_gap_s=1.5,        # Max 1.5s silence gap allowed within speech
        sample_rate=16000,
        name="VADTriggeredBuffer"
    )
    vad_buffer.is_streaming = True
    pipeline.add_node(vad_buffer)
    
    # Remote Ultravox execution with minimum duration protection
    remote_config = RemoteExecutorConfig(host=remote_host, port=50052, ssl_enabled=False)
    
    ultravox_instance = UltravoxNode(
        model_id="fixie-ai/ultravox-v0_5-llama-3_1-8b",
        system_prompt=(
            "You are a helpful assistant. Listen to what the user says and respond "
            "appropriately and concisely. Keep responses under 2 sentences. "
            "You have access to tools for getting time, weather, calculations, web search, and setting reminders."
        ),
        name="UltravoxNode",
        enable_conversation_history=True,
        conversation_history_minutes=5.0,  # Keep 5 minutes of conversation history
        tools=CONVERSATION_TOOLS,
        tool_executors=TOOL_EXECUTORS
    )

    remote_ultravox = RemoteObjectExecutionNode(
        obj_to_execute=ultravox_instance,
        remote_config=remote_config,
        name="RemoteUltravox",
        node_config={'streaming': True}
    )
    pipeline.add_node(remote_ultravox)
    
    logger.info("RemoteUltravox node added to pipeline - will initialize during pipeline setup")

    # Log text responses
    pipeline.add_node(TextLoggingNode(name="TextLogger"))

    # Kokoro TTS for speech synthesis
    kokoro_tts = KokoroTTSNode(
        lang_code='a',  # American English
        voice='af_heart',
        speed=1.0,
        sample_rate=24000,
        stream_chunks=True,  # Enable streaming
        name="KokoroTTS"
    )
    kokoro_tts.is_streaming = True
    
    # Use remote execution for Kokoro TTS too
    remote_tts = RemoteObjectExecutionNode(
        obj_to_execute=kokoro_tts,
        remote_config=remote_config,
        name="RemoteKokoroTTS",
        node_config={'streaming': True}
    )
    remote_tts.is_streaming = True
    pipeline.add_node(remote_tts)
    
    logger.info("⏳ RemoteKokoroTTS node added to pipeline - will initialize during pipeline setup")

    # Log audio output
    pipeline.add_node(MessageLoggingNode(
        message_prefix="🔊 SYNTHESIZED AUDIO",
        name="AudioLogger"
    ))
    
    return pipeline


async def register_webrtc_pipeline(registry: PipelineRegistry, remote_host: str = "127.0.0.1") -> str:
    """
    Register the WebRTC pipeline with the global registry for reuse and discovery.
    
    Args:
        registry: Pipeline registry instance
        remote_host: Host for remote execution
        
    Returns:
        Pipeline ID for the registered pipeline
    """
    # Create a pipeline instance
    pipeline = create_speech_to_speech_pipeline(remote_host)
    
    # Export the pipeline definition
    definition = pipeline.export_definition()
    
    # Register with local registry first
    pipeline_id = await registry.register_pipeline(
        name="webrtc_speech_to_speech",
        definition=definition,
        metadata={
            "description": "Complete WebRTC speech-to-speech pipeline with VAD, Ultravox, and Kokoro TTS",
            "category": "audio",
            "version": "1.0.0",
            "author": "RemoteMedia WebRTC Example",
            "tags": ["webrtc", "speech", "vad", "ultravox", "tts", "kokoro", "realtime"],
            "features": [
                "Voice Activity Detection with buffering",
                "Ultravox speech-to-text with tool calling",
                "Kokoro TTS synthesis",
                "Remote execution support",
                "Real-time WebRTC streaming"
            ],
            "tools": [
                "get_current_time",
                "get_weather", 
                "calculate",
                "search_web",
                "set_reminder"
            ],
            "requirements": [
                "aiortc",
                "ultravox",
                "kokoro-tts",
                "torch",
                "transformers"
            ],
            "remote_host": remote_host,
            "is_template": True,
            "use_case": "Real-time voice conversations with AI assistant capabilities"
        },
        owner_id="webrtc_server",
        access_level=AccessLevel.PUBLIC,
        persist=True
    )
    
    logger.info(f"📋 Registered WebRTC pipeline: {pipeline_id}")
    logger.info("   🎯 Pipeline features:")
    logger.info("   • Voice Activity Detection (VAD)")
    logger.info("   • Ultravox speech recognition with tools") 
    logger.info("   • Kokoro TTS speech synthesis")
    logger.info("   • Real-time WebRTC streaming")
    logger.info("   • Remote execution support")
    
    # Also register with the remote execution service for JavaScript client discovery
    try:
        logger.info("🌐 Registering pipeline with remote execution service...")
        remote_config = RemoteExecutorConfig(host=remote_host, port=50052, ssl_enabled=False)
        remote_client = RemoteExecutionClient(config=remote_config)
        await remote_client.connect()
        
        # Convert metadata to strings for gRPC
        string_metadata = {}
        for key, value in definition.get("metadata", {}).items():
            if isinstance(value, list):
                string_metadata[key] = ", ".join(str(v) for v in value)
            else:
                string_metadata[key] = str(value)
        
        # Register the pipeline with the remote service
        remote_pipeline_id = await remote_client.register_pipeline(
            pipeline_name="webrtc_speech_to_speech",
            definition=definition,
            metadata=string_metadata
        )
        
        logger.info(f"✅ Pipeline registered with remote service: {remote_pipeline_id}")
        logger.info("🔍 JavaScript clients can now discover this pipeline via PipelineClient.listPipelines()")
        
    except Exception as e:
        logger.warning(f"⚠️  Could not register with remote service: {e}")
        logger.info("💡 Local pipeline registration still available for this server")
    
    return pipeline_id


async def main():
    """Run the WebRTC server with pipeline integration."""
    
    # Configuration
    USE_ML_PIPELINE = os.environ.get("USE_ML", "false").lower() == "true"
    REMOTE_HOST = os.environ.get("REMOTE_HOST", "127.0.0.1")
    SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
    SERVER_PORT = int(os.environ.get("SERVER_PORT", "8080"))
    ENABLE_PERSISTENCE = os.environ.get("ENABLE_PERSISTENCE", "true").lower() == "true"
    
    logger.info("=== WebRTC Pipeline Server ===")
    logger.info(f"Server: {SERVER_HOST}:{SERVER_PORT}")
    logger.info(f"ML Pipeline: {'Enabled' if USE_ML_PIPELINE else 'Disabled'}")
    logger.info(f"Remote Host: {REMOTE_HOST}")
    logger.info(f"Persistence: {'Enabled' if ENABLE_PERSISTENCE else 'Disabled'}")
    
    # Initialize pipeline registry with persistence
    registry = PipelineRegistry(
        db_path="webrtc_pipelines.db" if ENABLE_PERSISTENCE else None,
        enable_persistence=ENABLE_PERSISTENCE
    )
    await registry.initialize()
    
    # Register the WebRTC pipeline for discovery and reuse
    try:
        pipeline_id = await register_webrtc_pipeline(registry, REMOTE_HOST)
        logger.info(f"✅ WebRTC pipeline registered and available for clients")
        
        # Log registry statistics
        pipeline_count = len(registry.pipelines)
        logger.info(f"📊 Registry contains {pipeline_count} pipeline(s)")
        
        if ENABLE_PERSISTENCE:
            logger.info("💾 Pipeline persisted to database for future sessions")
        
    except Exception as e:
        logger.error(f"❌ Failed to register WebRTC pipeline: {e}")
        # Continue without registry - server will still work
    
    # Create server configuration
    examples_dir = Path(__file__).parent
    config = WebRTCConfig(
        host=SERVER_HOST,
        port=SERVER_PORT,
        enable_cors=True,
        stun_servers=["stun:stun.l.google.com:19302"],
        static_files_path=str(examples_dir)  # Serve client files from examples directory
    )
    
    # Pipeline factory function
    def create_pipeline() -> Pipeline:
        return create_speech_to_speech_pipeline(remote_host=REMOTE_HOST)
    
    # Create and start server
    server = WebRTCServer(config=config, pipeline_factory=create_pipeline)
    
    try:
        await server.start()
        
        logger.info("WebRTC server is running!")
        logger.info("Connect with a WebRTC client:")
        logger.info(f"  📱 Web Client: http://localhost:{SERVER_PORT}/webrtc_client.html")
        logger.info(f"  🌐 WebSocket: ws://{SERVER_HOST}:{SERVER_PORT}/ws")
        logger.info(f"  ❤️  Health check: http://{SERVER_HOST}:{SERVER_PORT}/health")
        logger.info(f"  📊 Connections: http://{SERVER_HOST}:{SERVER_PORT}/connections")
        
        # Display pipeline registry information
        logger.info("")
        logger.info("🗂️  Pipeline Registry:")
        logger.info(f"   • Registered pipelines: {len(registry.pipelines)}")
        if ENABLE_PERSISTENCE:
            logger.info(f"   • Database: webrtc_pipelines.db")
            logger.info(f"   • Persistence: Enabled")
        else:
            logger.info(f"   • Persistence: Disabled (in-memory only)")
        
        for pid, registered in registry.pipelines.items():
            logger.info(f"   • {registered.name} ({pid[:12]}...)")
            logger.info(f"     Tags: {', '.join(registered.metadata.get('tags', []))}")
        
        if USE_ML_PIPELINE:
            logger.info("")
            logger.info("🛠️  Tool-enabled assistant is active with:")
            logger.info("   • Time & date information")
            logger.info("   • Weather information (mock)")
            logger.info("   • Mathematical calculations")
            logger.info("   • Web search (mock)")
            logger.info("   • Reminder setting")
        
        if not USE_ML_PIPELINE:
            logger.info("")
            logger.info("💡 To enable ML features (speech recognition, TTS):")
            logger.info("   1. Install ML dependencies: pip install -r requirements-ml.txt")
            logger.info("   2. Set environment variable: USE_ML=true")
            logger.info("   3. Start remote service if using remote execution")
        
        logger.info("")
        logger.info("💡 Pipeline Access:")
        logger.info("   • JavaScript clients can discover and execute this pipeline")
        logger.info("   • Use PipelineClient.listPipelines() to see available pipelines")  
        logger.info("   • Execute with PipelineClient.executePipeline('webrtc_speech_to_speech', data)")
        logger.info("")
        logger.info("Press Ctrl+C to stop the server")
        
        # Keep the server running
        while True:
            await asyncio.sleep(10)
            
            # Log connection statistics
            connections_count = len(server.connections)
            pipeline_count = len(registry.pipelines)
            if connections_count > 0 or pipeline_count > 1:  # More than just our WebRTC pipeline
                logger.info(f"Active connections: {connections_count}, Registered pipelines: {pipeline_count}")
                
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        
        # Save any in-memory pipelines before shutdown
        if ENABLE_PERSISTENCE and registry.pipelines:
            logger.info("💾 Ensuring all pipelines are persisted...")
            for pid, registered in registry.pipelines.items():
                try:
                    saved = await registry.save_pipeline(pid, "webrtc_server", AccessLevel.PUBLIC)
                    if saved:
                        logger.info(f"   ✅ Saved: {registered.name}")
                except Exception as e:
                    logger.warning(f"   ⚠️  Could not save {registered.name}: {e}")
                    
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
    finally:
        await server.stop()
        logger.info("Server stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"Failed to start server: {e}", exc_info=True)
