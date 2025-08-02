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

sys.path.insert(0, str(Path(__file__).parent.parent))

from remotemedia.core.pipeline import Pipeline
from remotemedia.core.node import RemoteExecutorConfig, Node
from remotemedia.nodes.audio import AudioTransform, VoiceActivityDetector
from remotemedia.nodes.ml import UltravoxNode
from remotemedia.nodes.transform import TextTransformNode
from remotemedia.nodes.remote import RemoteObjectExecutionNode
from remotemedia.nodes import PassThroughNode
from remotemedia.webrtc import WebRTCServer, WebRTCConfig

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
            "appropriately and concisely. Keep responses under 2 sentences."
        ),
        name="UltravoxNode",
        enable_conversation_history=True,
        conversation_history_minutes=5.0  # Keep 5 minutes of conversation history
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


async def main():
    """Run the WebRTC server with pipeline integration."""
    
    # Configuration
    USE_ML_PIPELINE = os.environ.get("USE_ML", "false").lower() == "true"
    REMOTE_HOST = os.environ.get("REMOTE_HOST", "127.0.0.1")
    SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
    SERVER_PORT = int(os.environ.get("SERVER_PORT", "8080"))
    
    logger.info("=== WebRTC Pipeline Server ===")
    logger.info(f"Server: {SERVER_HOST}:{SERVER_PORT}")
    logger.info(f"ML Pipeline: {'Enabled' if USE_ML_PIPELINE else 'Disabled'}")
    logger.info(f"Remote Host: {REMOTE_HOST}")
    
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
        
        if not USE_ML_PIPELINE:
            logger.info("")
            logger.info("💡 To enable ML features (speech recognition, TTS):")
            logger.info("   1. Install ML dependencies: pip install -r requirements-ml.txt")
            logger.info("   2. Set environment variable: USE_ML=true")
            logger.info("   3. Start remote service if using remote execution")
        
        logger.info("")
        logger.info("Press Ctrl+C to stop the server")
        
        # Keep the server running
        while True:
            await asyncio.sleep(10)
            
            # Log connection statistics
            connections_count = len(server.connections)
            if connections_count > 0:
                logger.info(f"Active connections: {connections_count}")
                
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
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
