#!/usr/bin/env python3
"""
VAD-triggered Ultravox with Kokoro TTS streaming: Speech-to-speech pipeline.

This example demonstrates a complete speech-to-speech pipeline:
1. VAD detects speech segments
2. Ultravox generates text responses 
3. Kokoro TTS synthesizes the responses back to audio in real-time

This creates a natural conversational flow with streaming audio output.

**TO RUN THIS EXAMPLE:**

1.  **Install ML dependencies:**
    $ pip install -r requirements-ml.txt
    $ pip install kokoro>=0.9.4 soundfile

2.  **Install espeak for Kokoro:**
    $ sudo apt-get install espeak-ng

3.  **Start the server:**
    $ PYTHONPATH=. python remote_service/src/server.py

4.  **Run this script:**
    $ python examples/vad_ultravox_kokoro_streaming.py
"""

import asyncio
import logging
import numpy as np
import soundfile as sf
import os
import sys
from pathlib import Path
from typing import AsyncGenerator, Any, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from remotemedia.core.pipeline import Pipeline
from remotemedia.core.node import RemoteExecutorConfig, Node
from remotemedia.nodes.source import MediaReaderNode, AudioTrackSource
from remotemedia.nodes.audio import AudioTransform, VoiceActivityDetector
from remotemedia.nodes.remote import RemoteObjectExecutionNode
from remotemedia.nodes.ml import UltravoxNode, KokoroTTSNode
from remotemedia.nodes import PassThroughNode
from vad_ultravox_nodes import VADTriggeredBuffer, UltravoxMinDurationWrapper, AudioOutputNode, TextLoggingNode

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reduce noise from some loggers
logging.getLogger("Pipeline").setLevel(logging.INFO)
logging.getLogger("remotemedia.nodes.remote").setLevel(logging.INFO)




async def create_speech_with_pauses(filepath: str, duration_s: int = 15, sample_rate: int = 44100):
    """Create demo audio with speech segments separated by pauses."""
    if os.path.exists(filepath):
        return
        
    logger.info(f"Creating demo audio with speech and pauses at '{filepath}'...")
    
    samples = int(sample_rate * duration_s)
    t = np.linspace(0., float(duration_s), samples)
    audio = np.zeros(samples)
    
    # Define speech segments with clear pauses
    speech_segments = [
        (1.0, 3.5, "Hello there"),        # 2.5s speech
        (5.0, 7.0, "How are you today"),  # 2.0s speech  
        (9.0, 12.0, "Tell me about artificial intelligence"), # 3.0s speech
        (13.5, 14.5, "Thank you")        # 1.0s speech
    ]
    
    for start, end, description in speech_segments:
        mask = (t >= start) & (t < end)
        # Create varied speech-like patterns
        freq_base = 200 + 150 * np.sin(2 * np.pi * 0.5 * t[mask])
        freq_mod = 100 * np.sin(2 * np.pi * 5 * t[mask])
        speech_signal = 0.3 * np.sin(2 * np.pi * (freq_base + freq_mod) * t[mask])
        audio[mask] = speech_signal
        logger.info(f"  Speech segment: {start:.1f}s-{end:.1f}s ({description})")
    
    # Add light background noise
    audio += 0.01 * np.random.randn(samples)
    
    await asyncio.to_thread(sf.write, filepath, audio.astype(np.float32), sample_rate)
    logger.info(f"Demo audio created: {duration_s}s with {len(speech_segments)} speech segments")


async def main():
    """Run the VAD-triggered Ultravox + Kokoro TTS pipeline."""
    REMOTE_HOST = os.environ.get("REMOTE_HOST", "127.0.0.1")
    logger.info("--- VAD-Triggered Ultravox + Kokoro TTS: Speech-to-Speech Pipeline ---")

    # Create demo audio
    demo_audio_path = "examples/audio.wav"
    await create_speech_with_pauses(demo_audio_path)
    
    pipeline = Pipeline()

    # Audio source
    pipeline.add_node(MediaReaderNode(
        path=demo_audio_path,
        chunk_size=4096,
        name="MediaReader"
    ))

    pipeline.add_node(AudioTrackSource(name="AudioTrackSource"))

    # Resample for VAD and Ultravox
    pipeline.add_node(AudioTransform(
        output_sample_rate=16000,
        output_channels=1,
        name="AudioTransform"
    ))

    # VAD to detect speech/silence
    vad = VoiceActivityDetector(
        frame_duration_ms=30,
        energy_threshold=0.02,
        speech_threshold=0.3,
        filter_mode=False,  # Keep metadata
        include_metadata=True,
        name="VAD"
    )
    vad.is_streaming = True
    pipeline.add_node(vad)

    # VAD-triggered buffer that only outputs complete speech segments
    vad_buffer = VADTriggeredBuffer(
        min_speech_duration_s=0.8,    # Minimum 0.8s of speech before triggering
        max_speech_duration_s=8.0,    # Force trigger after 8s of continuous speech
        silence_duration_s=1.0,       # 1000ms of silence to confirm speech end (more robust)
        sample_rate=16000,
        name="VADTriggeredBuffer"
    )
    vad_buffer.is_streaming = True
    pipeline.add_node(vad_buffer)

    # Remote Ultravox execution with minimum duration protection
    remote_config = RemoteExecutorConfig(host=REMOTE_HOST, port=50052, ssl_enabled=False)
    
    ultravox_instance = UltravoxNode(
        model_id="fixie-ai/ultravox-v0_5-llama-3_1-8b",
        system_prompt=(
            "You are a helpful assistant. Listen to what the user says and respond "
            "appropriately and concisely. Keep responses under 2 sentences."
        ),
        buffer_duration_s=10.0,  # Large buffer to process complete utterances (up to 10s)
        name="UltravoxNode"
    )
    
    # Wrap Ultravox with minimum duration protection
    protected_ultravox = UltravoxMinDurationWrapper(
        ultravox_node=ultravox_instance,
        min_duration_s=1.0,  # Require at least 1s of audio
        sample_rate=16000,
        name="ProtectedUltravox"
    )

    remote_node = RemoteObjectExecutionNode(
        obj_to_execute=protected_ultravox,
        remote_config=remote_config,
        name="RemoteUltravox",
        node_config={'streaming': True}
    )
    remote_node.is_streaming = True
    pipeline.add_node(remote_node)

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

    # Audio output node to save generated speech
    pipeline.add_node(AudioOutputNode(
        output_dir="generated_responses",
        name="AudioOutput"
    ))

    logger.info("Starting VAD-triggered speech-to-speech pipeline...")
    logger.info("Pipeline: Speech Input → VAD → Ultravox → Kokoro TTS → Audio Output")
    
    async with pipeline.managed_execution():
        try:
            response_count = 0
            async for result in pipeline.process():
                response_count += 1
                logger.debug(f"Received audio response {response_count}")
            logger.info(f"Pipeline completed. Generated {response_count} audio responses.")
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            raise

    logger.info("Speech-to-speech pipeline finished. Check 'generated_responses/' for audio files.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        logging.error("Please ensure:")
        logging.error("  1. The remote server is running")
        logging.error("  2. Kokoro TTS is installed: pip install kokoro>=0.9.4 soundfile")
        logging.error("  3. espeak is installed: sudo apt-get install espeak-ng")