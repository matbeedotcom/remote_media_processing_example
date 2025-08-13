"""
Supporting Node classes for VAD-triggered Ultravox with TTS streaming.

This module contains the custom Node implementations used in the speech-to-speech pipeline:
- VADTriggeredBuffer: Accumulates speech and triggers on speech end
- UltravoxMinDurationWrapper: Enforces minimum audio duration for Ultravox
- AudioOutputNode: Streams and saves generated audio
- TextLoggingNode: Logs text responses
"""

import asyncio
import logging
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import AsyncGenerator, Any, Optional, Tuple

from remotemedia.core.node import Node
from remotemedia.nodes import PassThroughNode

logger = logging.getLogger(__name__)


class VADTriggeredBuffer(Node):
    """
    A buffer that accumulates speech audio and triggers output only when speech ends.
    
    This node:
    1. Maintains a rolling buffer of the past 1 second of audio (pre-speech context)
    2. Accumulates audio chunks during speech segments
    3. Tracks speech/silence state transitions
    4. Outputs accumulated speech audio only when:
       - Speech ends (VAD transitions to silence for sufficient duration)
       - At least minimum duration of speech was accumulated
       - Speech was continuous (no long gaps)
    
    The buffer will discard the entire utterance if speech is interrupted by
    silence that exceeds max_silence_gap_s, ensuring only continuous speech
    segments trigger the voice assistant.
    """
    
    def __init__(
        self,
        min_speech_duration_s: float = 1.0,
        max_speech_duration_s: float = 10.0,
        silence_duration_s: float = 0.5,
        pre_speech_buffer_s: float = 1.0,
        max_silence_gap_s: float = 1.5,
        sample_rate: int = 16000,
        **kwargs
    ):
        """
        Initialize the VAD-triggered buffer.
        
        Args:
            min_speech_duration_s: Minimum speech duration before triggering (default: 1.0s)
            max_speech_duration_s: Maximum speech duration (forces trigger, default: 10.0s)
            silence_duration_s: Duration of silence needed to confirm speech end (default: 0.5s)
            pre_speech_buffer_s: Duration of audio to buffer before speech starts (default: 1.0s)
            max_silence_gap_s: Maximum allowed silence gap within speech before discarding (default: 1.5s)
            sample_rate: Expected audio sample rate (default: 16000)
        """
        super().__init__(**kwargs)
        self.min_speech_duration_s = min_speech_duration_s
        self.max_speech_duration_s = max_speech_duration_s
        self.silence_duration_s = silence_duration_s
        self.pre_speech_buffer_s = pre_speech_buffer_s
        self.max_silence_gap_s = max_silence_gap_s
        self.sample_rate = sample_rate
        self.is_streaming = True
        
        # State tracking
        self._speech_buffer = None
        self._pre_speech_buffer = []  # Rolling buffer for pre-speech context
        self._is_in_speech = False
        self._silence_accumulated_samples = 0
        self._speech_samples_count = 0  # Track actual speech samples (not including pre-buffer)
        
        # Derived parameters
        self.min_samples = int(min_speech_duration_s * sample_rate)
        self.max_samples = int(max_speech_duration_s * sample_rate)
        self.silence_samples = int(silence_duration_s * sample_rate)
        self.pre_speech_samples = int(pre_speech_buffer_s * sample_rate)
        self.max_silence_gap_samples = int(max_silence_gap_s * sample_rate)
        
    async def process(self, data_stream: AsyncGenerator[Any, None]) -> AsyncGenerator[Tuple[np.ndarray, int, dict], None]:
        """
        Process VAD-annotated audio stream and trigger output on speech end.
        
        Expected input: ((audio_data, sample_rate), vad_metadata) tuples
        """
        # Store original metadata to preserve session_id and other data
        original_metadata = {}
        
        async for data in data_stream:
            # Debug the data format
            logger.debug(f"VADTriggeredBuffer: Received data type: {type(data)}")
            
            if not isinstance(data, tuple) or len(data) != 2:
                logger.warning(f"VADTriggeredBuffer: Invalid input format: {type(data)}, len: {len(data) if hasattr(data, '__len__') else 'N/A'}")
                continue
                
            # The VAD output format is: ((audio_data, sample_rate), vad_metadata)
            item1, item2 = data
            
            if isinstance(item2, dict):
                # Correct format: ((audio_data, sample_rate), vad_metadata)
                if isinstance(item1, tuple) and len(item1) == 2:
                    audio_data, sample_rate = item1
                    vad_metadata = item2
                    # Store original metadata from the VAD metadata
                    if 'original_metadata' in vad_metadata:
                        original_metadata = vad_metadata['original_metadata'].copy()
                    elif not original_metadata:  # Only update if we don't have metadata yet
                        # Extract session_id and other metadata from VAD metadata
                        original_metadata = {k: v for k, v in vad_metadata.items() 
                                           if k not in ['is_speech', 'speech_ratio', 'avg_energy', 'frame_count']}
                else:
                    logger.warning(f"VADTriggeredBuffer: Expected (audio_data, sample_rate) tuple, got {type(item1)}")
                    continue
            else:
                logger.warning(f"VADTriggeredBuffer: Expected VAD metadata dict as second item, got {type(item2)}")
                continue
                
            if not isinstance(audio_data, np.ndarray):
                logger.warning(f"VADTriggeredBuffer: Expected numpy array audio, got {type(audio_data)}")
                continue
                
            # Ensure consistent sample rate
            if sample_rate != self.sample_rate:
                logger.warning(f"VADTriggeredBuffer: Sample rate mismatch: {sample_rate} vs {self.sample_rate}")
                continue
                
            # Flatten audio to 1D for processing
            if audio_data.ndim > 1:
                audio_flat = audio_data.flatten()
            else:
                audio_flat = audio_data
                
            is_speech = vad_metadata.get("is_speech", False)
            speech_ratio = vad_metadata.get("speech_ratio", 0.0)
            avg_energy = vad_metadata.get("avg_energy", 0.0)
            chunk_samples = len(audio_flat)
            
            # Always maintain the pre-speech rolling buffer
            self._maintain_pre_speech_buffer(audio_flat)
            
            # Process based on speech state
            if is_speech:
                triggered_audio = await self._handle_speech_chunk(audio_flat, chunk_samples)
                if triggered_audio is not None:
                    logger.info(
                        f"VADTriggeredBuffer: Triggering on max duration "
                        f"({len(triggered_audio)/self.sample_rate:.2f}s total audio)"
                    )
                    yield (triggered_audio.reshape(1, -1), self.sample_rate, original_metadata)
            else:
                # Check if we should trigger on silence
                triggered_audio = await self._handle_silence_chunk(audio_flat, chunk_samples)
                if triggered_audio is not None:
                    logger.info(
                        f"VADTriggeredBuffer: Triggering on speech end "
                        f"({len(triggered_audio)/self.sample_rate:.2f}s total audio)"
                    )
                    yield (triggered_audio.reshape(1, -1), self.sample_rate, original_metadata)
                    
    def _maintain_pre_speech_buffer(self, audio_chunk: np.ndarray):
        """Maintain a rolling buffer of pre-speech audio."""
        self._pre_speech_buffer.append(audio_chunk.copy())
        
        # Calculate total samples in buffer
        total_samples = sum(len(chunk) for chunk in self._pre_speech_buffer)
        
        # Remove oldest chunks if buffer exceeds target duration
        while total_samples > self.pre_speech_samples and len(self._pre_speech_buffer) > 1:
            removed = self._pre_speech_buffer.pop(0)
            total_samples -= len(removed)
    
    async def _handle_speech_chunk(self, audio_chunk: np.ndarray, chunk_samples: int) -> Optional[np.ndarray]:
        """Handle a chunk containing speech. Returns audio if max duration reached."""
        if not self._is_in_speech:
            # Starting new speech segment - include pre-speech context
            logger.info("VADTriggeredBuffer: Speech started - including pre-speech context")
            self._is_in_speech = True
            
            # Start with pre-speech buffer
            if self._pre_speech_buffer:
                self._speech_buffer = np.concatenate(self._pre_speech_buffer)
                pre_speech_duration = len(self._speech_buffer) / self.sample_rate
                logger.debug(f"VADTriggeredBuffer: Added {pre_speech_duration:.2f}s of pre-speech context")
            else:
                self._speech_buffer = np.array([], dtype=np.float32)
            
            # Add current speech chunk
            self._speech_buffer = np.concatenate([self._speech_buffer, audio_chunk])
            self._speech_samples_count = chunk_samples  # Only count actual speech samples
            self._silence_accumulated_samples = 0  # Reset silence counter
        else:
            # Continuing speech segment
            if self._speech_buffer is not None:
                self._speech_buffer = np.concatenate([self._speech_buffer, audio_chunk])
                self._speech_samples_count += chunk_samples  # Track actual speech samples
                # Reset silence counter when speech continues
                self._silence_accumulated_samples = 0
                
                # Check if we've exceeded max duration
                if len(self._speech_buffer) >= self.max_samples:
                    logger.info(
                        f"VADTriggeredBuffer: Max speech duration reached "
                        f"({len(self._speech_buffer)/self.sample_rate:.2f}s), forcing trigger"
                    )
                    # Trigger and reset
                    triggered_audio = self._speech_buffer.copy()
                    self._reset_state()
                    return triggered_audio
                    
        return None
        
    async def _handle_silence_chunk(self, silence_chunk: np.ndarray, chunk_samples: int) -> Optional[np.ndarray]:
        """Handle a chunk containing silence. Returns audio if trigger conditions are met."""
        if not self._is_in_speech or self._speech_buffer is None:
            # Reset silence tracking if we're not in speech
            self._silence_accumulated_samples = 0
            return None
            
        # Add the silence chunk to the buffer first (to preserve natural pauses)
        self._speech_buffer = np.concatenate([self._speech_buffer, silence_chunk])
        
        # Accumulate silence duration
        self._silence_accumulated_samples += chunk_samples
        silence_duration_s = self._silence_accumulated_samples / self.sample_rate
        
        logger.debug(f"VADTriggeredBuffer: Accumulated {silence_duration_s:.2f}s of silence (need {self.silence_duration_s:.2f}s)")
        
        # Check if silence gap exceeds maximum allowed - speech is not continuous
        if self._silence_accumulated_samples >= self.max_silence_gap_samples:
            logger.warning(
                f"VADTriggeredBuffer: Silence gap too long ({silence_duration_s:.2f}s >= {self.max_silence_gap_s}s), "
                f"discarding non-continuous speech"
            )
            self._reset_state()
            return None
        
        # Check if we have enough silence to confirm speech end
        if self._silence_accumulated_samples >= self.silence_samples:
            # Check if we have minimum speech duration (based on actual speech samples, not including pre-buffer)
            speech_duration_s = self._speech_samples_count / self.sample_rate
            total_duration_s = len(self._speech_buffer) / self.sample_rate
            
            if self._speech_samples_count >= self.min_samples:
                # Trigger!
                logger.info(f"VADTriggeredBuffer: Speech end confirmed after {silence_duration_s:.2f}s silence "
                           f"(speech: {speech_duration_s:.2f}s, total: {total_duration_s:.2f}s)")
                triggered_audio = self._speech_buffer.copy()
                self._reset_state()
                return triggered_audio
            else:
                logger.debug(
                    f"VADTriggeredBuffer: Speech too short "
                    f"({speech_duration_s:.2f}s < {self.min_speech_duration_s}s), discarding"
                )
                self._reset_state()
                
        return None
        
    def _reset_state(self):
        """Reset the buffer state."""
        self._speech_buffer = None
        self._is_in_speech = False
        self._silence_accumulated_samples = 0
        self._speech_samples_count = 0
        # Keep the pre-speech buffer for the next utterance


class UltravoxMinDurationWrapper(Node):
    """
    Wrapper around UltravoxNode that enforces minimum audio duration.
    
    This prevents generation on insufficient audio data.
    """
    
    def __init__(self, ultravox_node, min_duration_s: float = 1.0, sample_rate: int = 16000, **kwargs):
        super().__init__(**kwargs)
        self.ultravox_node = ultravox_node
        self.min_duration_s = min_duration_s
        self.sample_rate = sample_rate
        self.min_samples = int(min_duration_s * sample_rate)
        self.is_streaming = True
        
    async def initialize(self):
        """Initialize the wrapped Ultravox node."""
        if not self.ultravox_node.is_initialized:
            await self.ultravox_node.initialize()
        
    async def flush(self):
        """Flush the wrapped Ultravox node."""
        if hasattr(self.ultravox_node, 'flush'):
            return await self.ultravox_node.flush()
        return None
        
    async def cleanup(self):
        """Cleanup the wrapped Ultravox node."""
        await self.ultravox_node.cleanup()
        
    async def process(self, data_stream):
        """Filter out audio chunks that are too short."""
        async def filtered_stream():
            async for data in data_stream:
                if isinstance(data, tuple) and len(data) >= 2:
                    audio_data = data[0]
                    sample_rate = data[1]
                    metadata = data[2] if len(data) > 2 else {}
                    
                    if isinstance(audio_data, np.ndarray):
                        duration_s = audio_data.size / sample_rate
                        if duration_s >= self.min_duration_s:
                            logger.info(f"UltravoxWrapper: Processing {duration_s:.2f}s of audio (≥{self.min_duration_s:.2f}s minimum)")
                            # Preserve all parts of the tuple including metadata
                            yield data
                        else:
                            logger.warning(f"UltravoxWrapper: Rejecting {duration_s:.2f}s of audio (< {self.min_duration_s:.2f}s minimum)")
                            continue
                    else:
                        # Pass through non-audio data
                        yield data
                else:
                    # Pass through non-tuple data
                    yield data
                    
        async for result in self.ultravox_node.process(filtered_stream()):
            yield result


class AudioOutputNode(Node):
    """Node that streams generated audio to output and saves files."""
    
    def __init__(self, output_dir: str = "generated_audio", **kwargs):
        super().__init__(**kwargs)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.response_count = 0
        self.is_streaming = True
        
    async def process(self, data_stream):
        """Process audio chunks and save them."""
        async for data in data_stream:
            if isinstance(data, tuple) and len(data) >= 2:
                audio_data = data[0]
                sample_rate = data[1]
                metadata = data[2] if len(data) > 2 else {}
                
                if isinstance(audio_data, np.ndarray):
                    # Save the audio chunk
                    self.response_count += 1
                    output_file = self.output_dir / f"response_{self.response_count:03d}.wav"
                    
                    # Ensure audio is properly shaped for saving
                    if audio_data.ndim > 1:
                        audio_to_save = audio_data.flatten()
                    else:
                        audio_to_save = audio_data
                        
                    # Save asynchronously
                    await asyncio.to_thread(
                        sf.write, 
                        str(output_file), 
                        audio_to_save.astype(np.float32), 
                        sample_rate
                    )
                    
                    duration_s = len(audio_to_save) / sample_rate
                    logger.info(f"🔊 SAVED AUDIO: {output_file} ({duration_s:.2f}s, {sample_rate}Hz)")
                    
                    # Log session info if available
                    if metadata and 'session_id' in metadata:
                        logger.info(f"  Session ID: {metadata['session_id']}")
                    
                    # Also print text if available
                    print(f"🎵 Generated audio response #{self.response_count}: {duration_s:.2f}s")
                    
            # Pass through the data
            yield data


class VADLoggingNode(PassThroughNode):
    """Node that logs VAD events for debugging."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frame_count = 0
        self._last_is_speech = None
        
    def process(self, data):
        """Log VAD metadata and pass data through."""
        if isinstance(data, tuple) and len(data) == 2:
            item1, item2 = data
            
            # Check if this is VAD output: ((audio_data, sample_rate), vad_metadata)
            if isinstance(item2, dict) and isinstance(item1, tuple) and len(item1) == 2:
                audio_data, sample_rate = item1
                vad_metadata = item2
                
                if isinstance(audio_data, np.ndarray):
                    self.frame_count += 1
                    is_speech = vad_metadata.get("is_speech", False)
                    speech_ratio = vad_metadata.get("speech_ratio", 0.0)
                    avg_energy = vad_metadata.get("avg_energy", 0.0)
                    
                    # Only log speech transitions and periodic updates
                    should_log = False
                    
                    # Log speech start/end transitions
                    if self._last_is_speech is not None:
                        if not self._last_is_speech and is_speech:
                            logger.info("🟢 WebRTC VAD: SPEECH STARTED")
                            should_log = True
                        elif self._last_is_speech and not is_speech:
                            logger.info("🔴 WebRTC VAD: SPEECH ENDED")
                            should_log = True
                    
                    # Log periodic status (every 50 frames = ~2 seconds)
                    if self.frame_count % 50 == 0:
                        should_log = True
                    
                    if should_log:
                        speech_indicator = "🎤 SPEECH" if is_speech else "🔇 SILENCE"
                        logger.info(f"WebRTC VAD #{self.frame_count}: {speech_indicator} | ratio={speech_ratio:.3f} | energy={avg_energy:.4f}")
                    
                    self._last_is_speech = is_speech
        
        return data


class UltravoxImmediateProcessor(Node):
    """
    Wrapper around UltravoxNode that processes complete utterances immediately.
    
    This bypasses UltravoxNode's internal buffering and processes each complete
    utterance from VAD immediately, then clears the internal buffer.
    """
    
    def __init__(self, ultravox_node, min_duration_s: float = 1.0, sample_rate: int = 16000, **kwargs):
        super().__init__(**kwargs)
        self.ultravox_node = ultravox_node
        self.min_duration_s = min_duration_s
        self.sample_rate = sample_rate
        self.min_samples = int(min_duration_s * sample_rate)
        self.is_streaming = True
        
    async def initialize(self):
        """Initialize the wrapped Ultravox node."""
        if not self.ultravox_node.is_initialized:
            await self.ultravox_node.initialize()
        
    async def flush(self):
        """Flush the wrapped Ultravox node."""
        if hasattr(self.ultravox_node, 'flush'):
            return await self.ultravox_node.flush()
        return None
        
    async def cleanup(self):
        """Cleanup the wrapped Ultravox node."""
        await self.ultravox_node.cleanup()
        
    async def process(self, data_stream):
        """Process complete utterances immediately."""
        async for data in data_stream:
            if isinstance(data, tuple) and len(data) >= 2:
                audio_data = data[0]
                sample_rate = data[1]
                metadata = data[2] if len(data) > 2 else {}
                
                if isinstance(audio_data, np.ndarray):
                    duration_s = audio_data.size / sample_rate
                    if duration_s >= self.min_duration_s:
                        logger.info(f"UltravoxProcessor: Processing complete {duration_s:.2f}s utterance immediately")
                        
                        # Clear any existing buffer to prevent accumulation
                        self.ultravox_node.audio_buffer = np.array([], dtype=np.float32)
                        
                        # Process the complete utterance directly
                        if self.ultravox_node.llm_pipeline:
                            audio_flat = audio_data.flatten().astype(np.float32)
                            response = await self.ultravox_node._generate_response(audio_flat)
                            if response:
                                logger.info(f"UltravoxProcessor: Generated response for {duration_s:.2f}s utterance")
                                # Include metadata with response
                                yield (response, metadata)
                        else:
                            logger.error("UltravoxProcessor: Pipeline not initialized")
                    else:
                        logger.warning(f"UltravoxProcessor: Rejecting {duration_s:.2f}s of audio (< {self.min_duration_s:.2f}s minimum)")
            else:
                # Pass through non-audio data
                yield data


class MessageLoggingNode(PassThroughNode):
    """Node that logs processed messages."""
    
    def __init__(self, message_prefix: str = "", **kwargs):
        super().__init__(**kwargs)
        self.message_prefix = message_prefix
        
    def process(self, data):
        """Log the data and pass it through."""
        if isinstance(data, str):
            logger.info(f"{self.message_prefix}: {data}")
        elif isinstance(data, dict):
            if 'type' in data:
                logger.info(f"{self.message_prefix} ({data['type']}): {data}")
            else:
                logger.info(f"{self.message_prefix}: {data}")
        else:
            logger.info(f"{self.message_prefix}: {type(data)} - {data}")
        return data


class TextLoggingNode(PassThroughNode):
    """Node that logs text responses from Ultravox."""
    
    def process(self, data):
        """Log text responses."""
        if isinstance(data, (str, tuple)):
            text = data[0] if isinstance(data, tuple) else data
            print(f"\n🎤 ULTRAVOX RESPONSE: {text}\n")
        return data