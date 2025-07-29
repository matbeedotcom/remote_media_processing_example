"""
Kokoro TTS Node for high-quality text-to-speech synthesis.
"""

import logging
import numpy as np
from typing import Any, AsyncGenerator, Optional, Union, Tuple
import asyncio

from remotemedia.core.node import Node

logger = logging.getLogger(__name__)


class KokoroTTSNode(Node):
    """
    Text-to-speech synthesis using Kokoro TTS.
    
    Kokoro is an open-weight TTS model with 82 million parameters that delivers
    comparable quality to larger models while being significantly faster and more
    cost-efficient.
    """
    
    def __init__(
        self,
        lang_code: str = 'a',
        voice: str = 'af_heart',
        speed: float = 1.0,
        split_pattern: str = r'\n+',
        sample_rate: int = 24000,
        stream_chunks: bool = True,
        **kwargs
    ):
        """
        Initialize Kokoro TTS node.
        
        Args:
            lang_code: Language code ('a' for American English, 'b' for British English,
                      'e' for Spanish, 'f' for French, 'h' for Hindi, 'i' for Italian,
                      'j' for Japanese, 'p' for Brazilian Portuguese, 'z' for Mandarin Chinese)
            voice: Voice identifier (e.g., 'af_heart')
            speed: Speech speed multiplier (default: 1.0)
            split_pattern: Regex pattern for splitting text (default: r'\n+')
            sample_rate: Output sample rate (default: 24000)
            stream_chunks: Whether to stream audio chunks as they're generated (default: True)
        """
        super().__init__(**kwargs)
        self.lang_code = lang_code
        self.voice = voice
        self.speed = speed
        self.split_pattern = split_pattern
        self.sample_rate = sample_rate
        self.stream_chunks = stream_chunks
        self.is_streaming = stream_chunks
        
        self._pipeline = None
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize the Kokoro TTS pipeline."""
        if self._initialized:
            return
            
        try:
            # Import Kokoro here to avoid import errors if not installed
            from kokoro import KPipeline
            
            logger.info(f"Initializing Kokoro TTS with lang_code='{self.lang_code}', voice='{self.voice}'")
            
            # Initialize the pipeline in a thread to avoid blocking
            self._pipeline = await asyncio.to_thread(
                lambda: KPipeline(lang_code=self.lang_code)
            )
            
            self._initialized = True
            logger.info("Kokoro TTS pipeline initialized successfully")
            
        except ImportError as e:
            raise ImportError(
                "Kokoro TTS is not installed. Install with: pip install kokoro>=0.9.4 soundfile"
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro TTS: {e}")
            raise
            
        await super().initialize()
    
    async def cleanup(self) -> None:
        """Clean up the TTS pipeline."""
        if self._pipeline is not None:
            # Kokoro doesn't require explicit cleanup, but we'll reset the reference
            self._pipeline = None
            self._initialized = False
            logger.info("Kokoro TTS pipeline cleaned up")
        await super().cleanup()
    
    async def process(self, data: Any) -> AsyncGenerator[Tuple[np.ndarray, int], None]:
        """
        Process text input and generate speech audio.
        
        Args:
            data: Text string or stream of text strings to synthesize
            
        Yields:
            Audio data as numpy array with sample rate tuples
        """
        if not self._initialized:
            await self.initialize()
            
        if hasattr(data, '__aiter__'):
            # Handle streaming input
            async for result in self._process_stream(data):
                yield result
        else:
            # Handle single text input
            result = await self._process_single(data)
            if result:
                yield result
    
    async def _process_stream(self, data_stream: AsyncGenerator[Any, None]) -> AsyncGenerator[Tuple[np.ndarray, int], None]:
        """Process a stream of text inputs."""
        async for text_data in data_stream:
            # Extract text from various input formats
            text = self._extract_text(text_data)
            if text:
                logger.info(f"🎙️ Kokoro TTS: Starting synthesis for text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
                
                if self.stream_chunks:
                    # Stream individual audio chunks
                    chunk_count = 0
                    total_audio_duration = 0.0
                    async for audio_chunk in self._synthesize_streaming(text):
                        chunk_count += 1
                        chunk_duration = audio_chunk[0].shape[-1] / self.sample_rate
                        total_audio_duration += chunk_duration
                        logger.info(f"🎙️ Kokoro TTS: Streaming chunk {chunk_count}, duration={chunk_duration:.2f}s, total={total_audio_duration:.2f}s")
                        yield audio_chunk
                    logger.info(f"🎙️ Kokoro TTS: Completed synthesis - {chunk_count} chunks, total={total_audio_duration:.2f}s")
                else:
                    # Generate complete audio for this text
                    audio_result = await self._process_single(text)
                    if audio_result:
                        duration = audio_result[0].shape[-1] / self.sample_rate
                        logger.info(f"🎙️ Kokoro TTS: Generated complete audio: {duration:.2f}s")
                        yield audio_result
    
    async def _process_single(self, data: Any) -> Optional[Tuple[np.ndarray, int]]:
        """Process a single text input."""
        text = self._extract_text(data)
        if not text:
            return None
            
        if self.stream_chunks:
            # Collect all chunks into a single audio array
            audio_chunks = []
            async for audio_chunk, _ in self._synthesize_streaming(text):
                audio_chunks.append(audio_chunk)
            
            if audio_chunks:
                # Concatenate all chunks
                full_audio = np.concatenate(audio_chunks, axis=-1)
                return (full_audio, self.sample_rate)
            return None
        else:
            # Generate complete audio at once
            audio_chunks = []
            async for audio_chunk, _ in self._synthesize_streaming(text):
                audio_chunks.append(audio_chunk)
            
            if audio_chunks:
                full_audio = np.concatenate(audio_chunks, axis=-1)
                return (full_audio, self.sample_rate)
            return None
    
    def _extract_text(self, data: Any) -> Optional[str]:
        """Extract text from various input formats."""
        if isinstance(data, str):
            return data
        elif isinstance(data, tuple) and len(data) > 0:
            # Handle (text, metadata) tuples
            return str(data[0]) if data[0] else None
        elif hasattr(data, 'get') and 'text' in data:
            # Handle dict with text field
            return data.get('text')
        elif hasattr(data, 'text'):
            # Handle objects with text attribute
            return getattr(data, 'text', None)
        else:
            # Try to convert to string
            try:
                text = str(data).strip()
                return text if text else None
            except:
                logger.warning(f"Could not extract text from data: {type(data)}")
                return None
    
    async def _synthesize_streaming(self, text: str) -> AsyncGenerator[Tuple[np.ndarray, int], None]:
        """Generate audio chunks for the given text."""
        try:
            logger.info(f"🎙️ Kokoro TTS: Starting streaming synthesis for: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            # Run synthesis in a thread to avoid blocking
            generator = await asyncio.to_thread(
                self._create_generator, text
            )
            
            # Process each generated chunk
            chunk_count = 0
            total_samples = 0
            for i, (graphemes, phonemes, audio) in enumerate(generator):
                chunk_count += 1
                
                # Ensure audio is a numpy array
                if not isinstance(audio, np.ndarray):
                    audio = np.array(audio, dtype=np.float32)
                
                # Ensure audio is properly shaped (add channel dimension if needed)
                if audio.ndim == 1:
                    audio = audio.reshape(1, -1)  # Shape: (1, samples)
                elif audio.ndim > 2:
                    audio = audio.reshape(-1)  # Flatten to 1D
                    audio = audio.reshape(1, -1)  # Then add channel dimension
                
                chunk_samples = audio.shape[-1]
                total_samples += chunk_samples
                chunk_duration = chunk_samples / self.sample_rate
                total_duration = total_samples / self.sample_rate
                
                logger.info(f"🎙️ Kokoro TTS: Chunk {chunk_count}: '{graphemes[:30]}{'...' if len(graphemes) > 30 else ''}' "
                          f"-> {chunk_duration:.2f}s ({chunk_samples} samples) | Total: {total_duration:.2f}s")
                
                yield (audio, self.sample_rate)
                
            logger.info(f"🎙️ Kokoro TTS: Synthesis complete - {chunk_count} chunks, {total_duration:.2f}s total audio")
                
        except Exception as e:
            logger.error(f"Error during Kokoro TTS synthesis: {e}")
            raise
    
    def _create_generator(self, text: str):
        """Create the Kokoro generator (runs in thread)."""
        return self._pipeline(
            text,
            voice=self.voice,
            speed=self.speed,
            split_pattern=self.split_pattern
        )
    
    def get_config(self) -> dict:
        """Get node configuration."""
        config = super().get_config()
        config.update({
            "lang_code": self.lang_code,
            "voice": self.voice,
            "speed": self.speed,
            "split_pattern": self.split_pattern,
            "sample_rate": self.sample_rate,
            "stream_chunks": self.stream_chunks,
        })
        return config