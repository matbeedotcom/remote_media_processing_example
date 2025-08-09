#!/usr/bin/env python3
"""
RAW10 DataChannel Track for WebRTC

Transfers RAW10 sensor data via WebRTC data channels instead of video streams.
This provides uncompressed, full-quality sensor data for VLBI and astrophotography
applications requiring maximum precision.

Features:
- RAW10/RAW8 capture from Picamera2
- Chunked transfer over data channels
- Compression options (zlib, lz4)
- Frame synchronization
- Error recovery
"""

import asyncio
import numpy as np
import time
import json
import zlib
import struct
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import threading
from queue import Queue, Empty
from enum import Enum

try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

from picamera2 import Picamera2
from aiortc import RTCPeerConnection, RTCDataChannel

logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """Compression methods for data transfer"""
    NONE = "none"
    ZLIB = "zlib"
    LZ4 = "lz4"


@dataclass
class RAW10FrameHeader:
    """Header for RAW10 frame transmission"""
    frame_id: int
    timestamp: float
    width: int
    height: int
    bit_depth: int
    compression: str
    compressed_size: int
    uncompressed_size: int
    chunk_size: int = 65536  # 64KB chunks for data channel
    total_chunks: int = 0
    camera_id: int = 0
    sequence_number: int = 0
    
    def to_bytes(self) -> bytes:
        """Serialize header to bytes"""
        header_json = json.dumps(asdict(self))
        return header_json.encode('utf-8')
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'RAW10FrameHeader':
        """Deserialize header from bytes"""
        header_dict = json.loads(data.decode('utf-8'))
        return cls(**header_dict)


class RAW10DataChannelTrack:
    """
    Handles RAW10 image capture and transmission over WebRTC data channels.
    
    This replaces the video track with direct sensor data transmission,
    providing uncompressed or losslessly compressed RAW data.
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (3280, 2464),
        bit_depth: int = 10,
        framerate: float = 15.0,
        camera_num: int = 0,
        compression: CompressionType = CompressionType.ZLIB,
        chunk_size: int = 65536,
        max_queue_size: int = 5
    ):
        """
        Initialize RAW10 data channel track.
        
        Args:
            resolution: Capture resolution (width, height)
            bit_depth: Bit depth (8 or 10)
            framerate: Target framerate
            camera_num: Camera index
            compression: Compression type for transfer
            chunk_size: Size of chunks for data channel transfer
            max_queue_size: Maximum frames in queue
        """
        self.resolution = resolution
        self.bit_depth = bit_depth
        self.framerate = framerate
        self.camera_num = camera_num
        self.compression = compression
        self.chunk_size = chunk_size
        self.max_queue_size = max_queue_size
        
        # Camera and capture state
        self.picam2: Optional[Picamera2] = None
        self.capture_thread: Optional[threading.Thread] = None
        self.capture_queue = Queue(maxsize=max_queue_size)
        self.running = False
        
        # Data channel state
        self.data_channel: Optional[RTCDataChannel] = None
        self.frame_counter = 0
        self.sequence_number = 0
        
        # Performance monitoring
        self.stats = {
            'frames_captured': 0,
            'frames_sent': 0,
            'bytes_sent': 0,
            'compression_ratio': 0.0,
            'capture_fps': 0.0,
            'send_fps': 0.0
        }
        self.last_stats_time = time.time()
        self.last_capture_time = time.time()
        self.last_send_time = time.time()
        
        # Optimal framerates for different modes
        self.optimal_framerates = {
            (2560, 400, 8): 150.0,
            (5120, 720, 8): 50.0,
            (5120, 800, 8): 45.0,
            (2560, 400, 10): 110.0,
            (5120, 720, 10): 40.0,
            (5120, 800, 10): 35.0
        }
        
    def initialize_camera(self):
        """Initialize Picamera2 for RAW capture"""
        try:
            self.picam2 = Picamera2(camera_num=self.camera_num)
            
            # Find best sensor mode
            sensor_modes = self.picam2.sensor_modes
            best_mode = None
            
            for mode in sensor_modes:
                if mode['size'] == self.resolution and mode['bit_depth'] == self.bit_depth:
                    best_mode = mode
                    break
            
            if not best_mode:
                for mode in sensor_modes:
                    if mode['size'] == self.resolution:
                        best_mode = mode
                        logger.warning(f"Bit depth {self.bit_depth} not found, using {mode['bit_depth']}")
                        break
            
            if not best_mode and sensor_modes:
                best_mode = sensor_modes[0]
                logger.warning(f"Using default mode: {best_mode['size']} at {best_mode['bit_depth']}-bit")
            
            # Use optimal framerate if available
            key = (best_mode['size'][0], best_mode['size'][1], best_mode['bit_depth'])
            if key in self.optimal_framerates:
                self.framerate = self.optimal_framerates[key]
                logger.info(f"Using optimal framerate: {self.framerate}fps")
            
            # Configure for raw capture
            raw_format = best_mode['format']
            
            config = self.picam2.create_still_configuration(
                main={"size": self.resolution, "format": "RGB888"},
                raw={"size": best_mode['size'], "format": raw_format},
                buffer_count=2
            )
            
            self.picam2.configure(config)
            self.picam2.start()
            
            # Allow camera to warm up
            time.sleep(2)
            
            # Set manual controls
            controls = {}
            available_controls = self.picam2.camera_controls
            
            if "AeEnable" in available_controls:
                controls["AeEnable"] = False
            if "AwbEnable" in available_controls:
                controls["AwbEnable"] = False
            if "AnalogueGain" in available_controls:
                controls["AnalogueGain"] = 1.0
            if "ExposureTime" in available_controls:
                controls["ExposureTime"] = int(1000000 / self.framerate * 0.8)  # 80% of frame time
            
            if controls:
                self.picam2.set_controls(controls)
            
            logger.info(f"Camera {self.camera_num} initialized: {best_mode['size']} @ {self.bit_depth}-bit")
            
        except Exception as e:
            logger.error(f"Failed to initialize camera {self.camera_num}: {e}")
            raise
    
    def start_capture(self):
        """Start the capture thread"""
        if not self.running:
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.start()
            logger.info(f"Started RAW10 capture thread for camera {self.camera_num}")
    
    def stop_capture(self):
        """Stop the capture thread"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=5)
            self.capture_thread = None
        
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()
            self.picam2 = None
        
        logger.info(f"Stopped RAW10 capture for camera {self.camera_num}")
    
    def _capture_loop(self):
        """Capture loop running in separate thread"""
        frame_interval = 1.0 / self.framerate
        next_frame_time = time.time()
        
        while self.running:
            try:
                # Capture frame
                raw_data = self._capture_raw_frame()
                
                if raw_data is not None:
                    # Add to queue if not full
                    if not self.capture_queue.full():
                        self.capture_queue.put(raw_data, block=False)
                        self.stats['frames_captured'] += 1
                        
                        # Update capture FPS
                        current_time = time.time()
                        if current_time - self.last_capture_time > 1.0:
                            self.stats['capture_fps'] = self.stats['frames_captured'] / (current_time - self.last_stats_time)
                            self.last_capture_time = current_time
                    else:
                        # Drop oldest frame if queue is full
                        try:
                            self.capture_queue.get_nowait()
                            self.capture_queue.put(raw_data, block=False)
                            logger.debug("Dropped oldest frame from queue")
                        except Empty:
                            pass
                
                # Maintain framerate
                next_frame_time += frame_interval
                sleep_time = next_frame_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # Running behind, reset timing
                    next_frame_time = time.time()
                    
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)
    
    def _capture_raw_frame(self) -> Optional[Dict[str, Any]]:
        """Capture a single RAW frame"""
        if not self.picam2:
            return None
        
        try:
            # Capture raw array
            raw_data = self.picam2.capture_array("raw")
            
            # Fix alternating columns if needed (Arducam monochrome issue)
            raw_data = self._fix_alternating_columns(raw_data)
            
            # Get metadata
            metadata = self.picam2.capture_metadata()
            
            frame_data = {
                'data': raw_data,
                'timestamp': time.time(),
                'frame_id': self.frame_counter,
                'metadata': {
                    'exposure_time': metadata.get('ExposureTime', 0),
                    'analogue_gain': metadata.get('AnalogueGain', 1.0),
                    'digital_gain': metadata.get('DigitalGain', 1.0),
                    'sensor_temperature': metadata.get('SensorTemperature', 0),
                }
            }
            
            self.frame_counter += 1
            return frame_data
            
        except Exception as e:
            logger.error(f"Failed to capture frame: {e}")
            return None
    
    def _fix_alternating_columns(self, image_data: np.ndarray) -> np.ndarray:
        """Fix alternating column pattern from some sensors"""
        if len(image_data.shape) != 2 or image_data.shape[1] < 10:
            return image_data
        
        # Check for alternating pattern
        first_row = image_data[0, :min(100, image_data.shape[1])]
        even_cols = first_row[::2]
        odd_cols = first_row[1::2]
        
        even_zero_ratio = np.sum(even_cols == 0) / len(even_cols)
        odd_nonzero_ratio = np.sum(odd_cols > 0) / len(odd_cols)
        
        if even_zero_ratio > 0.8 and odd_nonzero_ratio > 0.5:
            logger.debug("Applying alternating column fix")
            # Remove zero columns
            return image_data[:, 1::2]
        
        return image_data
    
    async def set_data_channel(self, channel: RTCDataChannel):
        """Set the data channel for transmission"""
        self.data_channel = channel
        logger.info(f"Data channel set for camera {self.camera_num}")
        
        # Start sending frames
        asyncio.create_task(self._send_frames())
    
    async def _send_frames(self):
        """Send frames over the data channel"""
        while self.data_channel and self.data_channel.readyState == "open":
            try:
                # Get frame from queue (non-blocking with timeout)
                frame_data = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.capture_queue.get(timeout=0.1)
                )
                
                if frame_data:
                    await self._send_frame(frame_data)
                    
            except Empty:
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error sending frame: {e}")
                await asyncio.sleep(0.1)
    
    async def _send_frame(self, frame_data: Dict[str, Any]):
        """Send a single frame over the data channel"""
        if not self.data_channel or self.data_channel.readyState != "open":
            return
        
        raw_array = frame_data['data']
        
        # Compress if needed
        compressed_data, compression_used = self._compress_data(raw_array)
        
        # Create header
        header = RAW10FrameHeader(
            frame_id=frame_data['frame_id'],
            timestamp=frame_data['timestamp'],
            width=raw_array.shape[1] if len(raw_array.shape) > 1 else raw_array.shape[0],
            height=raw_array.shape[0] if len(raw_array.shape) > 1 else 1,
            bit_depth=self.bit_depth,
            compression=compression_used,
            compressed_size=len(compressed_data),
            uncompressed_size=raw_array.nbytes,
            chunk_size=self.chunk_size,
            total_chunks=(len(compressed_data) + self.chunk_size - 1) // self.chunk_size,
            camera_id=self.camera_num,
            sequence_number=self.sequence_number
        )
        
        try:
            # Send header
            header_bytes = header.to_bytes()
            header_message = struct.pack('!BH', 0x01, len(header_bytes)) + header_bytes  # Type 1: Header
            self.data_channel.send(header_message)
            
            # Send data in chunks
            for chunk_idx in range(header.total_chunks):
                start = chunk_idx * self.chunk_size
                end = min(start + self.chunk_size, len(compressed_data))
                chunk_data = compressed_data[start:end]
                
                # Chunk message: Type 2 + frame_id + chunk_idx + data
                chunk_message = struct.pack('!BII', 0x02, frame_data['frame_id'], chunk_idx) + chunk_data
                self.data_channel.send(chunk_message)
                
                # Small delay to avoid overwhelming the channel
                if chunk_idx % 10 == 0:
                    await asyncio.sleep(0.001)
            
            # Send metadata (optional)
            metadata_bytes = json.dumps(frame_data['metadata']).encode('utf-8')
            metadata_message = struct.pack('!BHI', 0x03, len(metadata_bytes), frame_data['frame_id']) + metadata_bytes
            self.data_channel.send(metadata_message)
            
            # Update stats
            self.stats['frames_sent'] += 1
            self.stats['bytes_sent'] += len(compressed_data) + len(header_bytes) + len(metadata_bytes)
            self.stats['compression_ratio'] = len(compressed_data) / header.uncompressed_size
            
            # Update send FPS
            current_time = time.time()
            if current_time - self.last_send_time > 1.0:
                self.stats['send_fps'] = self.stats['frames_sent'] / (current_time - self.last_stats_time)
                self.last_send_time = current_time
            
            self.sequence_number += 1
            
            logger.debug(f"Sent frame {frame_data['frame_id']}: {header.total_chunks} chunks, "
                        f"{header.compressed_size} bytes ({compression_used})")
            
        except Exception as e:
            logger.error(f"Failed to send frame {frame_data['frame_id']}: {e}")
    
    def _compress_data(self, data: np.ndarray) -> Tuple[bytes, str]:
        """Compress raw data for transmission"""
        raw_bytes = data.tobytes()
        
        if self.compression == CompressionType.NONE:
            return raw_bytes, "none"
        
        elif self.compression == CompressionType.ZLIB:
            compressed = zlib.compress(raw_bytes, level=1)  # Fast compression
            return compressed, "zlib"
        
        elif self.compression == CompressionType.LZ4 and HAS_LZ4:
            compressed = lz4.frame.compress(raw_bytes, compression_level=0)  # Fastest
            return compressed, "lz4"
        
        else:
            # Fallback to zlib if LZ4 not available
            compressed = zlib.compress(raw_bytes, level=1)
            return compressed, "zlib"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return self.stats.copy()


class RAW10DataChannelReceiver:
    """
    Receives and reconstructs RAW10 frames from data channel.
    
    This is the server-side component that receives the RAW data
    and reconstructs frames for processing.
    """
    
    def __init__(self, on_frame_callback=None):
        """
        Initialize receiver.
        
        Args:
            on_frame_callback: Callback function for completed frames
        """
        self.on_frame_callback = on_frame_callback
        self.pending_frames: Dict[int, Dict[str, Any]] = {}
        self.completed_frames = Queue()
        
    async def on_datachannel_message(self, message: bytes):
        """Handle incoming data channel message"""
        if len(message) < 1:
            return
        
        message_type = message[0]
        
        try:
            if message_type == 0x01:  # Header
                await self._handle_header(message[1:])
            elif message_type == 0x02:  # Chunk
                await self._handle_chunk(message[1:])
            elif message_type == 0x03:  # Metadata
                await self._handle_metadata(message[1:])
                
        except Exception as e:
            logger.error(f"Error handling message type {message_type}: {e}")
    
    async def _handle_header(self, data: bytes):
        """Handle frame header message"""
        header_length = struct.unpack('!H', data[:2])[0]
        header_bytes = data[2:2+header_length]
        header = RAW10FrameHeader.from_bytes(header_bytes)
        
        # Initialize frame reconstruction
        self.pending_frames[header.frame_id] = {
            'header': header,
            'chunks': {},
            'metadata': None,
            'start_time': time.time()
        }
        
        logger.debug(f"Received header for frame {header.frame_id}: "
                    f"{header.width}x{header.height} @ {header.bit_depth}-bit")
    
    async def _handle_chunk(self, data: bytes):
        """Handle frame data chunk"""
        frame_id, chunk_idx = struct.unpack('!II', data[:8])
        chunk_data = data[8:]
        
        if frame_id in self.pending_frames:
            self.pending_frames[frame_id]['chunks'][chunk_idx] = chunk_data
            
            # Check if frame is complete
            frame_info = self.pending_frames[frame_id]
            if len(frame_info['chunks']) == frame_info['header'].total_chunks:
                await self._reconstruct_frame(frame_id)
    
    async def _handle_metadata(self, data: bytes):
        """Handle frame metadata"""
        metadata_length = struct.unpack('!H', data[:2])[0]
        frame_id = struct.unpack('!I', data[2:6])[0]
        metadata_bytes = data[6:6+metadata_length]
        
        if frame_id in self.pending_frames:
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            self.pending_frames[frame_id]['metadata'] = metadata
    
    async def _reconstruct_frame(self, frame_id: int):
        """Reconstruct complete frame from chunks"""
        frame_info = self.pending_frames[frame_id]
        header = frame_info['header']
        
        # Combine chunks
        combined_data = b''
        for i in range(header.total_chunks):
            if i in frame_info['chunks']:
                combined_data += frame_info['chunks'][i]
            else:
                logger.error(f"Missing chunk {i} for frame {frame_id}")
                return
        
        # Decompress if needed
        if header.compression == "zlib":
            raw_bytes = zlib.decompress(combined_data)
        elif header.compression == "lz4" and HAS_LZ4:
            raw_bytes = lz4.frame.decompress(combined_data)
        else:
            raw_bytes = combined_data
        
        # Reconstruct numpy array
        dtype = np.uint16 if header.bit_depth > 8 else np.uint8
        raw_array = np.frombuffer(raw_bytes, dtype=dtype)
        
        # Reshape if 2D
        if header.height > 1:
            raw_array = raw_array.reshape((header.height, header.width))
        
        # Create frame data
        frame_data = {
            'frame_id': frame_id,
            'timestamp': header.timestamp,
            'camera_id': header.camera_id,
            'data': raw_array,
            'metadata': frame_info['metadata'],
            'reconstruction_time': time.time() - frame_info['start_time']
        }
        
        # Clean up
        del self.pending_frames[frame_id]
        
        # Callback or queue
        if self.on_frame_callback:
            await self.on_frame_callback(frame_data)
        else:
            self.completed_frames.put(frame_data)
        
        logger.debug(f"Reconstructed frame {frame_id} in {frame_data['reconstruction_time']:.3f}s")


def create_raw10_datachannel_track(
    camera_num: int = 0,
    resolution: Tuple[int, int] = (3280, 2464),
    bit_depth: int = 10,
    framerate: float = 15.0,
    compression: str = "zlib"
) -> RAW10DataChannelTrack:
    """
    Factory function to create RAW10 data channel track.
    
    Args:
        camera_num: Camera index
        resolution: Capture resolution
        bit_depth: Bit depth (8 or 10)
        framerate: Target framerate
        compression: Compression type ("none", "zlib", "lz4")
    
    Returns:
        Configured RAW10DataChannelTrack instance
    """
    compression_type = CompressionType.ZLIB
    if compression.lower() == "none":
        compression_type = CompressionType.NONE
    elif compression.lower() == "lz4" and HAS_LZ4:
        compression_type = CompressionType.LZ4
    
    track = RAW10DataChannelTrack(
        resolution=resolution,
        bit_depth=bit_depth,
        framerate=framerate,
        camera_num=camera_num,
        compression=compression_type
    )
    
    return track


if __name__ == "__main__":
    # Test the RAW10 data channel track
    import asyncio
    
    async def test_capture():
        track = create_raw10_datachannel_track(
            camera_num=0,
            resolution=(2560, 400),  # Fast mode
            bit_depth=8,
            framerate=50.0,
            compression="zlib"
        )
        
        try:
            track.initialize_camera()
            track.start_capture()
            
            # Let it run for a few seconds
            await asyncio.sleep(5)
            
            stats = track.get_stats()
            print(f"Capture stats: {stats}")
            
        finally:
            track.stop_capture()
    
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_capture())