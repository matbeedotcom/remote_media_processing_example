#!/usr/bin/env python3
"""
RAW10 DataChannel Receiver Node

Server-side node for receiving and processing RAW10 sensor data
transmitted via WebRTC data channels. Integrates with the VLBI
pipeline for high-precision astronomical imaging.

Features:
- Receives chunked RAW10 data from data channels
- Reconstructs frames from compressed chunks
- Converts RAW bayer data to usable formats
- Integrates with existing VLBI pipeline
- Multi-camera synchronization
"""

import asyncio
import numpy as np
import time
import json
import zlib
import struct
import logging
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from pathlib import Path
import sys
from queue import Queue
import cv2

try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "remote_media_processing"))

from remotemedia.core.node import Node

logger = logging.getLogger(__name__)


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
    chunk_size: int
    total_chunks: int
    camera_id: int
    sequence_number: int
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'RAW10FrameHeader':
        """Deserialize header from bytes"""
        header_dict = json.loads(data.decode('utf-8'))
        return cls(**header_dict)


class RAW10ReceiverNode(Node):
    """
    Pipeline node for receiving and processing RAW10 data from WebRTC data channels.
    
    This node receives chunked RAW sensor data, reconstructs frames,
    and converts them to formats suitable for the VLBI pipeline.
    """
    
    def __init__(
        self,
        num_cameras: int = 4,
        enable_debayer: bool = False,
        bayer_pattern: str = 'BGGR',
        output_format: str = 'mono',  # 'mono', 'rgb', 'raw'
        sync_tolerance_ms: float = 50.0,
        name: Optional[str] = None
    ):
        """
        Initialize RAW10 receiver node.
        
        Args:
            num_cameras: Number of cameras expected
            enable_debayer: Apply debayering to RAW data
            bayer_pattern: Bayer pattern (BGGR, RGGB, GBRG, GRBG)
            output_format: Output format for frames
            sync_tolerance_ms: Synchronization tolerance in milliseconds
            name: Node name
        """
        super().__init__(name=name or "RAW10Receiver")
        
        self.num_cameras = num_cameras
        self.enable_debayer = enable_debayer
        self.bayer_pattern = bayer_pattern
        self.output_format = output_format
        self.sync_tolerance_ms = sync_tolerance_ms
        
        # Frame reconstruction state
        self.pending_frames: Dict[int, Dict[str, Any]] = {}
        self.camera_frames: Dict[int, Queue] = {i: Queue(maxsize=10) for i in range(num_cameras)}
        self.reconstructed_frames = Queue()
        
        # Synchronization state
        self.frame_sync_buffer: Dict[float, Dict[int, np.ndarray]] = {}
        self.last_sync_time = time.time()
        
        # Statistics
        self.stats = {
            'frames_received': 0,
            'frames_reconstructed': 0,
            'frames_synchronized': 0,
            'bytes_received': 0,
            'decompression_time': 0.0,
            'debayer_time': 0.0,
            'sync_success_rate': 0.0
        }
        
        # Data channel handlers
        self.data_channels: Dict[int, Any] = {}
        
        logger.info(f"🎯 RAW10Receiver initialized for {num_cameras} cameras")
    
    async def on_datachannel(self, channel, camera_id: int = 0):
        """
        Handle new data channel connection.
        
        Args:
            channel: WebRTC data channel
            camera_id: Camera identifier
        """
        self.data_channels[camera_id] = channel
        
        @channel.on("message")
        async def on_message(message):
            await self.on_datachannel_message(message, camera_id)
        
        @channel.on("close")
        async def on_close():
            logger.info(f"Data channel closed for camera {camera_id}")
            if camera_id in self.data_channels:
                del self.data_channels[camera_id]
        
        logger.info(f"📡 Data channel connected for camera {camera_id}")
    
    async def on_datachannel_message(self, message: bytes, camera_id: int):
        """
        Handle incoming data channel message.
        
        Args:
            message: Raw message bytes
            camera_id: Camera identifier
        """
        if len(message) < 1:
            return
        
        message_type = message[0]
        
        try:
            if message_type == 0x01:  # Header
                await self._handle_header(message[1:], camera_id)
            elif message_type == 0x02:  # Chunk
                await self._handle_chunk(message[1:], camera_id)
            elif message_type == 0x03:  # Metadata
                await self._handle_metadata(message[1:], camera_id)
            
            self.stats['bytes_received'] += len(message)
            
        except Exception as e:
            logger.error(f"Error handling message type {message_type} from camera {camera_id}: {e}")
    
    async def _handle_header(self, data: bytes, camera_id: int):
        """Handle frame header message"""
        header_length = struct.unpack('!H', data[:2])[0]
        header_bytes = data[2:2+header_length]
        header = RAW10FrameHeader.from_bytes(header_bytes)
        
        # Initialize frame reconstruction
        frame_key = f"{camera_id}_{header.frame_id}"
        self.pending_frames[frame_key] = {
            'header': header,
            'chunks': {},
            'metadata': None,
            'start_time': time.time(),
            'camera_id': camera_id
        }
        
        logger.debug(f"📋 Header received - Camera {camera_id}, Frame {header.frame_id}: "
                    f"{header.width}x{header.height} @ {header.bit_depth}-bit, "
                    f"{header.total_chunks} chunks")
    
    async def _handle_chunk(self, data: bytes, camera_id: int):
        """Handle frame data chunk"""
        frame_id, chunk_idx = struct.unpack('!II', data[:8])
        chunk_data = data[8:]
        
        frame_key = f"{camera_id}_{frame_id}"
        
        if frame_key in self.pending_frames:
            self.pending_frames[frame_key]['chunks'][chunk_idx] = chunk_data
            
            # Check if frame is complete
            frame_info = self.pending_frames[frame_key]
            if len(frame_info['chunks']) == frame_info['header'].total_chunks:
                await self._reconstruct_frame(frame_key)
    
    async def _handle_metadata(self, data: bytes, camera_id: int):
        """Handle frame metadata"""
        metadata_length = struct.unpack('!H', data[:2])[0]
        frame_id = struct.unpack('!I', data[2:6])[0]
        metadata_bytes = data[6:6+metadata_length]
        
        frame_key = f"{camera_id}_{frame_id}"
        
        if frame_key in self.pending_frames:
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            self.pending_frames[frame_key]['metadata'] = metadata
    
    async def _reconstruct_frame(self, frame_key: str):
        """Reconstruct complete frame from chunks"""
        frame_info = self.pending_frames[frame_key]
        header = frame_info['header']
        camera_id = frame_info['camera_id']
        
        start_time = time.time()
        
        try:
            # Combine chunks
            combined_data = b''
            for i in range(header.total_chunks):
                if i in frame_info['chunks']:
                    combined_data += frame_info['chunks'][i]
                else:
                    logger.error(f"Missing chunk {i} for frame {frame_key}")
                    return
            
            # Decompress if needed
            decompress_start = time.time()
            
            if header.compression == "zlib":
                raw_bytes = zlib.decompress(combined_data)
            elif header.compression == "lz4" and HAS_LZ4:
                raw_bytes = lz4.frame.decompress(combined_data)
            else:
                raw_bytes = combined_data
            
            self.stats['decompression_time'] += time.time() - decompress_start
            
            # Reconstruct numpy array
            dtype = np.uint16 if header.bit_depth > 8 else np.uint8
            raw_array = np.frombuffer(raw_bytes, dtype=dtype)
            
            # Reshape if 2D
            if header.height > 1:
                raw_array = raw_array.reshape((header.height, header.width))
            
            # Process the raw data
            processed_frame = await self._process_raw_frame(
                raw_array, 
                header.bit_depth,
                camera_id
            )
            
            # Create frame data
            frame_data = {
                'frame_id': header.frame_id,
                'timestamp': header.timestamp,
                'camera_id': camera_id,
                'data': processed_frame,
                'raw_data': raw_array if self.output_format == 'raw' else None,
                'metadata': frame_info['metadata'],
                'width': header.width,
                'height': header.height,
                'bit_depth': header.bit_depth,
                'reconstruction_time': time.time() - start_time
            }
            
            # Add to camera queue
            if camera_id < self.num_cameras:
                if not self.camera_frames[camera_id].full():
                    self.camera_frames[camera_id].put(frame_data)
                else:
                    # Drop oldest frame
                    try:
                        self.camera_frames[camera_id].get_nowait()
                        self.camera_frames[camera_id].put(frame_data)
                    except:
                        pass
            
            # Update stats
            self.stats['frames_reconstructed'] += 1
            
            # Clean up
            del self.pending_frames[frame_key]
            
            logger.debug(f"✅ Reconstructed frame {header.frame_id} from camera {camera_id} "
                        f"in {frame_data['reconstruction_time']:.3f}s")
            
            # Try to synchronize frames
            await self._try_sync_frames()
            
        except Exception as e:
            logger.error(f"Failed to reconstruct frame {frame_key}: {e}")
            if frame_key in self.pending_frames:
                del self.pending_frames[frame_key]
    
    async def _process_raw_frame(self, raw_array: np.ndarray, bit_depth: int, camera_id: int) -> np.ndarray:
        """
        Process raw frame data based on output format.
        
        Args:
            raw_array: Raw sensor data
            bit_depth: Bit depth of raw data
            camera_id: Camera identifier
            
        Returns:
            Processed frame array
        """
        if self.output_format == 'raw':
            return raw_array
        
        # Normalize to 8-bit for processing
        if bit_depth > 8:
            max_val = (1 << bit_depth) - 1
            normalized = (raw_array.astype(np.float32) / max_val * 255).astype(np.uint8)
        else:
            normalized = raw_array
        
        if self.output_format == 'mono':
            # Return as monochrome
            return normalized
        
        elif self.output_format == 'rgb' and self.enable_debayer:
            # Debayer the raw data
            debayer_start = time.time()
            
            # Use OpenCV debayering
            bayer_codes = {
                'BGGR': cv2.COLOR_BAYER_BG2RGB,
                'RGGB': cv2.COLOR_BAYER_RG2RGB,
                'GBRG': cv2.COLOR_BAYER_GB2RGB,
                'GRBG': cv2.COLOR_BAYER_GR2RGB
            }
            
            if self.bayer_pattern in bayer_codes:
                rgb_frame = cv2.cvtColor(normalized, bayer_codes[self.bayer_pattern])
            else:
                # Fallback to simple interpolation
                rgb_frame = self._simple_debayer(normalized)
            
            self.stats['debayer_time'] += time.time() - debayer_start
            
            return rgb_frame
        
        else:
            # Default to monochrome
            return normalized
    
    def _simple_debayer(self, raw_array: np.ndarray) -> np.ndarray:
        """Simple debayering using bilinear interpolation"""
        h, w = raw_array.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Extract color channels based on Bayer pattern
        if self.bayer_pattern == 'BGGR':
            rgb[0::2, 0::2, 2] = raw_array[0::2, 0::2]  # Blue
            rgb[0::2, 1::2, 1] = raw_array[0::2, 1::2]  # Green
            rgb[1::2, 0::2, 1] = raw_array[1::2, 0::2]  # Green
            rgb[1::2, 1::2, 0] = raw_array[1::2, 1::2]  # Red
        # Add other patterns as needed
        
        # Simple interpolation for missing values
        # This is a placeholder - real implementation would use better interpolation
        for c in range(3):
            # Fill in missing values with nearest neighbor
            mask = rgb[:, :, c] == 0
            if np.any(mask):
                indices = np.where(~mask)
                from scipy.interpolate import griddata
                points = np.column_stack(indices)
                values = rgb[indices[0], indices[1], c]
                grid_x, grid_y = np.mgrid[0:h, 0:w]
                rgb[:, :, c] = griddata(points, values, (grid_x, grid_y), method='nearest')
        
        return rgb
    
    async def _try_sync_frames(self):
        """Try to synchronize frames from all cameras"""
        # Check if we have frames from all cameras
        frames_available = sum(1 for q in self.camera_frames.values() if not q.empty())
        
        if frames_available < self.num_cameras:
            return  # Not enough frames
        
        # Get latest frame from each camera
        camera_frames = {}
        timestamps = []
        
        for cam_id in range(self.num_cameras):
            if not self.camera_frames[cam_id].empty():
                frame = self.camera_frames[cam_id].get()
                camera_frames[cam_id] = frame
                timestamps.append(frame['timestamp'])
        
        if len(camera_frames) < self.num_cameras:
            # Put frames back if we don't have all cameras
            for cam_id, frame in camera_frames.items():
                self.camera_frames[cam_id].put(frame)
            return
        
        # Check timestamp synchronization
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        sync_delta_ms = (max_ts - min_ts) * 1000
        
        if sync_delta_ms <= self.sync_tolerance_ms:
            # Frames are synchronized
            self.stats['frames_synchronized'] += 1
            
            # Create synchronized frame set
            sync_data = {
                'frames': [camera_frames[i]['data'] for i in range(self.num_cameras)],
                'raw_frames': [camera_frames[i].get('raw_data') for i in range(self.num_cameras)],
                'timestamp': min_ts,
                'sync_delta_ms': sync_delta_ms,
                'camera_metadata': [camera_frames[i].get('metadata', {}) for i in range(self.num_cameras)],
                'frame_ids': [camera_frames[i]['frame_id'] for i in range(self.num_cameras)]
            }
            
            # Add to output queue
            if not self.reconstructed_frames.full():
                self.reconstructed_frames.put(sync_data)
            
            # Update sync success rate
            total_attempts = self.stats['frames_reconstructed'] // self.num_cameras
            if total_attempts > 0:
                self.stats['sync_success_rate'] = self.stats['frames_synchronized'] / total_attempts
            
            logger.debug(f"🔄 Synchronized frame set with {sync_delta_ms:.1f}ms delta")
        else:
            # Frames not synchronized, put back newest frames
            logger.debug(f"⚠️ Frame sync failed: {sync_delta_ms:.1f}ms > {self.sync_tolerance_ms}ms")
            
            # Keep oldest frame, put back others
            oldest_idx = timestamps.index(min_ts)
            for cam_id, frame in camera_frames.items():
                if cam_id != oldest_idx:
                    self.camera_frames[cam_id].put(frame)
    
    async def process(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process pipeline data and inject RAW10 frames.
        
        Args:
            data: Input pipeline data
            
        Returns:
            Enhanced data with RAW10 frames
        """
        # Check for synchronized frames
        if not self.reconstructed_frames.empty():
            sync_data = self.reconstructed_frames.get()
            
            # Merge with existing data
            enhanced_data = data.copy() if data else {}
            enhanced_data['frames'] = sync_data['frames']
            enhanced_data['raw_frames'] = sync_data['raw_frames']
            enhanced_data['timestamp'] = sync_data['timestamp']
            enhanced_data['sync_delta_ms'] = sync_data['sync_delta_ms']
            enhanced_data['raw10_metadata'] = sync_data['camera_metadata']
            enhanced_data['frame_source'] = 'raw10_datachannel'
            
            # Add stats
            enhanced_data['raw10_stats'] = self.get_stats()
            
            return enhanced_data
        
        # Pass through if no RAW frames available
        return data
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return self.stats.copy()


def create_raw10_receiver_node(
    num_cameras: int = 4,
    output_format: str = 'mono',
    enable_debayer: bool = False
) -> RAW10ReceiverNode:
    """
    Factory function to create RAW10 receiver node.
    
    Args:
        num_cameras: Number of cameras
        output_format: Output format ('mono', 'rgb', 'raw')
        enable_debayer: Enable debayering
        
    Returns:
        Configured RAW10ReceiverNode
    """
    return RAW10ReceiverNode(
        num_cameras=num_cameras,
        enable_debayer=enable_debayer,
        output_format=output_format
    )


if __name__ == "__main__":
    # Test the receiver node
    import asyncio
    
    async def test_receiver():
        receiver = create_raw10_receiver_node(
            num_cameras=4,
            output_format='mono'
        )
        
        # Simulate receiving data
        logger.info("RAW10 Receiver Node initialized")
        
        # In real usage, this would be connected to WebRTC data channels
        await asyncio.sleep(1)
        
        stats = receiver.get_stats()
        print(f"Receiver stats: {stats}")
    
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_receiver())