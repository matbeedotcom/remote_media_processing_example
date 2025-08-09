#!/usr/bin/env python3
"""
Hybrid RAW10 + Video WebRTC Client

Sends both RAW10 data through data channels AND processed video frames
for ChAruco detection. This allows the pipeline to work while still
transferring uncompressed sensor data.
"""

import asyncio
import json
import logging
import argparse
import sys
import numpy as np
import cv2
from typing import Optional, Dict, Any
from pathlib import Path
import time

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel, VideoStreamTrack
import aiohttp
from av import VideoFrame

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from raw10_datachannel_track import RAW10DataChannelTrack, create_raw10_datachannel_track

logger = logging.getLogger(__name__)


class ProcessedVideoTrack(VideoStreamTrack):
    """
    Video track that sends processed frames from RAW10 capture.
    This provides video for the pipeline while RAW data goes through data channels.
    """
    
    def __init__(self, raw_track: RAW10DataChannelTrack, width=1280, height=200):
        super().__init__()
        self.raw_track = raw_track
        self.width = width
        self.height = height
        self.frame_count = 0
        
    async def recv(self):
        """Get processed frame from RAW capture"""
        pts, time_base = await self.next_timestamp()
        
        # Try to get a frame from the RAW capture queue
        frame = None
        if self.raw_track and hasattr(self.raw_track, 'capture_queue'):
            try:
                # Non-blocking get from capture queue
                import queue
                frame_data = self.raw_track.capture_queue.get_nowait()
                
                if frame_data and 'data' in frame_data:
                    raw_array = frame_data['data']
                    
                    # Convert RAW to displayable format
                    if len(raw_array.shape) == 2:
                        # Normalize to 8-bit
                        if self.raw_track.bit_depth > 8:
                            max_val = (1 << self.raw_track.bit_depth) - 1
                            normalized = (raw_array.astype(np.float32) / max_val * 255).astype(np.uint8)
                        else:
                            normalized = raw_array.astype(np.uint8)
                        
                        # Convert grayscale to RGB
                        frame = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
                    else:
                        frame = raw_array
                    
                    # Resize with aspect ratio preservation
                    if frame.shape[:2] != (self.height, self.width):
                        frame = self._resize_preserve_aspect_ratio(frame, self.width, self.height)
                    
                    # Put frame back for data channel processing
                    try:
                        self.raw_track.capture_queue.put(frame_data, block=False)
                    except:
                        pass
                        
            except queue.Empty:
                pass
            except Exception as e:
                logger.debug(f"Error getting frame from RAW track: {e}")
        
        # If no RAW frame available, create a placeholder
        if frame is None:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            # Add frame counter for debugging
            cv2.putText(frame, f"Frame {self.frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        self.frame_count += 1
        
        # Convert to VideoFrame
        av_frame = VideoFrame.from_ndarray(frame, format='rgb24')
        av_frame.pts = pts
        av_frame.time_base = time_base
        
        return av_frame
    
    def _resize_preserve_aspect_ratio(self, image, target_width, target_height):
        """
        Resize image while preserving aspect ratio, padding with black if needed.
        
        Args:
            image: Input image (numpy array)
            target_width: Target width
            target_height: Target height
            
        Returns:
            Resized image with preserved aspect ratio
        """
        h, w = image.shape[:2]
        
        # Calculate scaling factor to fit within target dimensions
        scale_w = target_width / w
        scale_h = target_height / h
        scale = min(scale_w, scale_h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize the image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create output image with target dimensions (black background)
        if len(image.shape) == 3:
            output = np.zeros((target_height, target_width, image.shape[2]), dtype=image.dtype)
        else:
            output = np.zeros((target_height, target_width), dtype=image.dtype)
        
        # Calculate position to center the resized image
        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2
        
        # Place the resized image in the center
        if len(image.shape) == 3:
            output[y_offset:y_offset+new_h, x_offset:x_offset+new_w, :] = resized
        else:
            output[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return output


class HybridRAW10Client:
    """
    Hybrid WebRTC client that sends both RAW10 data and processed video.
    """
    
    def __init__(
        self,
        server_url: str,
        camera_num: int = 0,
        resolution: tuple = (3280, 2464),
        bit_depth: int = 10,
        framerate: float = 15.0,
        compression: str = "zlib",
        video_width: int = 1280,
        video_height: int = 200
    ):
        """
        Initialize hybrid client.
        
        Args:
            server_url: WebSocket server URL
            camera_num: Camera index
            resolution: RAW capture resolution
            bit_depth: Bit depth (8 or 10)
            framerate: RAW capture framerate
            compression: Compression for data channel
            video_width: Width for video stream
            video_height: Height for video stream
        """
        self.server_url = server_url
        self.camera_num = camera_num
        self.resolution = resolution
        self.bit_depth = bit_depth
        self.framerate = framerate
        self.compression = compression
        self.video_width = video_width
        self.video_height = video_height
        
        # WebRTC components
        self.pc: Optional[RTCPeerConnection] = None
        self.raw_track: Optional[RAW10DataChannelTrack] = None
        self.video_track: Optional[ProcessedVideoTrack] = None
        self.data_channel: Optional[RTCDataChannel] = None
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        
        # Connection state
        self.connected = False
        self.running = False
        
    async def connect(self):
        """Establish WebRTC connection with server"""
        try:
            # Connect to WebSocket
            session = aiohttp.ClientSession()
            self.ws = await session.ws_connect(self.server_url)
            logger.info(f"WebSocket connected to {self.server_url}")
            
            # Create peer connection
            self.pc = RTCPeerConnection()
            
            # Set up connection state handlers
            @self.pc.on("connectionstatechange")
            async def on_connection_state_change():
                logger.info(f"Connection state: {self.pc.connectionState}")
                if self.pc.connectionState == "connected":
                    self.connected = True
                elif self.pc.connectionState in ["failed", "closed"]:
                    self.connected = False
            
            # Initialize RAW10 track first
            self.raw_track = create_raw10_datachannel_track(
                camera_num=self.camera_num,
                resolution=self.resolution,
                bit_depth=self.bit_depth,
                framerate=self.framerate,
                compression=self.compression
            )
            
            # Initialize camera
            self.raw_track.initialize_camera()
            self.raw_track.start_capture()
            
            # Create data channel for RAW10
            self.data_channel = self.pc.createDataChannel(
                "raw10",
                ordered=True,
                maxRetransmits=3,
                protocol="raw10-v1"
            )
            
            @self.data_channel.on("open")
            async def on_datachannel_open():
                logger.info(f"Data channel opened for camera {self.camera_num}")
                # Start RAW10 transmission
                if self.raw_track:
                    await self.raw_track.set_data_channel(self.data_channel)
            
            # Add processed video track that uses RAW frames
            self.video_track = ProcessedVideoTrack(
                self.raw_track,
                width=self.video_width,
                height=self.video_height
            )
            self.pc.addTrack(self.video_track)
            logger.info(f"Added processed video track: {self.video_width}x{self.video_height}")
            
            # Create offer
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)
            
            # Send offer via WebSocket
            await self.ws.send_json({
                "type": "offer",
                "sdp": self.pc.localDescription.sdp,
                "offer_type": self.pc.localDescription.type,
                "client_info": {
                    "mode": "hybrid_raw10",
                    "camera_id": self.camera_num,
                    "raw_resolution": list(self.resolution),
                    "video_resolution": [self.video_width, self.video_height],
                    "bit_depth": self.bit_depth,
                    "framerate": self.framerate,
                    "compression": self.compression
                }
            })
            
            # Handle WebSocket messages
            asyncio.create_task(self._handle_ws_messages())
            
            logger.info(f"Hybrid mode: RAW10 ({self.resolution[0]}x{self.resolution[1]}) + "
                       f"Video ({self.video_width}x{self.video_height})")
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            await self.disconnect()
            raise
    
    async def _handle_ws_messages(self):
        """Handle incoming WebSocket messages"""
        try:
            async for msg in self.ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    
                    if data.get("type") == "answer":
                        # Set remote description
                        answer = RTCSessionDescription(
                            sdp=data["sdp"],
                            type=data.get("answer_type", "answer")
                        )
                        await self.pc.setRemoteDescription(answer)
                        logger.info("Answer received and set")
                        self.connected = True
                    
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {msg.data}")
                    break
                    
        except Exception as e:
            logger.error(f"Error handling WebSocket messages: {e}")
    
    async def run(self):
        """Run the client until stopped"""
        self.running = True
        
        try:
            while self.running and self.connected:
                # Monitor connection and stats
                if self.raw_track:
                    stats = self.raw_track.get_stats()
                    if stats['frames_sent'] % 100 == 0 and stats['frames_sent'] > 0:
                        logger.info(f"📊 Camera {self.camera_num} - "
                                   f"RAW captured: {stats['frames_captured']}, "
                                   f"sent: {stats['frames_sent']}, "
                                   f"compression: {stats['compression_ratio']:.2f}, "
                                   f"FPS: {stats['send_fps']:.1f}")
                
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in run loop: {e}")
        finally:
            await self.disconnect()
    
    async def disconnect(self):
        """Disconnect and cleanup"""
        self.running = False
        self.connected = False
        
        # Stop RAW capture
        if self.raw_track:
            self.raw_track.stop_capture()
            self.raw_track = None
        
        # Close WebSocket
        if self.ws and not self.ws.closed:
            await self.ws.close()
            self.ws = None
        
        # Close peer connection
        if self.pc:
            await self.pc.close()
            self.pc = None
        
        logger.info(f"Disconnected camera {self.camera_num}")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Hybrid RAW10 + Video WebRTC Client")
    parser.add_argument("--server", "-s", default="ws://localhost:8085/ws",
                       help="WebSocket server URL")
    parser.add_argument("--camera", "-c", type=int, default=0,
                       help="Camera index")
    parser.add_argument("--width", "-w", type=int, default=2560,
                       help="RAW capture width")
    parser.add_argument("--height", "-H", type=int, default=400,
                       help="RAW capture height")
    parser.add_argument("--bit-depth", "-b", type=int, choices=[8, 10], default=8,
                       help="Bit depth")
    parser.add_argument("--fps", "-f", type=float, default=50.0,
                       help="Capture framerate")
    parser.add_argument("--compression", choices=["none", "zlib", "lz4"], default="zlib",
                       help="Compression type")
    parser.add_argument("--video-width", type=int, default=1280,
                       help="Video stream width")
    parser.add_argument("--video-height", type=int, default=200,
                       help="Video stream height")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("🚀 Starting Hybrid RAW10 + Video Client")
    logger.info(f"📷 Camera: {args.camera}")
    logger.info(f"📊 RAW: {args.width}x{args.height} @ {args.bit_depth}-bit, {args.fps} fps")
    logger.info(f"📺 Video: {args.video_width}x{args.video_height}")
    logger.info(f"🔗 Server: {args.server}")
    logger.info(f"🗜️ Compression: {args.compression}")
    
    client = HybridRAW10Client(
        server_url=args.server,
        camera_num=args.camera,
        resolution=(args.width, args.height),
        bit_depth=args.bit_depth,
        framerate=args.fps,
        compression=args.compression,
        video_width=args.video_width,
        video_height=args.video_height
    )
    
    await client.connect()
    
    # Wait for connection
    for _ in range(10):
        if client.connected:
            break
        await asyncio.sleep(0.5)
    
    if not client.connected:
        logger.error("Failed to establish connection")
        return
    
    await client.run()


if __name__ == "__main__":
    asyncio.run(main())