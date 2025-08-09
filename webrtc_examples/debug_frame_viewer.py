#!/usr/bin/env python3
"""
Debug Frame Viewer for VLBI Pipeline

Saves debug frames from the hybrid client to inspect what's being sent
for ChAruco detection debugging.
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
sys.path.insert(0, str(Path(__file__).parent / "raspberrypi_webrtc_client"))
from raw10_datachannel_track import RAW10DataChannelTrack, create_raw10_datachannel_track

logger = logging.getLogger(__name__)

class DebugVideoTrack(VideoStreamTrack):
    """
    Debug video track that captures and saves frames for inspection.
    """
    
    def __init__(self, raw_track: RAW10DataChannelTrack, width=640, height=480, save_frames=True):
        super().__init__()
        self.raw_track = raw_track
        self.width = width
        self.height = height
        self.frame_count = 0
        self.save_frames = save_frames
        self.save_dir = Path("debug_frames")
        
        if self.save_frames:
            self.save_dir.mkdir(exist_ok=True)
            logger.info(f"🐛 Debug frames will be saved to {self.save_dir}")
        
    async def recv(self):
        """Get processed frame and save for debugging"""
        pts, time_base = await self.next_timestamp()
        
        # Try to get a frame from the RAW capture queue
        frame = None
        raw_data_info = ""
        
        if self.raw_track and hasattr(self.raw_track, 'capture_queue'):
            try:
                # Non-blocking get from capture queue
                import queue
                frame_data = self.raw_track.capture_queue.get_nowait()
                
                if frame_data and 'data' in frame_data:
                    raw_array = frame_data['data']
                    raw_data_info = f"RAW: {raw_array.shape}, dtype={raw_array.dtype}, min={raw_array.min()}, max={raw_array.max()}"
                    
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
                    
                    # Resize if needed
                    if frame.shape[:2] != (self.height, self.width):
                        frame = cv2.resize(frame, (self.width, self.height))
                    
                    # Put frame back for data channel processing
                    try:
                        self.raw_track.capture_queue.put(frame_data, block=False)
                    except:
                        pass
                        
            except queue.Empty:
                pass
            except Exception as e:
                logger.debug(f"Error getting frame from RAW track: {e}")
        
        # If no RAW frame available, create a test pattern
        if frame is None:
            frame = self._create_test_pattern()
            raw_data_info = "No RAW data - using test pattern"
        
        # Save debug frame every 30 frames
        if self.save_frames and (self.frame_count % 30 == 0 or self.frame_count < 10):
            debug_frame = frame.copy()
            
            # Add debug info overlay
            cv2.putText(debug_frame, f"Frame {self.frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(debug_frame, f"Size: {frame.shape}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(debug_frame, raw_data_info[:50], (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Save frame
            filename = self.save_dir / f"debug_frame_{self.frame_count:06d}.png"
            cv2.imwrite(str(filename), cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR))
            
            if self.frame_count < 10:
                logger.info(f"🐛 Saved debug frame {self.frame_count}: {filename}")
                logger.info(f"   {raw_data_info}")
        
        self.frame_count += 1
        
        # Convert to VideoFrame
        av_frame = VideoFrame.from_ndarray(frame, format='rgb24')
        av_frame.pts = pts
        av_frame.time_base = time_base
        
        return av_frame
    
    def _create_test_pattern(self):
        """Create a test pattern with ChAruco-like features"""
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 128
        
        # Create checkerboard pattern
        square_size = 40
        for y in range(0, self.height, square_size):
            for x in range(0, self.width, square_size):
                if ((x // square_size) + (y // square_size)) % 2 == 0:
                    frame[y:y+square_size, x:x+square_size] = 255
                else:
                    frame[y:y+square_size, x:x+square_size] = 50
        
        # Add frame counter for debugging
        cv2.putText(frame, f"TEST {self.frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        return frame


class DebugRAW10Client:
    """Debug RAW10 client that saves frames for inspection."""
    
    def __init__(
        self,
        server_url: str,
        camera_num: int = 0,
        resolution: tuple = (2560, 400),
        bit_depth: int = 8,
        framerate: float = 30.0,
        compression: str = "zlib",
        video_width: int = 640,
        video_height: int = 480
    ):
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
        self.video_track: Optional[DebugVideoTrack] = None
        self.data_channel: Optional[RTCDataChannel] = None
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        
        # Connection state
        self.connected = False
        self.running = False
        
    async def connect(self):
        """Establish WebRTC connection with debug features"""
        try:
            # Connect to WebSocket
            session = aiohttp.ClientSession()
            self.ws = await session.ws_connect(self.server_url)
            logger.info(f"🐛 Debug client WebSocket connected to {self.server_url}")
            
            # Create peer connection
            self.pc = RTCPeerConnection()
            
            # Set up connection state handlers
            @self.pc.on("connectionstatechange")
            async def on_connection_state_change():
                logger.info(f"🐛 Connection state: {self.pc.connectionState}")
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
                logger.info(f"🐛 Data channel opened for camera {self.camera_num}")
                # Start RAW10 transmission
                if self.raw_track:
                    await self.raw_track.set_data_channel(self.data_channel)
            
            # Add debug video track that saves frames
            self.video_track = DebugVideoTrack(
                self.raw_track,
                width=self.video_width,
                height=self.video_height,
                save_frames=True
            )
            self.pc.addTrack(self.video_track)
            logger.info(f"🐛 Added debug video track: {self.video_width}x{self.video_height}")
            
            # Create offer
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)
            
            # Send offer via WebSocket
            await self.ws.send_json({
                "type": "offer",
                "sdp": self.pc.localDescription.sdp,
                "offer_type": self.pc.localDescription.type,
                "client_info": {
                    "mode": "debug_raw10",
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
            
            logger.info(f"🐛 Debug mode: RAW10 ({self.resolution[0]}x{self.resolution[1]}) + "
                       f"Video ({self.video_width}x{self.video_height})")
            
        except Exception as e:
            logger.error(f"🐛 Failed to connect: {e}")
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
                        logger.info("🐛 Answer received and set")
                        self.connected = True
                    
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"🐛 WebSocket error: {msg.data}")
                    break
                    
        except Exception as e:
            logger.error(f"🐛 Error handling WebSocket messages: {e}")
    
    async def run(self):
        """Run the debug client"""
        self.running = True
        
        try:
            frame_save_counter = 0
            while self.running and self.connected:
                # Monitor connection and stats
                if self.raw_track:
                    stats = self.raw_track.get_stats()
                    if stats['frames_sent'] % 50 == 0 and stats['frames_sent'] > 0:
                        logger.info(f"🐛 Camera {self.camera_num} - "
                                   f"RAW captured: {stats['frames_captured']}, "
                                   f"sent: {stats['frames_sent']}, "
                                   f"compression: {stats['compression_ratio']:.2f}, "
                                   f"FPS: {stats['send_fps']:.1f}")
                
                frame_save_counter += 1
                
                # Stop after saving enough debug frames
                if frame_save_counter > 300:  # About 5 minutes at 1 Hz logging
                    logger.info("🐛 Debug capture complete, stopping...")
                    break
                
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("🐛 Interrupted by user")
        except Exception as e:
            logger.error(f"🐛 Error in run loop: {e}")
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
        
        logger.info(f"🐛 Disconnected debug camera {self.camera_num}")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Debug RAW10 Frame Viewer")
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
    parser.add_argument("--fps", "-f", type=float, default=30.0,
                       help="Capture framerate")
    parser.add_argument("--compression", choices=["none", "zlib", "lz4"], default="zlib",
                       help="Compression type")
    parser.add_argument("--video-width", type=int, default=640,
                       help="Video stream width")
    parser.add_argument("--video-height", type=int, default=480,
                       help="Video stream height")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("🐛 Starting Debug RAW10 Frame Viewer")
    logger.info(f"📷 Camera: {args.camera}")
    logger.info(f"📊 RAW: {args.width}x{args.height} @ {args.bit_depth}-bit, {args.fps} fps")
    logger.info(f"📺 Video: {args.video_width}x{args.video_height}")
    logger.info(f"🔗 Server: {args.server}")
    logger.info(f"🗜️ Compression: {args.compression}")
    
    client = DebugRAW10Client(
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
        logger.error("🐛 Failed to establish connection")
        return
    
    logger.info("🐛 Debug client connected, capturing frames...")
    await client.run()


if __name__ == "__main__":
    asyncio.run(main())