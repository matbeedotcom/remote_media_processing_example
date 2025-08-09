#!/usr/bin/env python3
"""
RAW10 WebRTC Client with WebSocket Signaling

Modified version of RAW10 client that uses WebSocket signaling
to work with the RemoteMedia WebRTC server infrastructure.
"""

import asyncio
import json
import logging
import argparse
import sys
from typing import Optional, Dict, Any
from pathlib import Path

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel, VideoStreamTrack
from aiortc.contrib.media import MediaPlayer
import aiohttp
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from raw10_datachannel_track import RAW10DataChannelTrack, create_raw10_datachannel_track

logger = logging.getLogger(__name__)


class DummyVideoTrack(VideoStreamTrack):
    """
    Dummy video track to satisfy WebRTC requirements.
    The actual data goes through data channels.
    """
    
    def __init__(self, width=640, height=480, fps=1):
        super().__init__()
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = 0
        
    async def recv(self):
        """Generate a minimal dummy frame"""
        pts, time_base = await self.next_timestamp()
        
        # Create a minimal black frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Add frame counter in corner for debugging
        self.frame_count += 1
        if self.frame_count % self.fps == 0:
            # Add a small white square every second
            frame[10:20, 10:20] = 255
        
        # Convert to VideoFrame
        from av import VideoFrame
        av_frame = VideoFrame.from_ndarray(frame, format='rgb24')
        av_frame.pts = pts
        av_frame.time_base = time_base
        
        return av_frame


class RAW10WebSocketClient:
    """
    WebRTC client with RAW10 data channel support using WebSocket signaling.
    """
    
    def __init__(
        self,
        server_url: str,
        camera_num: int = 0,
        resolution: tuple = (3280, 2464),
        bit_depth: int = 10,
        framerate: float = 15.0,
        compression: str = "zlib"
    ):
        """
        Initialize RAW10 WebSocket client.
        
        Args:
            server_url: WebSocket server URL (ws://...)
            camera_num: Camera index
            resolution: RAW capture resolution
            bit_depth: Bit depth (8 or 10)
            framerate: RAW capture framerate
            compression: Compression for data channel
        """
        self.server_url = server_url
        self.camera_num = camera_num
        self.resolution = resolution
        self.bit_depth = bit_depth
        self.framerate = framerate
        self.compression = compression
        
        # WebRTC components
        self.pc: Optional[RTCPeerConnection] = None
        self.raw_track: Optional[RAW10DataChannelTrack] = None
        self.data_channel: Optional[RTCDataChannel] = None
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        
        # Connection state
        self.connected = False
        self.running = False
        
    async def connect(self):
        """Establish WebRTC connection with server via WebSocket"""
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
                
                # Start RAW10 capture and transmission
                if self.raw_track:
                    await self.raw_track.set_data_channel(self.data_channel)
            
            @self.data_channel.on("close")
            async def on_datachannel_close():
                logger.info(f"Data channel closed for camera {self.camera_num}")
            
            # Initialize RAW10 track
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
            
            # Add dummy video track (required for WebRTC but not used for data)
            dummy_track = DummyVideoTrack(width=320, height=240, fps=1)
            self.pc.addTrack(dummy_track)
            logger.info("Added dummy video track for WebRTC compatibility")
            
            # Create offer
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)
            
            # Send offer via WebSocket
            await self.ws.send_json({
                "type": "offer",
                "offer": {
                    "sdp": self.pc.localDescription.sdp,
                    "type": self.pc.localDescription.type
                },
                "client_info": {
                    "camera_id": self.camera_num,
                    "raw_resolution": list(self.resolution),
                    "bit_depth": self.bit_depth,
                    "framerate": self.framerate,
                    "compression": self.compression,
                    "mode": "raw10_datachannel"
                }
            })
            
            # Handle WebSocket messages
            asyncio.create_task(self._handle_ws_messages())
            
            logger.info(f"Offer sent for camera {self.camera_num}")
            
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
                            sdp=data["answer"]["sdp"],
                            type=data["answer"]["type"]
                        )
                        await self.pc.setRemoteDescription(answer)
                        logger.info("Answer received and set")
                        self.connected = True
                    
                    elif data.get("type") == "ice_candidate":
                        # Add ICE candidate if needed
                        pass
                    
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
                        logger.info(f"Camera {self.camera_num} stats: "
                                   f"Captured: {stats['frames_captured']}, "
                                   f"Sent: {stats['frames_sent']}, "
                                   f"Compression: {stats['compression_ratio']:.2f}, "
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
    parser = argparse.ArgumentParser(description="RAW10 WebRTC Client with WebSocket Signaling")
    parser.add_argument("--server", "-s", default="ws://localhost:8085/ws",
                       help="WebSocket server URL")
    parser.add_argument("--camera", "-c", type=int, default=0,
                       help="Camera index")
    parser.add_argument("--width", "-w", type=int, default=3280,
                       help="RAW capture width")
    parser.add_argument("--height", "-H", type=int, default=2464,
                       help="RAW capture height")
    parser.add_argument("--bit-depth", "-b", type=int, choices=[8, 10], default=10,
                       help="Bit depth")
    parser.add_argument("--fps", "-f", type=float, default=15.0,
                       help="Capture framerate")
    parser.add_argument("--compression", choices=["none", "zlib", "lz4"], default="zlib",
                       help="Compression type")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create and run client
    logger.info(f"Starting RAW10 WebSocket client for camera {args.camera}")
    logger.info(f"Server: {args.server}")
    logger.info(f"Resolution: {args.width}x{args.height} @ {args.bit_depth}-bit")
    logger.info(f"Framerate: {args.fps} fps")
    logger.info(f"Compression: {args.compression}")
    
    client = RAW10WebSocketClient(
        server_url=args.server,
        camera_num=args.camera,
        resolution=(args.width, args.height),
        bit_depth=args.bit_depth,
        framerate=args.fps,
        compression=args.compression
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