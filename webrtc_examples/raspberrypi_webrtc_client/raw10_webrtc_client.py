#!/usr/bin/env python3
"""
Enhanced Raspberry Pi WebRTC Client with RAW10 DataChannel Support

Transfers uncompressed RAW10 sensor data via WebRTC data channels for
maximum quality in VLBI and astrophotography applications.

Features:
- RAW10/RAW8 sensor data capture
- Data channel transfer for uncompressed data
- Optional video preview stream
- Multi-camera support for quad arrays
- Frame synchronization
- Compression options (none, zlib, lz4)
"""

import asyncio
import json
import logging
import argparse
import sys
from typing import Optional, Dict, Any
from pathlib import Path

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel
from aiortc.contrib.signaling import TcpSocketSignaling, UnixSocketSignaling
import aiohttp

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from raw10_datachannel_track import RAW10DataChannelTrack, create_raw10_datachannel_track
from video_track import CameraVideoTrack

logger = logging.getLogger(__name__)


class RAW10WebRTCClient:
    """
    WebRTC client with RAW10 data channel support.
    
    Provides both video preview and RAW data transfer capabilities.
    """
    
    def __init__(
        self,
        server_url: str,
        camera_num: int = 0,
        resolution: tuple = (3280, 2464),
        bit_depth: int = 10,
        framerate: float = 15.0,
        compression: str = "zlib",
        enable_video_preview: bool = False,
        video_resolution: tuple = (640, 480),
        video_framerate: float = 30.0
    ):
        """
        Initialize RAW10 WebRTC client.
        
        Args:
            server_url: WebRTC server URL
            camera_num: Camera index
            resolution: RAW capture resolution
            bit_depth: Bit depth (8 or 10)
            framerate: RAW capture framerate
            compression: Compression for data channel
            enable_video_preview: Enable video stream preview
            video_resolution: Resolution for video preview
            video_framerate: Framerate for video preview
        """
        self.server_url = server_url
        self.camera_num = camera_num
        self.resolution = resolution
        self.bit_depth = bit_depth
        self.framerate = framerate
        self.compression = compression
        self.enable_video_preview = enable_video_preview
        self.video_resolution = video_resolution
        self.video_framerate = video_framerate
        
        # WebRTC components
        self.pc: Optional[RTCPeerConnection] = None
        self.raw_track: Optional[RAW10DataChannelTrack] = None
        self.video_track: Optional[CameraVideoTrack] = None
        self.data_channel: Optional[RTCDataChannel] = None
        
        # Connection state
        self.connected = False
        self.running = False
        
    async def connect(self):
        """Establish WebRTC connection with server"""
        try:
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
            
            # Create and configure data channel for RAW10
            self.data_channel = self.pc.createDataChannel(
                "raw10",
                ordered=True,
                maxRetransmits=3,
                maxPacketLifeTime=None,
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
            
            @self.data_channel.on("error")
            async def on_datachannel_error(error):
                logger.error(f"Data channel error: {error}")
            
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
            
            # Add video preview track if enabled
            if self.enable_video_preview:
                self.video_track = CameraVideoTrack(
                    camera_num=self.camera_num,
                    width=self.video_resolution[0],
                    height=self.video_resolution[1],
                    framerate=self.video_framerate
                )
                self.pc.addTrack(self.video_track)
                logger.info(f"Added video preview track: {self.video_resolution} @ {self.video_framerate}fps")
            
            # Create offer
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)
            
            # Send offer to server
            async with aiohttp.ClientSession() as session:
                # Prepare offer data with client capabilities
                offer_data = {
                    "sdp": self.pc.localDescription.sdp,
                    "type": self.pc.localDescription.type,
                    "client_info": {
                        "camera_id": self.camera_num,
                        "raw_resolution": list(self.resolution),
                        "bit_depth": self.bit_depth,
                        "framerate": self.framerate,
                        "compression": self.compression,
                        "has_video_preview": self.enable_video_preview,
                        "video_resolution": list(self.video_resolution) if self.enable_video_preview else None,
                        "capabilities": {
                            "raw10": True,
                            "raw8": True,
                            "data_channel": True,
                            "compression": ["none", "zlib", "lz4"] if self._has_lz4() else ["none", "zlib"]
                        }
                    }
                }
                
                # Send offer and get answer
                async with session.post(
                    f"{self.server_url}/offer",
                    json=offer_data,
                    headers={"Content-Type": "application/json"}
                ) as resp:
                    if resp.status != 200:
                        raise Exception(f"Failed to send offer: {resp.status}")
                    
                    answer_data = await resp.json()
            
            # Set remote description
            answer = RTCSessionDescription(
                sdp=answer_data["sdp"],
                type=answer_data["type"]
            )
            await self.pc.setRemoteDescription(answer)
            
            logger.info(f"WebRTC connection established for camera {self.camera_num}")
            self.connected = True
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            await self.disconnect()
            raise
    
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
        
        # Stop video track
        if self.video_track:
            self.video_track.stop()
            self.video_track = None
        
        # Close peer connection
        if self.pc:
            await self.pc.close()
            self.pc = None
        
        logger.info(f"Disconnected camera {self.camera_num}")
    
    def _has_lz4(self) -> bool:
        """Check if LZ4 is available"""
        try:
            import lz4.frame
            return True
        except ImportError:
            return False


class QuadRAW10WebRTCClient:
    """
    Multi-camera WebRTC client for quad array with RAW10 support.
    
    Manages 4 synchronized cameras with data channel transfer.
    """
    
    def __init__(
        self,
        server_url: str,
        camera_indices: list = [0, 1, 2, 3],
        resolution: tuple = (3280, 2464),
        bit_depth: int = 10,
        framerate: float = 15.0,
        compression: str = "zlib",
        enable_video_preview: bool = False
    ):
        """
        Initialize quad camera client.
        
        Args:
            server_url: WebRTC server URL
            camera_indices: List of camera indices
            resolution: RAW capture resolution
            bit_depth: Bit depth
            framerate: Capture framerate
            compression: Compression type
            enable_video_preview: Enable video preview
        """
        self.server_url = server_url
        self.camera_indices = camera_indices
        
        # Create client for each camera
        self.clients = []
        for cam_idx in camera_indices:
            client = RAW10WebRTCClient(
                server_url=server_url,
                camera_num=cam_idx,
                resolution=resolution,
                bit_depth=bit_depth,
                framerate=framerate,
                compression=compression,
                enable_video_preview=enable_video_preview
            )
            self.clients.append(client)
    
    async def connect_all(self):
        """Connect all cameras"""
        logger.info(f"Connecting {len(self.clients)} cameras...")
        
        # Connect cameras in parallel
        connect_tasks = [client.connect() for client in self.clients]
        results = await asyncio.gather(*connect_tasks, return_exceptions=True)
        
        # Check results
        connected_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Camera {self.camera_indices[i]} failed to connect: {result}")
            else:
                connected_count += 1
        
        logger.info(f"Connected {connected_count}/{len(self.clients)} cameras")
        return connected_count
    
    async def run(self):
        """Run all clients"""
        try:
            # Connect all cameras
            connected = await self.connect_all()
            
            if connected == 0:
                logger.error("No cameras connected")
                return
            
            # Run all clients
            run_tasks = [client.run() for client in self.clients if client.connected]
            await asyncio.gather(*run_tasks)
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in quad client: {e}")
        finally:
            await self.disconnect_all()
    
    async def disconnect_all(self):
        """Disconnect all cameras"""
        logger.info("Disconnecting all cameras...")
        disconnect_tasks = [client.disconnect() for client in self.clients]
        await asyncio.gather(*disconnect_tasks, return_exceptions=True)
        logger.info("All cameras disconnected")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="RAW10 WebRTC Client for Raspberry Pi")
    parser.add_argument("--server", "-s", default="http://localhost:8082",
                       help="WebRTC server URL")
    parser.add_argument("--camera", "-c", type=int, default=0,
                       help="Camera index (ignored if --quad)")
    parser.add_argument("--quad", action="store_true",
                       help="Use quad camera array")
    parser.add_argument("--cameras", type=int, nargs="+", default=[0, 1, 2, 3],
                       help="Camera indices for quad mode")
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
    parser.add_argument("--video-preview", action="store_true",
                       help="Enable video preview stream")
    parser.add_argument("--video-width", type=int, default=640,
                       help="Video preview width")
    parser.add_argument("--video-height", type=int, default=480,
                       help="Video preview height")
    parser.add_argument("--video-fps", type=float, default=30.0,
                       help="Video preview framerate")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create and run client
    if args.quad:
        logger.info(f"Starting quad camera client with cameras {args.cameras}")
        client = QuadRAW10WebRTCClient(
            server_url=args.server,
            camera_indices=args.cameras,
            resolution=(args.width, args.height),
            bit_depth=args.bit_depth,
            framerate=args.fps,
            compression=args.compression,
            enable_video_preview=args.video_preview
        )
        await client.run()
    else:
        logger.info(f"Starting single camera client for camera {args.camera}")
        client = RAW10WebRTCClient(
            server_url=args.server,
            camera_num=args.camera,
            resolution=(args.width, args.height),
            bit_depth=args.bit_depth,
            framerate=args.fps,
            compression=args.compression,
            enable_video_preview=args.video_preview,
            video_resolution=(args.video_width, args.video_height),
            video_framerate=args.video_fps
        )
        await client.connect()
        await client.run()


if __name__ == "__main__":
    asyncio.run(main())