#!/usr/bin/env python3
"""
Raspberry Pi WebRTC Client for ChAruco Calibration

A Python-based WebRTC client that streams camera feeds from Raspberry Pi
to the ChAruco calibration server. Supports multiple cameras and picamera2.

Features:
- Multiple camera support (USB cameras, picamera2, etc.)
- Configurable camera selection (specific camera or all cameras)
- Real-time video streaming to WebRTC server
- Camera enumeration and detection
- Automatic reconnection on connection loss
- Performance monitoring and frame rate control

Usage:
    # Stream from camera 0
    python raspberry_pi_webrtc_client.py --camera 0
    
    # Stream from all detected cameras
    python raspberry_pi_webrtc_client.py --camera all
    
    # Stream from specific cameras
    python raspberry_pi_webrtc_client.py --camera 0,2,3
    
    # Custom server and resolution
    python raspberry_pi_webrtc_client.py --server ws://192.168.1.100:8081/ws --width 1280 --height 720

Requirements:
    pip install aiortc opencv-python websockets asyncio-mqtt
    
    # For Raspberry Pi Camera:
    pip install picamera2
"""

import asyncio
import argparse
import json
import logging
import signal
import sys
import time
from typing import List, Optional, Dict, Any, Tuple
import cv2
import numpy as np
from dataclasses import dataclass
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRelay
import fractions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import picamera2 (optional, for Raspberry Pi camera)
try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
    logger.info("✅ Picamera2 available - Raspberry Pi camera support enabled")
except ImportError:
    HAS_PICAMERA2 = False
    logger.info("ℹ️  Picamera2 not available - Using OpenCV cameras only")


@dataclass
class CameraInfo:
    """Information about a detected camera."""
    index: int
    name: str
    type: str  # 'opencv', 'picamera2'
    width: int
    height: int
    fps: float
    available: bool = True


class CameraManager:
    """Manages camera discovery and access."""
    
    def __init__(self):
        self.cameras: Dict[int, CameraInfo] = {}
        self.discover_cameras()
    
    def discover_cameras(self):
        """Discover all available cameras."""
        self.cameras.clear()
        camera_index = 0
        
        # Try to find OpenCV cameras
        for i in range(8):  # Check first 8 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get camera properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Try to read a frame to verify camera works
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.cameras[camera_index] = CameraInfo(
                        index=camera_index,
                        name=f"USB Camera {i}",
                        type='opencv',
                        width=width if width > 0 else 640,
                        height=height if height > 0 else 480,
                        fps=fps if fps > 0 else 30.0
                    )
                    camera_index += 1
                    logger.info(f"📹 Found OpenCV camera {i}: {width}x{height} @ {fps}fps")
            cap.release()
        
        # Try to find Raspberry Pi camera
        if HAS_PICAMERA2:
            try:
                picam = Picamera2()
                camera_config = picam.create_video_configuration()
                
                # Get default resolution
                width = camera_config.get('main', {}).get('size', [640, 480])[0]
                height = camera_config.get('main', {}).get('size', [640, 480])[1]
                
                self.cameras[camera_index] = CameraInfo(
                    index=camera_index,
                    name="Raspberry Pi Camera",
                    type='picamera2',
                    width=width,
                    height=height,
                    fps=30.0
                )
                logger.info(f"📷 Found Raspberry Pi camera: {width}x{height}")
                picam.close()
            except Exception as e:
                logger.debug(f"Raspberry Pi camera not available: {e}")
        
        logger.info(f"🔍 Discovered {len(self.cameras)} camera(s)")
    
    def get_camera_list(self) -> List[CameraInfo]:
        """Get list of all available cameras."""
        return list(self.cameras.values())
    
    def get_camera(self, index: int) -> Optional[CameraInfo]:
        """Get camera info by index."""
        return self.cameras.get(index)


class CameraVideoTrack(VideoStreamTrack):
    """Custom video track for camera streaming."""
    
    def __init__(self, camera_info: CameraInfo, width: int = 640, height: int = 480, fps: int = 30):
        super().__init__()
        self.camera_info = camera_info
        self.width = width
        self.height = height
        self.fps = fps
        self.camera = None
        self.frame_count = 0
        self.start_time = time.time()
        
        # Initialize camera
        self._init_camera()
    
    def _init_camera(self):
        """Initialize the camera based on type."""
        try:
            if self.camera_info.type == 'opencv':
                self.camera = cv2.VideoCapture(self.camera_info.index)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.camera.set(cv2.CAP_PROP_FPS, self.fps)
                
                if not self.camera.isOpened():
                    raise RuntimeError(f"Failed to open OpenCV camera {self.camera_info.index}")
                
            elif self.camera_info.type == 'picamera2' and HAS_PICAMERA2:
                self.camera = Picamera2()
                config = self.camera.create_video_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"}
                )
                self.camera.configure(config)
                self.camera.start()
                
            logger.info(f"✅ Initialized {self.camera_info.name} at {self.width}x{self.height}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize camera {self.camera_info.index}: {e}")
            self.camera = None
    
    async def recv(self):
        """Receive the next video frame."""
        if self.camera is None:
            # Return a black frame if camera failed
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(frame, f"Camera {self.camera_info.index} Error", 
                       (50, self.height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            try:
                frame = self._capture_frame()
                self.frame_count += 1
                
                # Log performance periodically
                if self.frame_count % (self.fps * 10) == 0:  # Every 10 seconds
                    elapsed = time.time() - self.start_time
                    actual_fps = self.frame_count / elapsed
                    logger.info(f"📊 Camera {self.camera_info.index}: {actual_fps:.1f} FPS")
                    
            except Exception as e:
                logger.error(f"Error capturing frame from camera {self.camera_info.index}: {e}")
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Add camera ID overlay
        cv2.putText(frame, f"Cam {self.camera_info.index}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Convert to aiortc VideoFrame
        from aiortc import VideoFrame
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = self.frame_count
        video_frame.time_base = fractions.Fraction(1, self.fps)
        
        return video_frame
    
    def _capture_frame(self) -> np.ndarray:
        """Capture a frame from the camera."""
        if self.camera_info.type == 'opencv':
            ret, frame = self.camera.read()
            if not ret or frame is None:
                raise RuntimeError("Failed to capture OpenCV frame")
            
            # Resize if needed
            if frame.shape[:2] != (self.height, self.width):
                frame = cv2.resize(frame, (self.width, self.height))
            
            return frame
            
        elif self.camera_info.type == 'picamera2' and HAS_PICAMERA2:
            frame = self.camera.capture_array()
            
            # Convert RGB to BGR for OpenCV compatibility
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Resize if needed
            if frame.shape[:2] != (self.height, self.width):
                frame = cv2.resize(frame, (self.width, self.height))
            
            return frame
        
        else:
            raise RuntimeError(f"Unsupported camera type: {self.camera_info.type}")
    
    def stop(self):
        """Stop the camera."""
        if self.camera is not None:
            try:
                if self.camera_info.type == 'opencv':
                    self.camera.release()
                elif self.camera_info.type == 'picamera2' and HAS_PICAMERA2:
                    self.camera.stop()
                    self.camera.close()
                logger.info(f"🛑 Stopped camera {self.camera_info.index}")
            except Exception as e:
                logger.error(f"Error stopping camera {self.camera_info.index}: {e}")


class RaspberryPiWebRTCClient:
    """WebRTC client for Raspberry Pi camera streaming."""
    
    def __init__(
        self,
        server_url: str = "ws://localhost:8081/ws",
        width: int = 640,
        height: int = 480,
        fps: int = 30
    ):
        self.server_url = server_url
        self.width = width
        self.height = height
        self.fps = fps
        
        # WebRTC components
        self.pc: Optional[RTCPeerConnection] = None
        self.websocket = None
        
        # Camera management
        self.camera_manager = CameraManager()
        self.active_tracks: List[CameraVideoTrack] = []
        self.selected_cameras: List[int] = []
        
        # Connection state
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(self.stop())
    
    def list_cameras(self):
        """List all available cameras."""
        cameras = self.camera_manager.get_camera_list()
        
        if not cameras:
            logger.warning("No cameras detected!")
            return
        
        logger.info("📹 Available cameras:")
        for cam in cameras:
            logger.info(f"  [{cam.index}] {cam.name} ({cam.type}) - {cam.width}x{cam.height} @ {cam.fps}fps")
    
    def select_cameras(self, camera_spec: str):
        """Select cameras to stream from specification."""
        self.selected_cameras.clear()
        
        if camera_spec.lower() == "all":
            # Select all available cameras
            self.selected_cameras = list(self.camera_manager.cameras.keys())
        else:
            # Parse comma-separated camera indices
            try:
                indices = [int(x.strip()) for x in camera_spec.split(",")]
                for idx in indices:
                    if idx in self.camera_manager.cameras:
                        self.selected_cameras.append(idx)
                    else:
                        logger.warning(f"Camera {idx} not found")
            except ValueError:
                logger.error(f"Invalid camera specification: {camera_spec}")
                return False
        
        if not self.selected_cameras:
            logger.error("No valid cameras selected")
            return False
        
        logger.info(f"🎯 Selected cameras: {self.selected_cameras}")
        return True
    
    async def connect(self):
        """Connect to the WebRTC server."""
        try:
            logger.info(f"🔌 Connecting to {self.server_url}")
            
            # Create peer connection
            self.pc = RTCPeerConnection({
                "iceServers": [{"urls": "stun:stun.l.google.com:19302"}]
            })
            
            # Setup event handlers
            @self.pc.on("connectionstatechange")
            async def on_connectionstatechange():
                logger.info(f"Connection state: {self.pc.connectionState}")
                if self.pc.connectionState == "connected":
                    self.connected = True
                    self.reconnect_attempts = 0
                elif self.pc.connectionState in ["disconnected", "failed", "closed"]:
                    self.connected = False
            
            # Add video tracks for selected cameras
            for camera_idx in self.selected_cameras:
                camera_info = self.camera_manager.get_camera(camera_idx)
                if camera_info:
                    track = CameraVideoTrack(
                        camera_info, 
                        width=self.width, 
                        height=self.height, 
                        fps=self.fps
                    )
                    self.active_tracks.append(track)
                    self.pc.addTrack(track)
                    logger.info(f"📡 Added video track for camera {camera_idx}")
            
            # Connect WebSocket
            self.websocket = await websockets.connect(self.server_url)
            
            # Create and send offer
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)
            
            offer_message = {
                "type": "offer",
                "sdp": offer.sdp
            }
            await self.websocket.send(json.dumps(offer_message))
            logger.info("📤 Sent WebRTC offer")
            
            # Wait for answer
            response = await self.websocket.recv()
            answer_data = json.loads(response)
            
            if answer_data.get("type") == "answer":
                answer = RTCSessionDescription(
                    sdp=answer_data["sdp"],
                    type=answer_data["type"]
                )
                await self.pc.setRemoteDescription(answer)
                logger.info("📥 Received and set WebRTC answer")
                
                return True
            else:
                logger.error(f"Unexpected response: {answer_data}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False
    
    async def run(self):
        """Main client loop."""
        while self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                success = await self.connect()
                if not success:
                    self.reconnect_attempts += 1
                    logger.warning(f"Connection failed, attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}")
                    await asyncio.sleep(5)
                    continue
                
                logger.info("✅ Connected to WebRTC server")
                logger.info("🎬 Streaming video...")
                
                # Keep connection alive and handle messages
                while self.connected:
                    try:
                        if self.websocket:
                            # Check for any incoming messages (like data channel messages)
                            try:
                                message = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
                                # Handle any additional messages from server
                                logger.debug(f"Received message: {message}")
                            except asyncio.TimeoutError:
                                pass  # No message, continue
                        
                        await asyncio.sleep(0.1)
                        
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("WebSocket connection closed")
                        break
                    except Exception as e:
                        logger.error(f"Error in main loop: {e}")
                        break
                
                # Connection lost, try to reconnect
                self.reconnect_attempts += 1
                logger.warning(f"Connection lost, reconnecting... ({self.reconnect_attempts}/{self.max_reconnect_attempts})")
                await self.stop()
                await asyncio.sleep(5)
                
            except KeyboardInterrupt:
                logger.info("Received shutdown signal")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                self.reconnect_attempts += 1
                await asyncio.sleep(5)
        
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached, giving up")
    
    async def stop(self):
        """Stop the client and cleanup resources."""
        logger.info("🛑 Stopping WebRTC client...")
        
        self.connected = False
        
        # Stop all camera tracks
        for track in self.active_tracks:
            try:
                track.stop()
            except Exception as e:
                logger.error(f"Error stopping track: {e}")
        self.active_tracks.clear()
        
        # Close peer connection
        if self.pc:
            try:
                await self.pc.close()
            except Exception as e:
                logger.error(f"Error closing peer connection: {e}")
            self.pc = None
        
        # Close WebSocket
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.error(f"Error closing websocket: {e}")
            self.websocket = None
        
        logger.info("✅ Client stopped")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Raspberry Pi WebRTC Client for ChAruco Calibration")
    
    parser.add_argument(
        "--server", 
        default="ws://localhost:8081/ws",
        help="WebRTC server WebSocket URL"
    )
    
    parser.add_argument(
        "--camera",
        default="0",
        help="Camera specification: camera index, 'all', or comma-separated indices (e.g., '0,1,2')"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Video width (default: 640)"
    )
    
    parser.add_argument(
        "--height", 
        type=int,
        default=480,
        help="Video height (default: 480)"
    )
    
    parser.add_argument(
        "--fps",
        type=int, 
        default=30,
        help="Video FPS (default: 30)"
    )
    
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available cameras and exit"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true", 
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create client
    client = RaspberryPiWebRTCClient(
        server_url=args.server,
        width=args.width,
        height=args.height,
        fps=args.fps
    )
    
    # List cameras if requested
    if args.list_cameras:
        client.list_cameras()
        return
    
    # Select cameras
    if not client.select_cameras(args.camera):
        logger.error("Failed to select cameras")
        return
    
    # Show configuration
    logger.info("=== Raspberry Pi WebRTC Client ===")
    logger.info(f"Server: {args.server}")
    logger.info(f"Resolution: {args.width}x{args.height} @ {args.fps}fps")
    logger.info(f"Selected cameras: {client.selected_cameras}")
    logger.info("")
    
    # List available cameras
    client.list_cameras()
    logger.info("")
    logger.info("🚀 Starting WebRTC client...")
    logger.info("Press Ctrl+C to stop")
    
    # Run client
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        logger.info("Client stopped by user")
    except Exception as e:
        logger.error(f"Client failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()