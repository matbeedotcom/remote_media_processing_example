"""
Raspberry Pi WebRTC Client

Main client class for streaming camera feeds to the ChAruco calibration server.
Handles WebRTC connection, camera management, and automatic reconnection.
"""

import asyncio
import json
import logging
import signal
import time
from typing import List, Optional, Dict, Any
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer

from camera_manager import CameraManager, close_picamera2_instance
from video_track import CameraVideoTrack

logger = logging.getLogger(__name__)


class RaspberryPiWebRTCClient:
    """WebRTC client for Raspberry Pi camera streaming."""
    
    def __init__(
        self,
        server_url: str = "ws://localhost:8081/ws",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        auto_reconnect: bool = True,
        max_reconnect_attempts: int = 10,
        reconnect_delay: float = 5.0
    ):
        self.server_url = server_url
        self.width = width
        self.height = height
        self.fps = fps
        self.auto_reconnect = auto_reconnect
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        
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
        self.shutdown_requested = False
        self.connection_start_time = None
        
        # Statistics
        self.stats = {
            'total_connections': 0,
            'total_disconnections': 0,
            'total_frames_sent': 0,
            'connection_uptime': 0
        }
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.shutdown_requested = True
            # Create shutdown task if we're in an event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.stop())
            except RuntimeError:
                pass  # No event loop running
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def list_cameras(self):
        """List all available cameras."""
        cameras = self.camera_manager.get_camera_list()
        
        if not cameras:
            logger.warning("No cameras detected!")
            print("No cameras found. Please check your camera connections.")
            return
        
        print("📹 Available cameras:")
        for cam in cameras:
            capabilities = self.camera_manager.get_camera_capabilities(cam.index)
            print(f"  [{cam.index}] {cam.name} ({cam.type})")
            print(f"      Resolution: {cam.width}x{cam.height} @ {cam.fps}fps")
            if cam.device_path:
                print(f"      Device: {cam.device_path}")
            
            # Show additional capabilities
            if 'supported_resolutions' in capabilities:
                resolutions = capabilities['supported_resolutions'][:3]  # Show first 3
                res_str = ', '.join([f"{w}x{h}" for w, h in resolutions])
                print(f"      Supported: {res_str}{'...' if len(capabilities['supported_resolutions']) > 3 else ''}")
    
    def select_cameras(self, camera_spec: str) -> bool:
        """Select cameras to stream from specification."""
        # Only clear if we're not already connected (prevent reconnection issues)
        if not self.connected:
            self.selected_cameras.clear()
        else:
            logger.debug(f"⚠️  Not clearing selected cameras - already connected")
        
        new_cameras = []
        if camera_spec.lower() == "all":
            # Select all available cameras
            new_cameras = list(self.camera_manager.cameras.keys())
        else:
            # Parse comma-separated camera indices
            try:
                indices = [int(x.strip()) for x in camera_spec.split(",")]
                for idx in indices:
                    if idx in self.camera_manager.cameras:
                        new_cameras.append(idx)
                    else:
                        logger.warning(f"Camera {idx} not found")
            except ValueError:
                logger.error(f"Invalid camera specification: {camera_spec}")
                return False
        
        # Only add cameras that aren't already selected
        for cam_idx in new_cameras:
            if cam_idx not in self.selected_cameras:
                self.selected_cameras.append(cam_idx)
        
        if not self.selected_cameras:
            logger.error("No valid cameras selected")
            return False
        
        # Remove duplicates while preserving order
        self.selected_cameras = list(dict.fromkeys(self.selected_cameras))
        
        # Validate selected cameras
        valid_cameras = []
        for cam_idx in self.selected_cameras:
            if self.camera_manager.validate_camera(cam_idx):
                valid_cameras.append(cam_idx)
            else:
                logger.warning(f"Camera {cam_idx} validation failed")
        
        self.selected_cameras = valid_cameras
        
        if not self.selected_cameras:
            logger.error("No cameras passed validation")
            return False
        
        logger.info(f"🎯 Selected cameras: {self.selected_cameras}")
        
        # Log if we have more than one camera selected
        if len(self.selected_cameras) > 1:
            logger.warning(f"⚠️  Multiple cameras selected: {self.selected_cameras}. This may cause resource conflicts.")
        
        return True
    
    async def _cleanup_connection(self):
        """Clean up existing connection and tracks."""
        # Stop and clean up existing video tracks
        if self.active_tracks:
            logger.debug(f"🧹 Cleaning up {len(self.active_tracks)} existing video tracks")
            for track in self.active_tracks:
                try:
                    track.stop()
                except Exception as e:
                    logger.debug(f"Error stopping track: {e}")
            self.active_tracks.clear()
        
        # Close existing peer connection
        if self.pc:
            try:
                await self.pc.close()
            except Exception as e:
                logger.debug(f"Error closing peer connection: {e}")
            self.pc = None
        
        # Close existing websocket
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.debug(f"Error closing websocket: {e}")
            self.websocket = None
        
        self.connected = False
    
    async def connect(self) -> bool:
        """Connect to the WebRTC server."""
        try:
            logger.info(f"🔌 Connecting to {self.server_url}")
            
            # Clean up any existing connections and tracks first
            await self._cleanup_connection()
            
            # Create peer connection with enhanced ICE configuration
            ice_servers = [
                RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
                RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
                RTCIceServer(urls=["stun:stun.cloudflare.com:3478"]),  # Additional STUN server
            ]
            
            # Create RTCConfiguration with compatibility for different aiortc versions
            try:
                config = RTCConfiguration(
                    iceServers=ice_servers,
                    iceTransportPolicy="all",  # Use both STUN and TURN if available
                    bundlePolicy="balanced",   # Better for single video track
                    rtcpMuxPolicy="require"     # Standard for modern WebRTC
                )
            except TypeError:
                # Fallback for older aiortc versions that don't support all parameters
                logger.debug("🔄 Using fallback RTCConfiguration for older aiortc version")
                config = RTCConfiguration(iceServers=ice_servers)
            
            logger.debug(f"🌐 Using {len(ice_servers)} STUN servers for ICE negotiation")
            self.pc = RTCPeerConnection(config)
            
            # Setup event handlers with enhanced debugging
            @self.pc.on("connectionstatechange")
            async def on_connectionstatechange():
                state = self.pc.connectionState
                ice_state = self.pc.iceConnectionState
                logger.info(f"🔗 Connection state: {state} | ICE state: {ice_state}")
                
                if state == "connected":
                    self.connected = True
                    self.reconnect_attempts = 0
                    self.connection_start_time = time.time()
                    self.stats['total_connections'] += 1
                    logger.info(f"🎉 WebRTC connection established successfully!")
                elif state in ["disconnected", "failed", "closed"]:
                    logger.warning(f"⚠️  Connection state changed to {state} (ICE: {ice_state})")
                    if self.connected:
                        self.stats['total_disconnections'] += 1
                        if self.connection_start_time:
                            self.stats['connection_uptime'] += time.time() - self.connection_start_time
                    self.connected = False
            
            @self.pc.on("icegatheringstatechange")
            async def on_icegatheringstatechange():
                logger.info(f"🧊 ICE gathering state: {self.pc.iceGatheringState}")
            
            @self.pc.on("iceconnectionstatechange")
            async def on_iceconnectionstatechange():
                ice_state = self.pc.iceConnectionState
                logger.info(f"🌍 ICE connection state: {ice_state}")
                
                if ice_state == "failed":
                    logger.error(f"❌ ICE connection failed - this usually indicates network connectivity issues")
                elif ice_state == "disconnected":
                    logger.warning(f"⚠️  ICE connection disconnected")
                elif ice_state == "connected":
                    logger.info(f"✅ ICE connection established")
                elif ice_state == "completed":
                    logger.info(f"✨ ICE connection completed (optimal path found)")
            
            # Add track event debugging
            @self.pc.on("track")
            def on_track(track):
                logger.info(f"📡 Received remote track: {track.kind} (ID: {track.id})")
                
                @track.on("ended")
                def on_ended():
                    logger.info(f"🔚 Remote track ended: {track.kind} (ID: {track.id})")
            
            # Add video tracks for selected cameras
            for camera_idx in self.selected_cameras:
                camera_info = self.camera_manager.get_camera(camera_idx)
                if camera_info:
                    track = CameraVideoTrack(
                        camera_info, 
                        width=self.width, 
                        height=self.height, 
                        fps=self.fps,
                        camera_id=camera_idx
                    )
                    self.active_tracks.append(track)
                    self.pc.addTrack(track)
                    logger.info(f"📡 Added video track for camera {camera_idx} ({camera_info.name})")
            
            if not self.active_tracks:
                logger.error("No camera tracks created")
                return False
            
            # Connect WebSocket
            try:
                self.websocket = await asyncio.wait_for(
                    websockets.connect(self.server_url),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.error("WebSocket connection timeout")
                return False
            
            # Create and send offer
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)
            
            offer_message = {
                "type": "offer",
                "sdp": offer.sdp,
                "cameras": [
                    {
                        "id": track.camera_id,
                        "name": track.camera_info.name,
                        "type": track.camera_info.type,
                        "resolution": [track.width, track.height],
                        "fps": track.fps
                    }
                    for track in self.active_tracks
                ]
            }
            
            await self.websocket.send(json.dumps(offer_message))
            logger.info("📤 Sent WebRTC offer")
            
            # Wait for answer with better error handling
            try:
                logger.debug("⏳ Waiting for WebRTC answer from server...")
                response = await asyncio.wait_for(self.websocket.recv(), timeout=15.0)  # Increased timeout
                answer_data = json.loads(response)
                logger.debug(f"📨 Received response: {answer_data.get('type', 'unknown')}")
            except asyncio.TimeoutError:
                logger.error("⏰ Timeout waiting for WebRTC answer (15s) - server may be unresponsive")
                return False
            except json.JSONDecodeError as e:
                logger.error(f"📜 Invalid JSON response from server: {e}")
                return False
            
            if answer_data.get("type") == "answer":
                answer = RTCSessionDescription(
                    sdp=answer_data["sdp"],
                    type=answer_data["type"]
                )
                await self.pc.setRemoteDescription(answer)
                logger.info("📥 Received and set WebRTC answer")
                
                # Wait a bit for ICE to establish before considering connection successful
                logger.debug("⏳ Waiting for ICE connection to establish...")
                await asyncio.sleep(2.0)  # Give ICE time to connect
                
                # Check if connection is still valid
                if self.pc.connectionState in ["closed", "failed"]:
                    logger.error(f"❌ Connection failed during ICE negotiation: {self.pc.connectionState}")
                    return False
                
                return True
            elif answer_data.get("type") == "error":
                error_msg = answer_data.get("message", "Unknown error")
                logger.error(f"❌ Server returned error: {error_msg}")
                return False
            else:
                logger.error(f"❓ Unexpected response type: {answer_data}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False
    
    async def run(self):
        """Main client loop with automatic reconnection."""
        logger.info(f"🚀 Starting WebRTC client with {len(self.selected_cameras)} camera(s)")
        
        while not self.shutdown_requested and self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                # Attempt connection
                success = await self.connect()
                if not success:
                    if self.auto_reconnect:
                        self.reconnect_attempts += 1
                        logger.warning(f"Connection failed, attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}")
                        await asyncio.sleep(self.reconnect_delay)
                        continue
                    else:
                        break
                
                logger.info("✅ Connected to WebRTC server")
                logger.info("🎬 Streaming video from cameras...")
                
                # Monitor connection and handle messages
                await self._connection_loop()
                
                # Connection lost
                if not self.shutdown_requested and self.auto_reconnect:
                    self.reconnect_attempts += 1
                    logger.warning(f"Connection lost, reconnecting... ({self.reconnect_attempts}/{self.max_reconnect_attempts})")
                    await self._cleanup_connection()
                    await asyncio.sleep(self.reconnect_delay)
                else:
                    break
                
            except KeyboardInterrupt:
                logger.info("Received shutdown signal")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                if self.auto_reconnect:
                    self.reconnect_attempts += 1
                    await asyncio.sleep(self.reconnect_delay)
                else:
                    break
        
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached, giving up")
        
        await self.stop()
    
    async def _connection_loop(self):
        """Main connection monitoring loop with enhanced stability monitoring."""
        last_stats_time = time.time()
        stats_interval = 30.0  # Report stats every 30 seconds
        
        # Wait for stable connection
        logger.debug("⏳ Waiting for connection to stabilize...")
        stable_checks = 0
        max_stable_checks = 10  # Check for 1 second (10 * 0.1s)
        
        while stable_checks < max_stable_checks and not self.shutdown_requested:
            if self.pc.connectionState == "connected" and self.pc.iceConnectionState in ["connected", "completed"]:
                stable_checks += 1
            else:
                stable_checks = 0  # Reset if connection becomes unstable
                if self.pc.connectionState in ["closed", "failed"]:
                    logger.error(f"❌ Connection failed during stabilization: {self.pc.connectionState}")
                    return
            await asyncio.sleep(0.1)
        
        if stable_checks >= max_stable_checks:
            logger.info("✨ Connection stabilized - entering monitoring loop")
        else:
            logger.warning("⚠️  Connection did not stabilize - continuing anyway")
        
        while self.connected and not self.shutdown_requested:
            try:
                # Check connection health
                if self.pc.connectionState in ["closed", "failed"]:
                    logger.warning(f"⚠️  Connection state changed to {self.pc.connectionState} - breaking loop")
                    break
                
                # Handle WebSocket messages
                if self.websocket:
                    try:
                        message = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
                        await self._handle_message(message)
                    except asyncio.TimeoutError:
                        pass  # No message, continue
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("🔌 WebSocket connection closed")
                        break
                
                # Report statistics periodically
                current_time = time.time()
                if current_time - last_stats_time >= stats_interval:
                    await self._report_stats()
                    last_stats_time = current_time
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in connection loop: {e}")
                break
    
    async def _handle_message(self, message: str):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "ping":
                # Respond to ping
                pong = {"type": "pong", "timestamp": time.time()}
                await self.websocket.send(json.dumps(pong))
            
            elif message_type == "stats_request":
                # Send statistics
                stats = await self._get_detailed_stats()
                response = {"type": "stats_response", "stats": stats}
                await self.websocket.send(json.dumps(response))
            
            else:
                logger.debug(f"Received message: {data}")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _report_stats(self):
        """Report client statistics."""
        stats = await self._get_detailed_stats()
        logger.info("📊 Client Statistics:")
        logger.info(f"   • Connections: {stats['total_connections']}")
        logger.info(f"   • Active cameras: {len(self.active_tracks)}")
        logger.info(f"   • Connection uptime: {stats['current_uptime']:.1f}s")
        
        for track_stats in stats['camera_stats']:
            logger.info(f"   • Camera {track_stats['camera_id']}: {track_stats['fps']:.1f} FPS, {track_stats['frame_count']} frames")
    
    async def _get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed client statistics."""
        current_uptime = 0
        if self.connected and self.connection_start_time:
            current_uptime = time.time() - self.connection_start_time
        
        camera_stats = [track.get_stats() for track in self.active_tracks]
        
        return {
            'connected': self.connected,
            'total_connections': self.stats['total_connections'],
            'total_disconnections': self.stats['total_disconnections'],
            'reconnect_attempts': self.reconnect_attempts,
            'current_uptime': current_uptime,
            'total_uptime': self.stats['connection_uptime'] + current_uptime,
            'server_url': self.server_url,
            'selected_cameras': self.selected_cameras,
            'camera_stats': camera_stats,
            'stream_config': {
                'width': self.width,
                'height': self.height,
                'fps': self.fps
            }
        }
    
    async def _cleanup_connection(self):
        """Cleanup current connection without stopping cameras."""
        logger.debug("Cleaning up connection...")
        
        self.connected = False
        
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
    
    async def stop(self):
        """Stop the client and cleanup all resources."""
        logger.info("🛑 Stopping WebRTC client...")
        
        self.shutdown_requested = True
        self.connected = False
        
        # Stop all camera tracks
        for track in self.active_tracks:
            try:
                track.stop()
            except Exception as e:
                logger.error(f"Error stopping track: {e}")
        self.active_tracks.clear()
        
        # Cleanup connection
        await self._cleanup_connection()
        
        # Close Picamera2 singleton if it was used
        try:
            close_picamera2_instance()
        except Exception as e:
            logger.debug(f"Error closing Picamera2 singleton: {e}")
        
        # Final statistics
        final_stats = await self._get_detailed_stats()
        logger.info("📊 Final Statistics:")
        logger.info(f"   • Total connections: {final_stats['total_connections']}")
        logger.info(f"   • Total uptime: {final_stats['total_uptime']:.1f}s")
        logger.info(f"   • Cameras used: {len(self.selected_cameras)}")
        
        logger.info("✅ Client stopped")
    
    def refresh_cameras(self):
        """Refresh the camera list."""
        logger.info("🔄 Refreshing camera list...")
        self.camera_manager.refresh_cameras()
    
    async def test_cameras(self) -> Dict[int, bool]:
        """Test all selected cameras and return results."""
        logger.info("🧪 Testing cameras...")
        results = {}
        
        for camera_idx in self.selected_cameras:
            try:
                camera_info = self.camera_manager.get_camera(camera_idx)
                if not camera_info:
                    results[camera_idx] = False
                    continue
                
                # Create a temporary track to test the camera
                test_track = CameraVideoTrack(
                    camera_info, 
                    width=320, 
                    height=240, 
                    fps=15,
                    camera_id=camera_idx
                )
                
                # Try to get a frame
                try:
                    frame = await test_track.recv()
                    results[camera_idx] = frame is not None
                    logger.info(f"✅ Camera {camera_idx} test passed")
                except Exception as e:
                    logger.error(f"❌ Camera {camera_idx} test failed: {e}")
                    results[camera_idx] = False
                finally:
                    test_track.stop()
                    
            except Exception as e:
                logger.error(f"Error testing camera {camera_idx}: {e}")
                results[camera_idx] = False
        
        return results