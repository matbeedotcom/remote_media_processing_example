"""
Custom Video Track for Camera Streaming

Implements aiortc VideoStreamTrack for streaming camera feeds to WebRTC server.
Supports both OpenCV and Picamera2 cameras with performance monitoring.
"""

import cv2
import numpy as np
import time
import logging
import fractions
from typing import Optional
from aiortc import VideoStreamTrack
from av import VideoFrame

try:
    from .camera_manager import CameraInfo, HAS_PICAMERA2
except ImportError:
    # Fallback for direct execution
    from camera_manager import CameraInfo, HAS_PICAMERA2

logger = logging.getLogger(__name__)

if HAS_PICAMERA2:
    from picamera2 import Picamera2

# Global camera instance management for picamera2
_picamera_instances = {}
_picamera_lock = {}


class CameraVideoTrack(VideoStreamTrack):
    """Custom video track for camera streaming."""
    
    def __init__(
        self, 
        camera_info: CameraInfo, 
        width: int = 640, 
        height: int = 480, 
        fps: int = 30,
        camera_id: Optional[int] = None
    ):
        super().__init__()
        self.camera_info = camera_info
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_id = camera_id or camera_info.index
        
        # Camera instance
        self.camera = None
        self.camera_lock = False
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_report = time.time()
        self.fps_report_interval = 10.0  # Report FPS every 10 seconds
        
        # Error handling
        self.error_count = 0
        self.max_errors = 10
        self.last_error_time = 0
        
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
                
                # Set buffer size to reduce latency
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                if not self.camera.isOpened():
                    raise RuntimeError(f"Failed to open OpenCV camera {self.camera_info.index}")
                
                # Verify actual settings
                actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
                
                logger.info(f"📹 OpenCV camera {self.camera_info.index}: {actual_width}x{actual_height} @ {actual_fps}fps")
                
            elif self.camera_info.type == 'picamera2' and HAS_PICAMERA2:
                # Use singleton pattern for picamera2 to prevent resource conflicts
                camera_key = f"pi_camera_{self.camera_info.index}"
                
                if camera_key not in _picamera_instances:
                    logger.debug(f"🔍 Creating new Picamera2 instance for {camera_key}")
                    _picamera_instances[camera_key] = Picamera2()
                    _picamera_lock[camera_key] = False
                else:
                    logger.debug(f"🔍 Reusing existing Picamera2 instance for {camera_key}")
                
                self.camera = _picamera_instances[camera_key]
                
                # Check if camera is already configured and running
                if _picamera_lock[camera_key]:
                    logger.debug(f"📷 Camera {camera_key} already configured and running")
                    return
                
                # Mark as being configured
                _picamera_lock[camera_key] = True
                
                # Create configuration with error handling for sensor modes
                try:
                    config = self.camera.create_video_configuration(
                        main={"size": (self.width, self.height), "format": "RGB888"}
                    )
                    self.camera.configure(config)
                    
                    # Get actual configuration to verify settings
                    actual_config = self.camera.camera_configuration()
                    main_stream = actual_config.get('main', {})
                    actual_size = main_stream.get('size', (self.width, self.height))
                    
                    # Update dimensions if they were adjusted by the camera
                    if actual_size != (self.width, self.height):
                        logger.info(f"📷 Camera adjusted resolution from {self.width}x{self.height} to {actual_size[0]}x{actual_size[1]}")
                        self.width, self.height = actual_size
                    
                    self.camera.start()
                    
                    # Test capture to ensure camera is working
                    test_frame = self.camera.capture_array()
                    if test_frame is None:
                        raise RuntimeError("Test capture failed")
                    
                    logger.info(f"📷 Raspberry Pi camera: {self.width}x{self.height} @ {self.fps}fps")
                    
                except Exception as config_error:
                    logger.error(f"Failed to configure picamera2: {config_error}")
                    # Try with a more basic configuration
                    try:
                        config = self.camera.create_video_configuration()
                        self.camera.configure(config)
                        self.camera.start()
                        
                        # Update dimensions from actual config
                        actual_config = self.camera.camera_configuration()
                        main_stream = actual_config.get('main', {})
                        actual_size = main_stream.get('size', (640, 480))
                        self.width, self.height = actual_size
                        
                        logger.info(f"📷 Raspberry Pi camera (fallback config): {self.width}x{self.height}")
                    except Exception as fallback_error:
                        logger.error(f"Fallback configuration also failed: {fallback_error}")
                        raise RuntimeError(f"Could not configure picamera2: {config_error}")
                
            logger.info(f"✅ Initialized {self.camera_info.name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize camera {self.camera_info.index}: {e}")
            self.camera = None
    
    async def recv(self):
        """Receive the next video frame."""
        # Handle camera errors
        if self.error_count >= self.max_errors:
            return self._create_error_frame("Too many camera errors")
        
        if self.camera is None:
            return self._create_error_frame("Camera not initialized")
        
        # Prevent concurrent access
        if self.camera_lock:
            return self._create_error_frame("Camera busy")
        
        try:
            self.camera_lock = True
            frame = self._capture_frame()
            self.frame_count += 1
            
            # Reset error count on successful frame
            self.error_count = 0
            
            # Report performance periodically
            current_time = time.time()
            if current_time - self.last_fps_report >= self.fps_report_interval:
                elapsed = current_time - self.start_time
                actual_fps = self.frame_count / elapsed if elapsed > 0 else 0
                logger.info(f"📊 Camera {self.camera_id}: {actual_fps:.1f} FPS ({self.frame_count} frames)")
                self.last_fps_report = current_time
                    
        except Exception as e:
            self.error_count += 1
            self.last_error_time = time.time()
            logger.error(f"Error capturing frame from camera {self.camera_id}: {e}")
            frame = self._create_error_frame(f"Capture error: {str(e)[:50]}")
        finally:
            self.camera_lock = False
        
        # Add camera ID and timestamp overlay
        frame = self._add_overlay(frame)
        
        # Convert to aiortc VideoFrame
        try:
            video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
            video_frame.pts = self.frame_count
            video_frame.time_base = fractions.Fraction(1, self.fps)
            return video_frame
        except Exception as e:
            logger.error(f"Error creating VideoFrame: {e}")
            # Return a simple black frame
            black_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            video_frame = VideoFrame.from_ndarray(black_frame, format="bgr24")
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
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            
            return frame
            
        elif self.camera_info.type == 'picamera2' and HAS_PICAMERA2:
            frame = self.camera.capture_array()
            
            if frame is None:
                raise RuntimeError("Failed to capture Picamera2 frame")
            
            # Convert RGB to BGR for OpenCV compatibility
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Resize if needed
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            
            return frame
        
        else:
            raise RuntimeError(f"Unsupported camera type: {self.camera_info.type}")
    
    def _create_error_frame(self, error_message: str) -> np.ndarray:
        """Create an error frame with status information."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Add error message
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (0, 0, 255)  # Red
        
        # Multi-line error message
        lines = [
            f"Camera {self.camera_id} Error",
            error_message[:40],  # Truncate long messages
            f"Errors: {self.error_count}/{self.max_errors}"
        ]
        
        line_height = 30
        start_y = (self.height - len(lines) * line_height) // 2
        
        for i, line in enumerate(lines):
            y = start_y + i * line_height
            # Get text size to center it
            text_size = cv2.getTextSize(line, font, 0.6, 2)[0]
            x = (self.width - text_size[0]) // 2
            cv2.putText(frame, line, (x, y), font, 0.6, text_color, 2)
        
        return frame
    
    def _add_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add informational overlay to the frame."""
        # Camera ID in top-left corner
        cv2.putText(
            frame, 
            f"Cam {self.camera_id}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.0, 
            (0, 255, 255),  # Yellow
            2
        )
        
        # Frame count in top-right corner
        frame_text = f"#{self.frame_count}"
        text_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(
            frame,
            frame_text,
            (self.width - text_size[0] - 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # White
            2
        )
        
        # Timestamp in bottom-left corner
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        cv2.putText(
            frame,
            timestamp,
            (10, self.height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),  # Light gray
            1
        )
        
        # Error indicator if there have been recent errors
        if self.error_count > 0 and time.time() - self.last_error_time < 5.0:
            cv2.putText(
                frame,
                f"Errors: {self.error_count}",
                (10, self.height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),  # Orange
                1
            )
        
        return frame
    
    def stop(self):
        """Stop the camera and cleanup resources."""
        if self.camera is not None:
            try:
                if self.camera_info.type == 'opencv':
                    self.camera.release()
                elif self.camera_info.type == 'picamera2' and HAS_PICAMERA2:
                    camera_key = f"pi_camera_{self.camera_info.index}"
                    
                    # Only stop if we're the ones who started it
                    if camera_key in _picamera_lock and _picamera_lock[camera_key]:
                        logger.debug(f"🛑 Stopping picamera2 instance {camera_key}")
                        self.camera.stop()
                        _picamera_lock[camera_key] = False
                    else:
                        logger.debug(f"🛑 Not stopping picamera2 instance {camera_key} - not our responsibility")
                
                logger.info(f"🛑 Stopped camera {self.camera_id} ({self.camera_info.name})")
                
            except Exception as e:
                logger.error(f"Error stopping camera {self.camera_id}: {e}")
            finally:
                self.camera = None
    
    def get_stats(self) -> dict:
        """Get camera performance statistics."""
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        return {
            'camera_id': self.camera_id,
            'camera_name': self.camera_info.name,
            'camera_type': self.camera_info.type,
            'frame_count': self.frame_count,
            'fps': fps,
            'error_count': self.error_count,
            'uptime': elapsed,
            'resolution': (self.width, self.height)
        }