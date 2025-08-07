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
import os
from pathlib import Path
from typing import Optional
from aiortc import VideoStreamTrack
from av import VideoFrame

try:
    from .camera_manager import CameraInfo, HAS_PICAMERA2, get_picamera2_instance
except ImportError:
    # Fallback for direct execution
    from camera_manager import CameraInfo, HAS_PICAMERA2, get_picamera2_instance

logger = logging.getLogger(__name__)

if HAS_PICAMERA2:
    from picamera2 import Picamera2

# Global camera state management for picamera2
_picamera2_configured = False
_picamera2_in_use = False
_picamera2_config_lock = False


class CameraVideoTrack(VideoStreamTrack):
    """Custom video track for camera streaming."""
    
    def __init__(
        self, 
        camera_info: CameraInfo, 
        width: int = 640, 
        height: int = 480, 
        fps: int = 30,
        camera_id: Optional[int] = None,
        debug_preview: bool = False,
        save_preview_frames: bool = False,
        preview_dir: Optional[str] = None
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
        
        # Debug and preview settings
        self.debug_preview = debug_preview or os.environ.get('DEBUG_PREVIEW', '').lower() == 'true'
        self.save_preview_frames = save_preview_frames or os.environ.get('SAVE_PREVIEW_FRAMES', '').lower() == 'true'
        self.preview_dir = preview_dir or os.environ.get('PREVIEW_DIR', '/tmp/webrtc_preview')
        self.preview_interval = int(os.environ.get('PREVIEW_INTERVAL', '30'))  # Save every N frames
        self.last_preview_save = 0
        
        # Frame statistics
        self.frame_stats = {
            'min_brightness': 255,
            'max_brightness': 0,
            'avg_brightness': 0,
            'total_brightness': 0,
            'blank_frames': 0,
            'dark_frames': 0,
            'bright_frames': 0
        }
        
        # Setup preview directory if needed
        if self.save_preview_frames:
            self._setup_preview_dir()
        
        # Initialize camera
        self._init_camera()
        
        # Log debug settings
        if self.debug_preview or self.save_preview_frames:
            logger.info(f"🔍 Debug mode enabled for camera {self.camera_id}:")
            if self.debug_preview:
                logger.info(f"   • Frame analysis enabled")
            if self.save_preview_frames:
                logger.info(f"   • Saving preview frames to: {self.preview_dir}")
                logger.info(f"   • Preview interval: every {self.preview_interval} frames")
    
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
                global _picamera2_configured, _picamera2_in_use, _picamera2_config_lock
                
                # Get the singleton Picamera2 instance
                self.camera = get_picamera2_instance()
                if self.camera is None:
                    raise RuntimeError("Failed to get Picamera2 singleton instance")
                
                logger.debug(f"🔍 Using singleton Picamera2 instance for camera {self.camera_info.index}")
                
                # Check if camera is already configured and running
                if _picamera2_configured and _picamera2_in_use:
                    logger.debug(f"📷 Picamera2 already configured and in use")
                    return
                
                # Prevent concurrent configuration
                if _picamera2_config_lock:
                    logger.debug(f"📷 Picamera2 configuration in progress, waiting...")
                    return
                
                # Mark as being configured
                _picamera2_config_lock = True
                
                # Create configuration with error handling for sensor modes
                try:
                    # Log the requested resolution
                    logger.info(f"🎯 Requesting Picamera2 resolution: {self.width}x{self.height}")
                    
                    # Get sensor modes to find best match
                    sensor_modes = self.camera.sensor_modes
                    if sensor_modes:
                        logger.debug(f"📷 Available sensor modes: {len(sensor_modes)}")
                        
                        # Find best matching mode for requested resolution
                        best_mode = None
                        for mode in sensor_modes:
                            mode_size = mode.get('size', (0, 0))
                            if mode_size[0] >= self.width and mode_size[1] >= self.height:
                                if best_mode is None or (mode_size[0] * mode_size[1] < best_mode['size'][0] * best_mode['size'][1]):
                                    best_mode = mode
                        
                        # If no mode is large enough, use the largest available
                        if best_mode is None:
                            best_mode = max(sensor_modes, key=lambda m: m['size'][0] * m['size'][1])
                            logger.warning(f"⚠️  No sensor mode >= {self.width}x{self.height}, using largest: {best_mode['size']}")
                        else:
                            logger.info(f"✅ Selected sensor mode: {best_mode['size']} for requested {self.width}x{self.height}")
                    
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
                    else:
                        logger.info(f"✅ Camera configured at requested resolution: {self.width}x{self.height}")
                    
                    self.camera.start()
                    
                    # Test capture to ensure camera is working
                    test_frame = self.camera.capture_array()
                    if test_frame is None:
                        raise RuntimeError("Test capture failed")
                    
                    # Mark as configured and in use
                    _picamera2_configured = True
                    _picamera2_in_use = True
                    
                    logger.info(f"📷 Raspberry Pi camera streaming at: {self.width}x{self.height} @ {self.fps}fps")
                    
                    # Log the actual frame size from test capture
                    logger.info(f"🎬 Test frame shape: {test_frame.shape} (H×W×C)")
                    
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
                        
                        # Mark as configured and in use
                        _picamera2_configured = True
                        _picamera2_in_use = True
                        
                        logger.info(f"📷 Raspberry Pi camera (fallback config): {self.width}x{self.height}")
                    except Exception as fallback_error:
                        logger.error(f"Fallback configuration also failed: {fallback_error}")
                        raise RuntimeError(f"Could not configure picamera2: {config_error}")
                    finally:
                        _picamera2_config_lock = False
                
                except Exception as e:
                    _picamera2_config_lock = False
                    raise e
                
            logger.info(f"✅ Initialized {self.camera_info.name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize camera {self.camera_info.index}: {e}")
            self.camera = None
    
    async def recv(self):
        """Receive the next video frame."""
        # Handle camera errors
        if self.error_count >= self.max_errors:
            error_frame = self._create_error_frame("Too many camera errors")
            logger.debug(f"📤 Sending error frame: too many errors")
            return error_frame
        
        if self.camera is None:
            error_frame = self._create_error_frame("Camera not initialized")
            logger.debug(f"📤 Sending error frame: camera not initialized")
            return error_frame
        
        # Prevent concurrent access (only for OpenCV cameras)
        if self.camera_info.type == 'opencv' and self.camera_lock:
            return self._create_error_frame("Camera busy")
        
        # For picamera2, check if it's available
        if self.camera_info.type == 'picamera2' and not _picamera2_configured:
            return self._create_error_frame("Camera not configured")
        
        try:
            # Only set lock for OpenCV cameras
            if self.camera_info.type == 'opencv':
                self.camera_lock = True
            frame = self._capture_frame()
            self.frame_count += 1
            
            # Reset error count on successful frame
            self.error_count = 0
            
            # Analyze frame if debug preview is enabled
            if self.debug_preview:
                self._analyze_frame(frame)
            
            # Save preview frame if enabled
            if self.save_preview_frames and (self.frame_count % self.preview_interval == 0):
                self._save_preview_frame(frame)
            
            # Report performance periodically
            current_time = time.time()
            if current_time - self.last_fps_report >= self.fps_report_interval:
                elapsed = current_time - self.start_time
                actual_fps = self.frame_count / elapsed if elapsed > 0 else 0
                logger.info(f"📊 Camera {self.camera_id}: {actual_fps:.1f} FPS ({self.frame_count} frames)")
                
                # Include frame analysis if debug preview is enabled
                if self.debug_preview and self.frame_count > 0:
                    avg_brightness = self.frame_stats['total_brightness'] / self.frame_count
                    logger.info(f"   🎨 Frame stats: Avg brightness: {avg_brightness:.1f}, "
                               f"Blank: {self.frame_stats['blank_frames']}, "
                               f"Dark: {self.frame_stats['dark_frames']}, "
                               f"Bright: {self.frame_stats['bright_frames']}")
                
                self.last_fps_report = current_time
                    
        except Exception as e:
            self.error_count += 1
            self.last_error_time = time.time()
            logger.error(f"Error capturing frame from camera {self.camera_id}: {e}")
            frame = self._create_error_frame(f"Capture error: {str(e)[:50]}")
        finally:
            # Only release lock for OpenCV cameras
            if self.camera_info.type == 'opencv':
                self.camera_lock = False
        
        # Add camera ID and timestamp overlay
        frame = self._add_overlay(frame)
        
        # Convert to aiortc VideoFrame
        try:
            video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
            video_frame.pts = self.frame_count
            video_frame.time_base = fractions.Fraction(1, self.fps)
            
            # Log frame sending periodically (less frequently for high resolution)
            log_interval = 60 if self.width > 1920 else 30
            if self.frame_count % log_interval == 0 or self.frame_count <= 5:
                logger.info(f"📤 Sending video frame #{self.frame_count} to WebRTC: {frame.shape} {frame.dtype} → PTS={video_frame.pts}")
            
            return video_frame
        except Exception as e:
            logger.error(f"Error creating VideoFrame: {e}")
            # Return a simple black frame
            black_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            video_frame = VideoFrame.from_ndarray(black_frame, format="bgr24")
            video_frame.pts = self.frame_count
            video_frame.time_base = fractions.Fraction(1, self.fps)
            logger.debug(f"📤 Sending fallback black frame #{self.frame_count}")
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
                    global _picamera2_configured, _picamera2_in_use
                    
                    # Only stop if we're the ones using it
                    if _picamera2_configured and _picamera2_in_use:
                        logger.debug(f"🛑 Stopping picamera2 singleton instance")
                        try:
                            self.camera.stop()
                        except Exception as stop_error:
                            logger.debug(f"Error stopping picamera2: {stop_error}")
                        _picamera2_configured = False
                        _picamera2_in_use = False
                    else:
                        logger.debug(f"🛑 Not stopping picamera2 - not currently in use")
                
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
            'resolution': (self.width, self.height),
            'debug_preview': self.debug_preview,
            'save_preview_frames': self.save_preview_frames,
            'frame_stats': self.frame_stats if self.debug_preview else None
        }
    
    def _setup_preview_dir(self):
        """Setup directory for preview frames."""
        try:
            preview_path = Path(self.preview_dir) / f"camera_{self.camera_id}"
            preview_path.mkdir(parents=True, exist_ok=True)
            self.preview_dir = str(preview_path)
            logger.info(f"📁 Preview directory created: {self.preview_dir}")
        except Exception as e:
            logger.error(f"Failed to create preview directory: {e}")
            self.save_preview_frames = False
    
    def _analyze_frame(self, frame: np.ndarray):
        """Analyze frame for debugging purposes."""
        try:
            # Calculate brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            # Update statistics
            self.frame_stats['total_brightness'] += brightness
            self.frame_stats['min_brightness'] = min(self.frame_stats['min_brightness'], brightness)
            self.frame_stats['max_brightness'] = max(self.frame_stats['max_brightness'], brightness)
            
            # Classify frame
            if brightness < 10:
                self.frame_stats['blank_frames'] += 1
                if self.frame_count % 100 == 0:  # Log occasionally
                    logger.warning(f"⚫ Camera {self.camera_id}: Detected blank/black frame (brightness: {brightness:.1f})")
            elif brightness < 50:
                self.frame_stats['dark_frames'] += 1
            elif brightness > 200:
                self.frame_stats['bright_frames'] += 1
            
            # Check for common issues
            if self.frame_count == 1:  # First frame analysis
                logger.info(f"🎬 First frame from camera {self.camera_id}: "
                           f"shape={frame.shape}, dtype={frame.dtype}, brightness={brightness:.1f}")
            
            # Detect if frame is all one color (stuck camera)
            std_dev = np.std(gray)
            if std_dev < 1.0 and self.frame_count % 100 == 0:
                logger.warning(f"🔴 Camera {self.camera_id}: Possible stuck frame detected (std dev: {std_dev:.2f})")
            
        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
    
    def _save_preview_frame(self, frame: np.ndarray):
        """Save a preview frame to disk for debugging."""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            filename = f"frame_{timestamp}_{self.frame_count:06d}.jpg"
            filepath = os.path.join(self.preview_dir, filename)
            
            # Add debug info overlay
            debug_frame = frame.copy()
            
            # Add debug text overlay
            debug_info = [
                f"Frame #{self.frame_count}",
                f"Camera {self.camera_id}",
                f"Time: {timestamp}",
                f"Shape: {frame.shape}"
            ]
            
            y_offset = 30
            for info in debug_info:
                cv2.putText(
                    debug_frame,
                    info,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),  # Green
                    1
                )
                y_offset += 20
            
            # Save frame
            success = cv2.imwrite(filepath, debug_frame)
            
            if success:
                logger.info(f"💾 Saved preview frame: {filename}")
                
                # Also save frame info
                info_file = filepath.replace('.jpg', '_info.txt')
                with open(info_file, 'w') as f:
                    f.write(f"Frame #{self.frame_count}\n")
                    f.write(f"Camera ID: {self.camera_id}\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"Shape: {frame.shape}\n")
                    f.write(f"Data type: {frame.dtype}\n")
                    f.write(f"Min value: {np.min(frame)}\n")
                    f.write(f"Max value: {np.max(frame)}\n")
                    f.write(f"Mean value: {np.mean(frame):.2f}\n")
                    if self.debug_preview:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        f.write(f"Brightness: {np.mean(gray):.2f}\n")
                        f.write(f"Std Dev: {np.std(gray):.2f}\n")
            else:
                logger.error(f"Failed to save preview frame: {filepath}")
            
            self.last_preview_save = self.frame_count
            
        except Exception as e:
            logger.error(f"Error saving preview frame: {e}")