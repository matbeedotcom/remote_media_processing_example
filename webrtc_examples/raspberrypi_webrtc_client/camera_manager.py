"""
Camera Manager for Raspberry Pi WebRTC Client

Handles camera discovery, enumeration, and management for multiple camera types
including USB cameras and Raspberry Pi cameras.
"""

import cv2
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

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
    device_path: Optional[str] = None


class CameraManager:
    """Manages camera discovery and access."""
    
    def __init__(self):
        self.cameras: Dict[int, CameraInfo] = {}
        self.discover_cameras()
    
    def discover_cameras(self):
        """Discover all available cameras."""
        self.cameras.clear()
        camera_index = 0
        
        logger.info("🔍 Discovering cameras...")
        
        # Try to find OpenCV cameras
        logger.debug("🔍 Starting OpenCV camera discovery...")
        opencv_cameras = self._discover_opencv_cameras()
        logger.debug(f"🔍 OpenCV discovery returned {len(opencv_cameras)} camera(s)")
        
        for cam_info in opencv_cameras:
            self.cameras[camera_index] = cam_info
            cam_info.index = camera_index
            logger.debug(f"✅ Added OpenCV camera {camera_index}: {cam_info.name}")
            camera_index += 1
        
        # Try to find Raspberry Pi camera
        logger.debug("🔍 Starting Raspberry Pi camera discovery...")
        logger.debug(f"🔍 HAS_PICAMERA2 = {HAS_PICAMERA2}")
        
        if not HAS_PICAMERA2:
            logger.debug("❌ Skipping Pi camera discovery - picamera2 not available")
        else:
            logger.debug("✅ picamera2 is available, attempting Pi camera discovery...")
            
        pi_camera = self._discover_pi_camera()
        if pi_camera:
            pi_camera.index = camera_index
            self.cameras[camera_index] = pi_camera
            logger.debug(f"✅ Added Pi camera {camera_index}: {pi_camera.name}")
            camera_index += 1
        else:
            logger.debug("❌ No Raspberry Pi camera found")
        
        logger.info(f"🔍 Discovered {len(self.cameras)} camera(s)")
    
    def _discover_opencv_cameras(self) -> List[CameraInfo]:
        """Discover OpenCV/USB cameras."""
        cameras = []
        
        for i in range(8):  # Check first 8 camera indices
            cap = None
            try:
                cap = cv2.VideoCapture(i)
                if not cap.isOpened():
                    continue
                
                # Get camera properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Set a reasonable resolution for testing
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                # Try to read a frame to verify camera works
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    camera_info = CameraInfo(
                        index=0,  # Will be set later
                        name=f"USB Camera {i}",
                        type='opencv',
                        width=actual_width if actual_width > 0 else 640,
                        height=actual_height if actual_height > 0 else 480,
                        fps=fps if fps > 0 else 30.0,
                        device_path=f"/dev/video{i}"
                    )
                    cameras.append(camera_info)
                    logger.info(f"📹 Found OpenCV camera {i}: {actual_width}x{actual_height} @ {fps}fps")
                    
            except Exception as e:
                logger.debug(f"Error checking OpenCV camera {i}: {e}")
            finally:
                if cap is not None:
                    cap.release()
        
        return cameras
    
    def _discover_pi_camera(self) -> Optional[CameraInfo]:
        """Discover Raspberry Pi camera."""
        if not HAS_PICAMERA2:
            logger.debug("❌ _discover_pi_camera: HAS_PICAMERA2 is False")
            return None
        
        logger.debug("🔍 _discover_pi_camera: Starting picamera2 detection...")
        try:
            picam = Picamera2()
            
            # Get sensor modes to identify camera type and capabilities
            sensor_modes = picam.sensor_modes
            camera_name = "Raspberry Pi Camera"
            default_width, default_height = 640, 480
            
            if sensor_modes:
                # Log available sensor modes for debugging
                logger.debug(f"Available sensor modes: {len(sensor_modes)}")
                for i, mode in enumerate(sensor_modes):
                    logger.debug(f"  Mode {i}: {mode}")
                
                # Try to identify specific camera types
                first_mode = sensor_modes[0]
                mode_size = first_mode.get('size', (640, 480))
                
                # Check for Arducam PiVariety camera patterns
                if any(mode.get('size') == (2560, 400) for mode in sensor_modes):
                    camera_name = "Arducam PiVariety Camera"
                    # Use a reasonable default resolution for streaming
                    default_width, default_height = 640, 480
                    logger.info("🎭 Detected Arducam PiVariety camera")
                elif any(mode.get('size', (0,0))[0] >= 5000 for mode in sensor_modes):
                    camera_name = "High Resolution Pi Camera"
                    default_width, default_height = 1280, 720
                else:
                    # Use the first available mode as default
                    default_width, default_height = mode_size[0], mode_size[1]
                    # But limit to reasonable streaming resolution
                    if default_width > 1920:
                        default_width, default_height = 1280, 720
                    elif default_width > 1280:
                        default_width, default_height = 1280, 720
                    elif default_width < 640:
                        default_width, default_height = 640, 480
            
            # Test camera configuration with a suitable resolution
            try:
                test_config = picam.create_video_configuration(
                    main={"size": (default_width, default_height), "format": "RGB888"}
                )
                picam.configure(test_config)
                
                # Test that we can start and get the actual configuration
                picam.start()
                actual_config = picam.camera_configuration()
                picam.stop()
                
                # Get actual resolution from the configuration
                main_stream = actual_config.get('main', {})
                actual_size = main_stream.get('size', (default_width, default_height))
                actual_width, actual_height = actual_size
                
                camera_info = CameraInfo(
                    index=0,  # Will be set later
                    name=camera_name,
                    type='picamera2',
                    width=actual_width,
                    height=actual_height,
                    fps=30.0,
                    device_path="/dev/video0"
                )
                
                logger.info(f"📷 Found {camera_name}: {actual_width}x{actual_height}")
                picam.close()
                return camera_info
                
            except Exception as config_error:
                logger.debug(f"Error testing camera configuration: {config_error}")
                # Fall back to basic info without testing
                camera_info = CameraInfo(
                    index=0,
                    name=camera_name,
                    type='picamera2',
                    width=default_width,
                    height=default_height,
                    fps=30.0,
                    device_path="/dev/video0"
                )
                
                logger.info(f"📷 Found {camera_name}: {default_width}x{default_height} (untested)")
                picam.close()
                return camera_info
            
        except Exception as e:
            logger.debug(f"❌ _discover_pi_camera: Raspberry Pi camera detection failed: {e}")
            logger.debug(f"❌ _discover_pi_camera: Exception type: {type(e).__name__}")
            return None
    
    def get_camera_list(self) -> List[CameraInfo]:
        """Get list of all available cameras."""
        return list(self.cameras.values())
    
    def get_camera(self, index: int) -> Optional[CameraInfo]:
        """Get camera info by index."""
        return self.cameras.get(index)
    
    def refresh_cameras(self):
        """Refresh camera discovery."""
        logger.info("🔄 Refreshing camera list...")
        self.discover_cameras()
    
    def validate_camera(self, index: int) -> bool:
        """Validate that a camera is still available and working."""
        camera_info = self.get_camera(index)
        if not camera_info:
            return False
        
        try:
            if camera_info.type == 'opencv':
                # Try to open and read from OpenCV camera
                cap = cv2.VideoCapture(index)
                if not cap.isOpened():
                    return False
                ret, frame = cap.read()
                cap.release()
                return ret and frame is not None
            
            elif camera_info.type == 'picamera2' and HAS_PICAMERA2:
                # Try to create Picamera2 instance
                picam = Picamera2()
                picam.close()
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Camera {index} validation failed: {e}")
            return False
    
    def get_camera_capabilities(self, index: int) -> Dict[str, any]:
        """Get detailed camera capabilities."""
        camera_info = self.get_camera(index)
        if not camera_info:
            return {}
        
        capabilities = {
            'index': camera_info.index,
            'name': camera_info.name,
            'type': camera_info.type,
            'current_resolution': (camera_info.width, camera_info.height),
            'current_fps': camera_info.fps,
            'device_path': camera_info.device_path
        }
        
        if camera_info.type == 'opencv':
            # Get additional OpenCV capabilities
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                capabilities.update({
                    'supported_resolutions': self._get_opencv_resolutions(cap),
                    'brightness': cap.get(cv2.CAP_PROP_BRIGHTNESS),
                    'contrast': cap.get(cv2.CAP_PROP_CONTRAST),
                    'saturation': cap.get(cv2.CAP_PROP_SATURATION),
                    'gain': cap.get(cv2.CAP_PROP_GAIN),
                    'exposure': cap.get(cv2.CAP_PROP_EXPOSURE)
                })
            cap.release()
        
        elif camera_info.type == 'picamera2' and HAS_PICAMERA2:
            # Get Raspberry Pi camera capabilities
            try:
                picam = Picamera2()
                sensor_modes = picam.sensor_modes
                capabilities.update({
                    'sensor_modes': len(sensor_modes),
                    'max_resolution': max([(mode['size']) for mode in sensor_modes], key=lambda x: x[0]*x[1]) if sensor_modes else None
                })
                picam.close()
            except Exception as e:
                logger.debug(f"Could not get Pi camera capabilities: {e}")
        
        return capabilities
    
    def _get_opencv_resolutions(self, cap) -> List[tuple]:
        """Get supported resolutions for OpenCV camera."""
        # Common resolutions to test
        test_resolutions = [
            (320, 240), (640, 480), (800, 600), (1024, 768),
            (1280, 720), (1280, 960), (1920, 1080), (2560, 1440)
        ]
        
        supported = []
        original_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        original_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        for width, height in test_resolutions:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_width == width and actual_height == height:
                supported.append((width, height))
        
        # Restore original resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, original_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, original_height)
        
        return supported