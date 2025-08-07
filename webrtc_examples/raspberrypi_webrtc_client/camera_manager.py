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

# Global Picamera2 singleton management
_picamera2_instance = None
_picamera2_initialized = False
_picamera2_sensor_modes = None
_picamera2_camera_name = None

def get_picamera2_instance():
    """Get the singleton Picamera2 instance."""
    global _picamera2_instance, _picamera2_initialized, _picamera2_sensor_modes, _picamera2_camera_name
    
    if not HAS_PICAMERA2:
        return None
        
    if _picamera2_instance is None and not _picamera2_initialized:
        try:
            logger.debug("🔍 Creating singleton Picamera2 instance...")
            _picamera2_instance = Picamera2()
            
            # Cache sensor modes and camera identification on first access
            _picamera2_sensor_modes = _picamera2_instance.sensor_modes
            _picamera2_camera_name = "Raspberry Pi Camera"
            
            if _picamera2_sensor_modes:
                # Try to identify specific camera types
                if any(mode.get('size') == (2560, 400) for mode in _picamera2_sensor_modes):
                    _picamera2_camera_name = "Arducam PiVariety Camera"
                    logger.info("🎭 Detected Arducam PiVariety camera")
                elif any(mode.get('size', (0,0))[0] >= 5000 for mode in _picamera2_sensor_modes):
                    _picamera2_camera_name = "High Resolution Pi Camera"
            
            _picamera2_initialized = True
            logger.debug(f"✅ Singleton Picamera2 instance created: {_picamera2_camera_name}")
            
        except Exception as e:
            logger.debug(f"❌ Failed to create Picamera2 instance: {e}")
            _picamera2_initialized = True  # Mark as attempted
            return None
    
    return _picamera2_instance

def get_picamera2_info():
    """Get cached Picamera2 sensor information without creating new instance."""
    get_picamera2_instance()  # Ensure instance is created
    return _picamera2_sensor_modes, _picamera2_camera_name

def close_picamera2_instance():
    """Close the singleton Picamera2 instance."""
    global _picamera2_instance, _picamera2_initialized
    
    if _picamera2_instance is not None:
        try:
            _picamera2_instance.close()
            logger.debug("🛑 Closed singleton Picamera2 instance")
        except Exception as e:
            logger.debug(f"Error closing Picamera2: {e}")
        finally:
            _picamera2_instance = None
            _picamera2_initialized = False


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
        self._max_picamera_resolution = None  # Store max resolution for picamera
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
        """Discover Raspberry Pi camera using singleton instance."""
        if not HAS_PICAMERA2:
            logger.debug("❌ _discover_pi_camera: HAS_PICAMERA2 is False")
            return None
        
        logger.debug("🔍 _discover_pi_camera: Starting picamera2 detection...")
        try:
            # Get cached sensor info without creating new instance
            sensor_modes, camera_name = get_picamera2_info()
            
            if sensor_modes is None or camera_name is None:
                logger.debug("❌ _discover_pi_camera: Failed to get Picamera2 info")
                return None
            
            default_width, default_height = 640, 480
            max_width, max_height = 640, 480
            
            if sensor_modes:
                # Log available sensor modes for debugging
                logger.info(f"📷 Available sensor modes for {camera_name}: {len(sensor_modes)}")
                
                # Find the maximum resolution available
                for i, mode in enumerate(sensor_modes):
                    mode_size = mode.get('size', (0, 0))
                    width, height = mode_size[0], mode_size[1]
                    logger.info(f"  Mode {i}: {width}x{height}")
                    
                    # Track maximum resolution
                    if width * height > max_width * max_height:
                        max_width, max_height = width, height
                
                logger.info(f"📷 Maximum available resolution: {max_width}x{max_height}")
                
                # Set default to maximum resolution found
                default_width, default_height = max_width, max_height
                
                # For known camera types, we might want to limit resolution for performance
                if "PiVariety" in camera_name:
                    # PiVariety cameras might have very high res, limit for streaming
                    if default_width > 2560:
                        default_width, default_height = 2560, 1440
                elif "High Resolution" in camera_name:
                    # High res camera, use full resolution if reasonable
                    if default_width > 4056:  # Pi Camera HQ max is 4056x3040
                        default_width, default_height = 4056, 3040
                
                # Store the maximum resolution info
                self._max_picamera_resolution = (max_width, max_height)
            
            # Create camera info without testing configuration (avoids conflicts)
            camera_info = CameraInfo(
                index=0,  # Will be set later
                name=camera_name,
                type='picamera2',
                width=default_width,
                height=default_height,
                fps=30.0,
                device_path="/dev/video0"
            )
            
            logger.info(f"📷 Found {camera_name}: Default {default_width}x{default_height}, Max: {max_width}x{max_height}")
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
    
    def get_max_picamera_resolution(self) -> Optional[tuple]:
        """Get the maximum resolution available for the Picamera2."""
        return self._max_picamera_resolution
    
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
                # Check if singleton instance is available (don't create new one)
                picam = get_picamera2_instance()
                return picam is not None
            
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
            # Get Raspberry Pi camera capabilities from cached info
            try:
                sensor_modes, camera_name = get_picamera2_info()
                if sensor_modes:
                    max_res = max([(mode['size']) for mode in sensor_modes], key=lambda x: x[0]*x[1]) if sensor_modes else None
                    available_resolutions = [(mode['size']) for mode in sensor_modes] if sensor_modes else []
                    capabilities.update({
                        'sensor_modes': len(sensor_modes),
                        'max_resolution': max_res,
                        'available_resolutions': available_resolutions
                    })
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