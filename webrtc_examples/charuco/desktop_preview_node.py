"""
Desktop preview node that saves live preview images to disk for local viewing.
Creates image files that can be viewed on the desktop without a web browser.
"""

from typing import Any, Dict, Optional, List
import logging
import numpy as np
import cv2
import os
from dataclasses import dataclass
import sys
from pathlib import Path
import time
import json
import glob

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '..', 'remote_media_processing'))

from remotemedia.core.node import Node

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logger.warning("Pygame not available - direct screen rendering disabled")

logger = logging.getLogger(__name__)


@dataclass
class DesktopPreviewConfig:
    """Configuration for desktop preview rendering."""
    enable_pygame_display: bool = True  # Use pygame for direct screen rendering
    enable_opencv_display: bool = False  # Use cv2.imshow (backup option)
    window_width: int = 1200  # Display window width
    window_height: int = 800  # Display window height
    window_title: str = "ChAruco Live Preview"
    enable_file_saving: bool = False  # Disable file saving for performance
    output_dir: str = "live_preview"
    save_interval: float = 5.0  # Save very infrequently if enabled
    max_saved_images: int = 3
    image_format: str = "jpg"
    # Warping configuration
    enable_warping: bool = True  # Enable homography-based warping
    calibration_file: str = "charuco/camera_calibration.json"  # Camera calibration file
    homography_dir: str = None  # Directory containing homographies.json (auto-find if None)
    canvas_width: int = 1920  # Canvas size for warped views
    canvas_height: int = 1080
    blend_mode: str = "overlay"  # "overlay", "average", "max"
    undistort_images: bool = True  # Apply lens distortion correction


class DesktopPreviewNode(Node):
    """
    Renders live preview frames directly to the screen using pygame or OpenCV.
    
    Input: Dict with 'preview_frame' (from LivePreviewNode)
    Output: Same dict passed through
    """
    
    def __init__(
        self,
        config: Optional[DesktopPreviewConfig] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name or "DesktopPreview")
        self.config = config or DesktopPreviewConfig()
        
        # State tracking
        self.last_save_time = 0
        self.frame_count = 0
        self.pygame_screen = None
        self.pygame_clock = None
        
        # Warping data
        self.calibration_data = {}
        self.homographies = {}
        self.reference_camera = None
        self.last_homography_check = 0
        self.homography_check_interval = 2.0  # Check for new homographies every 2 seconds
        
        # Load calibration data immediately
        if self.config.enable_warping:
            self.load_calibration_data()
            # Don't load homographies yet - will check dynamically
        
        # Initialize pygame display if available and enabled
        if self.config.enable_pygame_display and PYGAME_AVAILABLE:
            try:
                pygame.init()
                pygame.display.set_caption(self.config.window_title)
                self.pygame_screen = pygame.display.set_mode(
                    (self.config.window_width, self.config.window_height)
                )
                self.pygame_clock = pygame.time.Clock()
                logger.info(f"Pygame display initialized: {self.config.window_width}x{self.config.window_height}")
            except Exception as e:
                logger.error(f"Failed to initialize pygame: {e}")
                self.config.enable_pygame_display = False
        
        # Fallback to OpenCV if pygame not available
        if not self.config.enable_pygame_display and self.config.enable_opencv_display:
            try:
                cv2.namedWindow(self.config.window_title, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.config.window_title, 
                               self.config.window_width, self.config.window_height)
                logger.info("OpenCV display window created as fallback")
            except Exception as e:
                logger.warning(f"Could not create OpenCV display window: {e}")
                self.config.enable_opencv_display = False
        
        # Optional file saving
        if self.config.enable_file_saving:
            self.output_path = Path(self.config.output_dir)
            self.output_path.mkdir(exist_ok=True)
            self.saved_files = []
            logger.info(f"File saving enabled to: {self.output_path.absolute()}")
        
        logger.info(f"Desktop preview initialized - Direct screen rendering enabled")
    
    def load_calibration_data(self):
        """Load camera calibration data."""
        try:
            cal_path = Path(self.config.calibration_file)
            if not cal_path.exists():
                logger.warning(f"Calibration file not found: {cal_path}")
                return
            
            with open(cal_path, 'r') as f:
                self.calibration_data = json.load(f)
            
            logger.info(f"📷 Loaded calibration for {len(self.calibration_data)} cameras")
            
        except Exception as e:
            logger.error(f"Error loading calibration data: {e}")
            self.config.enable_warping = False
    
    def load_homography_data(self):
        """Load homography matrices."""
        try:
            if self.config.homography_dir:
                homo_dir = Path(self.config.homography_dir)
            else:
                # Auto-find most recent calibration results
                cal_dirs = sorted(glob.glob("calibration_results_*"), reverse=True)
                if not cal_dirs:
                    logger.warning("No calibration results directories found")
                    return
                homo_dir = Path(cal_dirs[0])
                logger.info(f"🔍 Auto-selected homography directory: {homo_dir}")
            
            homo_file = homo_dir / "homographies.json"
            if not homo_file.exists():
                logger.warning(f"Homography file not found: {homo_file}")
                return
            
            with open(homo_file, 'r') as f:
                homo_data = json.load(f)
            
            # Convert to numpy arrays
            for camera_key, matrix_data in homo_data.items():
                self.homographies[camera_key] = np.array(matrix_data, dtype=np.float32)
            
            # Find reference camera (identity matrix)
            for camera_key, H in self.homographies.items():
                if np.allclose(H, np.eye(3), atol=1e-6):
                    self.reference_camera = camera_key
                    break
            
            logger.info(f"🔄 Loaded {len(self.homographies)} homography matrices")
            logger.info(f"🎯 Reference camera: {self.reference_camera}")
            
        except Exception as e:
            logger.error(f"Error loading homography data: {e}")
            self.config.enable_warping = False
    
    def undistort_image(self, image: np.ndarray, camera_id: str) -> np.ndarray:
        """Undistort image using camera calibration parameters."""
        if not self.config.undistort_images or camera_id not in self.calibration_data:
            return image
        
        try:
            cal_data = self.calibration_data[camera_id]
            camera_matrix = np.array(cal_data['camera_matrix'], dtype=np.float32)
            dist_coeffs = np.array(cal_data['dist_coeffs'], dtype=np.float32).flatten()
            
            return cv2.undistort(image, camera_matrix, dist_coeffs)
        except Exception as e:
            logger.error(f"Error undistorting image for camera {camera_id}: {e}")
            return image
    
    def warp_image(self, image: np.ndarray, camera_key: str) -> np.ndarray:
        """Warp image using homography matrix."""
        if camera_key not in self.homographies:
            return image
        
        try:
            H = self.homographies[camera_key]
            canvas_size = (self.config.canvas_width, self.config.canvas_height)
            
            warped = cv2.warpPerspective(
                image, H, canvas_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
            
            return warped
        except Exception as e:
            logger.error(f"Error warping image for {camera_key}: {e}")
            return image
    
    def blend_images(self, images_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Blend multiple warped images into a combined view."""
        if not images_dict:
            return np.zeros((self.config.canvas_height, self.config.canvas_width, 3), dtype=np.uint8)
        
        images = list(images_dict.values())
        
        if self.config.blend_mode == "overlay":
            # Additive blending preserving all camera data
            combined = np.zeros_like(images[0], dtype=np.float32)
            pixel_count = np.zeros(images[0].shape[:2], dtype=np.float32)
            
            for img in images:
                # Create mask for non-black pixels
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mask = (gray > 5).astype(np.float32)
                
                combined += img.astype(np.float32) * mask[..., np.newaxis]
                pixel_count += mask
            
            # Average overlapping areas
            pixel_count[pixel_count == 0] = 1
            combined = (combined / pixel_count[..., np.newaxis]).astype(np.uint8)
            
        elif self.config.blend_mode == "average":
            combined = np.mean(images, axis=0).astype(np.uint8)
            
        elif self.config.blend_mode == "max":
            combined = np.max(images, axis=0)
            
        else:
            combined = images[0]
        
        return combined
    
    def create_warped_view(self, camera_frames: Dict[str, np.ndarray]) -> np.ndarray:
        """Create combined warped view from multiple camera frames."""
        warped_images = {}
        
        for camera_id, frame in camera_frames.items():
            # Convert camera ID to camera_key format
            camera_key = f"camera_{camera_id}" if not camera_id.startswith("camera_") else camera_id
            
            # Undistort if enabled
            if self.config.undistort_images:
                frame = self.undistort_image(frame, camera_id)
            
            # Warp the image
            warped = self.warp_image(frame, camera_key)
            warped_images[camera_key] = warped
        
        # Blend all warped images
        return self.blend_images(warped_images)
    
    def create_warped_view_with_homographies(self, camera_frames: Dict[str, np.ndarray], homographies: Dict[str, np.ndarray]) -> np.ndarray:
        """Create combined warped view from multiple camera frames using provided homographies."""
        warped_images = {}
        
        for camera_id, frame in camera_frames.items():
            # Convert camera ID to camera_key format
            camera_key = f"camera_{camera_id}" if not camera_id.startswith("camera_") else camera_id
            
            # Undistort if enabled
            if self.config.undistort_images:
                frame = self.undistort_image(frame, camera_id)
            
            # Warp the image using provided homographies
            if camera_key in homographies:
                H = homographies[camera_key]
                canvas_size = (self.config.canvas_width, self.config.canvas_height)
                
                warped = cv2.warpPerspective(
                    frame, H, canvas_size,
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0)
                )
                warped_images[camera_key] = warped
        
        # Blend all warped images
        return self.blend_images(warped_images)
    
    def render_pygame(self, image: np.ndarray):
        """Render image directly to pygame screen."""
        if not self.pygame_screen:
            return
        
        try:
            # Handle pygame events (prevent window from becoming unresponsive)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    self.pygame_screen = None
                    return
            
            # Convert OpenCV image (BGR) to pygame surface (RGB)
            height, width = image.shape[:2]
            
            # Resize image to fit window while maintaining aspect ratio
            window_w, window_h = self.config.window_width, self.config.window_height
            scale = min(window_w / width, window_h / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            resized_image = cv2.resize(image, (new_width, new_height))
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            
            # Create pygame surface
            surface = pygame.surfarray.make_surface(rgb_image.swapaxes(0, 1))
            
            # Clear screen and center the image
            self.pygame_screen.fill((0, 0, 0))  # Black background
            x_offset = (window_w - new_width) // 2
            y_offset = (window_h - new_height) // 2
            self.pygame_screen.blit(surface, (x_offset, y_offset))
            
            # Update display
            pygame.display.flip()
            
            # Control frame rate
            if self.pygame_clock:
                self.pygame_clock.tick(30)  # 30 FPS max
                
        except Exception as e:
            logger.error(f"Pygame rendering error: {e}")
            self.config.enable_pygame_display = False
    
    def render_opencv(self, image: np.ndarray):
        """Render image using OpenCV (fallback)."""
        if not self.config.enable_opencv_display:
            return
            
        try:
            # Resize to fit window
            height, width = image.shape[:2]
            window_w, window_h = self.config.window_width, self.config.window_height
            scale = min(window_w / width, window_h / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            resized_image = cv2.resize(image, (new_width, new_height))
            
            cv2.imshow(self.config.window_title, resized_image)
            cv2.waitKey(1)  # Non-blocking
        except Exception as e:
            logger.error(f"OpenCV rendering error: {e}")
            self.config.enable_opencv_display = False
    
    def should_save_frame(self) -> bool:
        """Check if frame should be saved based on configuration."""
        current_time = time.time()
        if current_time - self.last_save_time >= self.config.save_interval:
            self.last_save_time = current_time
            return True
        return False
    
    def save_preview_image(self, image: np.ndarray) -> str:
        """Save preview image to disk and return the file path."""
        self.frame_count += 1
        
        # Generate filename
        filename = f"live_preview_{self.frame_count:06d}.{self.config.image_format}"
        filepath = self.output_path / filename
        
        # Save image
        try:
            success = cv2.imwrite(str(filepath), image)
            if success:
                self.saved_files.append(str(filepath))
                
                # Clean up old files
                if len(self.saved_files) > self.config.max_saved_images:
                    old_file = self.saved_files.pop(0)
                    try:
                        os.remove(old_file)
                    except OSError:
                        pass  # File might already be deleted
                
                return str(filepath)
            else:
                logger.error(f"Failed to save image to {filepath}")
                return ""
        except Exception as e:
            logger.error(f"Error saving preview image: {e}")
            return ""
    
    def display_opencv(self, image: np.ndarray):
        """Display image using OpenCV if enabled."""
        if not self.config.enable_opencv_display:
            return
        
        try:
            cv2.imshow("ChAruco Live Preview", image)
            cv2.waitKey(1)  # Non-blocking wait
        except Exception as e:
            logger.error(f"Error displaying OpenCV preview: {e}")
            self.config.enable_opencv_display = False
    
    def create_latest_symlink(self, filepath: str):
        """Create a 'latest.jpg' symlink pointing to the most recent image."""
        try:
            latest_path = self.output_path / f"latest.{self.config.image_format}"
            
            # Remove existing symlink if it exists
            if latest_path.is_symlink() or latest_path.exists():
                latest_path.unlink()
            
            # Create new symlink
            latest_path.symlink_to(Path(filepath).name)
        except Exception as e:
            logger.debug(f"Could not create latest symlink: {e}")
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and render preview frames directly to screen."""
        try:
            preview_frame = data.get('preview_frame')
            camera_frames = data.get('camera_frames', {})  # Multiple camera frames
            images = data.get('images', [])  # List format from pipeline
            frames = data.get('frames', [])  # Direct frame list from pipeline
            
            # Check for homography data from previous nodes
            pipeline_homographies = data.get('homographies', [])
            pipeline_valid_cameras = data.get('valid_cameras', [])
            pipeline_reference_camera = data.get('reference_camera')
            
            self.frame_count += 1
            
            # Debug logging for first few frames or when homographies change  
            if self.frame_count <= 5 or (pipeline_homographies and len(pipeline_homographies) > 0) or self.frame_count % 50 == 1:
                logger.info(f"🔍 DesktopPreview frame #{self.frame_count} debug:")
                logger.info(f"   Data keys: {list(data.keys())}")
                logger.info(f"   Pipeline homographies: {len(pipeline_homographies) if pipeline_homographies else 0}")
                logger.info(f"   Pipeline valid cameras: {pipeline_valid_cameras}")
                logger.info(f"   Pipeline reference camera: {pipeline_reference_camera}")
                logger.info(f"   Current stored homographies: {len(self.homographies)}")
                logger.info(f"   Preview frame available: {preview_frame is not None}")
                logger.info(f"   Frames available: {len(frames)}")
                logger.info(f"   Warping enabled: {self.config.enable_warping}")
                logger.info(f"   Homographies computed: {data.get('homographies_computed', False)}")
            
            # Use homographies directly from pipeline data (don't store them)
            current_homographies = {}
            if pipeline_homographies and len(pipeline_homographies) > 0:
                # Convert list homographies to dictionary format
                for i, H in enumerate(pipeline_homographies):
                    if H is not None:
                        current_homographies[f"camera_{i}"] = H
                
                if len(current_homographies) > 0:
                    logger.info(f"🔄 Using homographies from pipeline: {len(current_homographies)} matrices")
                    logger.info(f"🎯 Reference camera: {pipeline_reference_camera}")
            else:
                current_homographies = {}
            
            # Determine what to render
            render_frame = None
            
            if self.config.enable_warping and current_homographies and frames and len(frames) > 1:
                # Generate warped view directly from frames using current homographies
                logger.info(f"🎬 Generating warped view from {len(frames)} frames with {len(current_homographies)} homographies")
                cam_dict = {str(i): img for i, img in enumerate(frames)}
                render_frame = self.create_warped_view_with_homographies(cam_dict, current_homographies)
                
                # Log success
                if self.frame_count % 30 == 1:
                    logger.info(f"✅ Warped view generated: {render_frame.shape}")
                    
            elif preview_frame is not None:
                # Regular preview mode
                render_frame = preview_frame
                if self.frame_count % 30 == 1:
                    logger.info(f"📺 Using preview frame: {render_frame.shape}")
            else:
                if self.frame_count % 30 == 1:
                    logger.info(f"⚠️ No frame to render - homographies: {len(self.homographies)}, frames: {len(frames)}")
                return data  # No frame to render
            
            # DIRECT SCREEN RENDERING - Primary method
            if self.config.enable_pygame_display and self.pygame_screen:
                self.render_pygame(render_frame)
            elif self.config.enable_opencv_display:
                self.render_opencv(render_frame)
            
            # Optional file saving (disabled by default for performance)
            if self.config.enable_file_saving and self.should_save_frame():
                saved_path = self.save_preview_image(render_frame)
                if saved_path and self.frame_count % 50 == 1:
                    logger.info(f"💾 Saved preview: {saved_path}")
                data['saved_preview_path'] = saved_path
            
            # Log rendering status occasionally
            if self.frame_count % 100 == 1:
                render_method = "pygame" if self.pygame_screen else "opencv" if self.config.enable_opencv_display else "none"
                warp_status = "warped" if self.config.enable_warping and current_homographies else "regular"
                logger.info(f"🖥️  Direct rendering: {render_method} ({warp_status}, {self.frame_count} frames)")
            
            # Add warped frame to output data
            if self.config.enable_warping and render_frame is not None:
                data['warped_preview'] = render_frame
            
            return data
            
        except Exception as e:
            logger.error(f"Error in desktop preview: {e}")
            return data
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.pygame_screen:
                pygame.quit()
            if self.config.enable_opencv_display:
                cv2.destroyAllWindows()
        except Exception as e:
            logger.error(f"Error cleaning up display: {e}")
    
    def get_latest_preview_path(self) -> Optional[str]:
        """Get path to the latest saved preview image."""
        latest_path = self.output_path / f"latest.{self.config.image_format}"
        if latest_path.exists():
            return str(latest_path)
        return None
    
    def get_preview_directory(self) -> str:
        """Get the directory path where preview images are saved."""
        return str(self.output_path.absolute())