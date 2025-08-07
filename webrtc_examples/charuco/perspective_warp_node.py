"""
Perspective warp node for aligning multiple camera views to a common reference frame.
Computes homographies from camera poses and warps images.
"""

from typing import Any, Dict, Optional, List, Tuple
import logging
import numpy as np
import cv2
from dataclasses import dataclass
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '..', 'remote_media_processing'))

from remotemedia.core.node import Node
from charuco_detection_node import PoseResult

logger = logging.getLogger(__name__)


@dataclass
class WarpConfig:
    """Configuration for perspective warping."""
    output_width: int = 1920
    output_height: int = 1080
    reference_camera: int = 0  # Index of reference camera
    interpolation: int = cv2.INTER_LINEAR
    border_mode: int = cv2.BORDER_CONSTANT
    border_value: Tuple[int, int, int] = (0, 0, 0)


class PerspectiveWarpNode(Node):
    """
    Computes homographies from camera poses and warps images to align them.
    
    Input: Dict with 'images' (List of numpy arrays), 'poses' (List[PoseResult]),
           'camera_matrices' (List of 3x3 matrices), optional 'reference_index'
    Output: Dict with 'warped_images', 'homographies', 'combined_view'
    """
    
    def __init__(
        self,
        config: Optional[WarpConfig] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name or "PerspectiveWarp")
        self.config = config or WarpConfig()
        
        # Cache for homographies
        self.cached_homographies: Dict[int, np.ndarray] = {}
        self.last_valid_homographies: Dict[int, np.ndarray] = {}
        
        logger.info(f"Initialized perspective warp with output size "
                   f"{self.config.output_width}x{self.config.output_height}")
    
    def compute_homography_from_pose(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Compute homography from camera pose (rvec, tvec) and intrinsics.
        Assumes the ChAruco board defines the world plane (Z=0).
        """
        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # For planar homography, we use columns 0,1 of R and the translation
        # H = K * [r1 r2 t] where r1, r2 are first two columns of R
        H = camera_matrix @ np.hstack([R[:, :2], tvec])
        
        return H
    
    def compute_relative_homography(
        self,
        H_ref: np.ndarray,
        H_target: np.ndarray
    ) -> np.ndarray:
        """
        Compute homography that maps target camera view to reference camera view.
        H_relative = H_ref * inv(H_target)
        """
        try:
            H_target_inv = np.linalg.inv(H_target)
            H_relative = H_ref @ H_target_inv
            # Normalize homography
            H_relative = H_relative / H_relative[2, 2]
            return H_relative
        except np.linalg.LinAlgError:
            logger.warning("Failed to invert homography")
            return np.eye(3)
    
    def warp_image(
        self,
        image: np.ndarray,
        homography: np.ndarray
    ) -> np.ndarray:
        """
        Warp image using homography matrix.
        """
        warped = cv2.warpPerspective(
            image,
            homography,
            (self.config.output_width, self.config.output_height),
            flags=self.config.interpolation,
            borderMode=self.config.border_mode,
            borderValue=self.config.border_value
        )
        return warped
    
    def create_combined_view(
        self,
        warped_images: List[np.ndarray],
        blend_mode: str = 'average'
    ) -> np.ndarray:
        """
        Combine multiple warped images into a single view.
        
        Args:
            warped_images: List of warped images
            blend_mode: 'average', 'max', 'overlay', or 'grid'
        """
        if not warped_images:
            return np.zeros((self.config.output_height, self.config.output_width, 3), dtype=np.uint8)
        
        if blend_mode == 'average':
            # Simple averaging
            combined = np.mean(warped_images, axis=0).astype(np.uint8)
        
        elif blend_mode == 'max':
            # Maximum intensity
            combined = np.max(warped_images, axis=0)
        
        elif blend_mode == 'overlay':
            # Overlay with transparency
            combined = warped_images[0].copy()
            for img in warped_images[1:]:
                mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 0
                combined[mask] = img[mask]
        
        elif blend_mode == 'grid':
            # Create grid view
            n = len(warped_images)
            if n == 1:
                combined = warped_images[0]
            elif n == 2:
                combined = np.hstack(warped_images[:2])
            elif n <= 4:
                # 2x2 grid
                row1 = np.hstack(warped_images[:2])
                row2 = np.hstack(warped_images[2:4] if n > 2 else [np.zeros_like(warped_images[0])] * (4-n))
                combined = np.vstack([row1, row2])
            else:
                # Use first 4 images
                combined = self.create_combined_view(warped_images[:4], 'grid')
        
        else:
            combined = warped_images[0]
        
        return combined
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process images and poses to create warped views."""
        try:
            images = data.get('images', [])
            poses = data.get('poses', [])
            camera_matrices = data.get('camera_matrices', [])
            reference_index = data.get('reference_index', self.config.reference_camera)
            blend_mode = data.get('blend_mode', 'overlay')
            
            if not images:
                logger.warning("No images provided")
                return {'warped_images': [], 'homographies': [], 'combined_view': None}
            
            # Ensure we have camera matrices for all cameras
            if len(camera_matrices) < len(images):
                logger.warning(f"Insufficient camera matrices: {len(camera_matrices)} < {len(images)}")
                # Use identity matrix as fallback
                default_K = np.array([[1000, 0, images[0].shape[1]/2],
                                     [0, 1000, images[0].shape[0]/2],
                                     [0, 0, 1]], dtype=np.float32)
                camera_matrices.extend([default_K] * (len(images) - len(camera_matrices)))
            
            # Compute homographies for each camera
            homographies = []
            valid_indices = []
            
            # Get reference homography if pose is available
            H_ref = None
            if reference_index < len(poses) and poses[reference_index].rvec is not None:
                H_ref = self.compute_homography_from_pose(
                    poses[reference_index].rvec,
                    poses[reference_index].tvec,
                    camera_matrices[reference_index]
                )
            
            for i, (image, pose) in enumerate(zip(images, poses[:len(images)])):
                if pose.rvec is not None and pose.tvec is not None:
                    # Compute homography from pose
                    H_camera = self.compute_homography_from_pose(
                        pose.rvec,
                        pose.tvec,
                        camera_matrices[i]
                    )
                    
                    if H_ref is not None and i != reference_index:
                        # Compute relative homography to reference
                        H_relative = self.compute_relative_homography(H_ref, H_camera)
                    else:
                        # Use identity for reference camera or if no reference
                        H_relative = np.eye(3) if i == reference_index else H_camera
                    
                    homographies.append(H_relative)
                    valid_indices.append(i)
                    self.cached_homographies[i] = H_relative
                    self.last_valid_homographies[i] = H_relative
                    
                elif i in self.last_valid_homographies:
                    # Use last valid homography if current pose is invalid
                    homographies.append(self.last_valid_homographies[i])
                    valid_indices.append(i)
                    logger.debug(f"Using cached homography for camera {i}")
                else:
                    # No valid homography available
                    homographies.append(np.eye(3))
                    logger.debug(f"No valid homography for camera {i}")
            
            # Warp images
            warped_images = []
            for i, (image, H) in enumerate(zip(images, homographies)):
                if i in valid_indices or np.array_equal(H, np.eye(3)):
                    warped = self.warp_image(image, H)
                    warped_images.append(warped)
                else:
                    # Return original image if no valid homography
                    warped_images.append(image)
            
            # Create combined view
            combined_view = self.create_combined_view(warped_images, blend_mode)
            
            # Add debug visualization if requested
            if data.get('debug_visualization', False):
                combined_view = self.add_debug_overlay(combined_view, poses, valid_indices)
            
            return {
                'warped_images': warped_images,
                'homographies': homographies,
                'combined_view': combined_view,
                'valid_cameras': valid_indices,
                'reference_camera': reference_index
            }
            
        except Exception as e:
            logger.error(f"Error in perspective warp: {e}")
            return {
                'warped_images': images,
                'homographies': [np.eye(3)] * len(images),
                'combined_view': images[0] if images else None,
                'error': str(e)
            }
    
    def add_debug_overlay(
        self,
        image: np.ndarray,
        poses: List[PoseResult],
        valid_indices: List[int]
    ) -> np.ndarray:
        """Add debug information overlay to the image."""
        overlay = image.copy()
        
        # Add text for each valid camera
        y_offset = 30
        for i in valid_indices:
            if i < len(poses) and poses[i].rvec is not None:
                text = f"Cam {i}: Full={poses[i].is_full_board}"
                cv2.putText(overlay, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
        
        # Add grid lines for alignment verification
        h, w = overlay.shape[:2]
        grid_size = 100
        alpha = 0.3
        
        # Vertical lines
        for x in range(0, w, grid_size):
            cv2.line(overlay, (x, 0), (x, h), (255, 255, 0), 1)
        
        # Horizontal lines
        for y in range(0, h, grid_size):
            cv2.line(overlay, (0, y), (w, y), (255, 255, 0), 1)
        
        # Blend with original
        return cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
    
    def reset_cache(self):
        """Reset cached homographies."""
        self.cached_homographies.clear()
        self.last_valid_homographies.clear()
        logger.info("Reset homography cache")