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
    min_corners_for_homography: int = 8  # Minimum corners for homography computation


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
        
        # No caching - use synchronized frames only
        
        logger.info(f"Initialized perspective warp with output size "
                   f"{self.config.output_width}x{self.config.output_height}")
    
    
    def compute_homography_from_charuco_corners(
        self,
        charuco_corners: np.ndarray,
        charuco_ids: np.ndarray,
        board,
        target_corners: np.ndarray,
        target_ids: np.ndarray
    ) -> np.ndarray:
        """
        Compute homography between two sets of ChAruco corner correspondences.
        This is more robust than using pose estimation for multi-camera alignment.
        """
        try:
            # Find corresponding corners between source and target
            src_points = []
            dst_points = []
            common_ids = []
            
            for i, corner_id in enumerate(charuco_ids.flatten()):
                # Find this corner ID in the target set
                target_indices = np.where(target_ids.flatten() == corner_id)[0]
                if len(target_indices) > 0:
                    target_idx = target_indices[0]
                    src_points.append(charuco_corners[i][0])  # Remove extra dimension
                    dst_points.append(target_corners[target_idx][0])  # Remove extra dimension
                    common_ids.append(corner_id)
            
            # Log corner correspondence details
            src_ids = set(charuco_ids.flatten())
            target_ids_set = set(target_ids.flatten())
            logger.debug(f"Corner correspondence: src_ids={sorted(src_ids)}, target_ids={sorted(target_ids_set)}, common={sorted(common_ids)}")
            
            if len(src_points) >= 6:  # Require at least 6 corner correspondences
                src_points = np.array(src_points, dtype=np.float32)
                dst_points = np.array(dst_points, dtype=np.float32)
                
                logger.debug(f"Computing homography with {len(src_points)} corner correspondences")
                
                # Compute homography with RANSAC for robustness
                H, mask = cv2.findHomography(src_points, dst_points, 
                                           cv2.RANSAC, 
                                           ransacReprojThreshold=3.0)
                
                if H is not None:
                    # Check if we have reasonable inliers
                    inliers = np.sum(mask) if mask is not None else len(src_points)
                    logger.debug(f"Homography computed: {inliers}/{len(src_points)} inliers")
                    
                    if inliers >= 6:  # At least 6 inliers required for robust homography
                        return H
                    else:
                        logger.warning(f"Insufficient inliers for homography: {inliers}/{len(src_points)}")
                        return np.eye(3)
                else:
                    logger.warning("cv2.findHomography returned None")
                    return np.eye(3)
            else:
                logger.warning(f"Insufficient corner correspondences: {len(src_points)} (need >= 6)")
                return np.eye(3)
                
        except Exception as e:
            logger.error(f"Error computing homography from ChAruco corners: {e}")
            return np.eye(3)
    
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
    
    def calculate_combined_canvas_size(
        self,
        images: List[np.ndarray],
        homographies: List[np.ndarray]
    ) -> Tuple[int, int, np.ndarray]:
        """
        Calculate the minimum canvas size needed to contain all warped images.
        Returns (width, height, translation_matrix) where translation_matrix
        shifts coordinates to handle negative values.
        """
        all_corners = []
        
        for img, H in zip(images, homographies):
            h, w = img.shape[:2]
            # Image corners in original coordinates
            corners = np.array([
                [0, 0, 1],
                [w, 0, 1], 
                [w, h, 1],
                [0, h, 1]
            ]).T
            
            # Transform corners to target space
            transformed_corners = H @ corners
            # Convert from homogeneous coordinates
            transformed_corners = transformed_corners[:2] / transformed_corners[2]
            all_corners.extend(transformed_corners.T)
        
        if not all_corners:
            return self.config.output_width, self.config.output_height, np.eye(3)
        
        all_corners = np.array(all_corners)
        
        # Find bounding box
        min_x = np.min(all_corners[:, 0])
        max_x = np.max(all_corners[:, 0])
        min_y = np.min(all_corners[:, 1])
        max_y = np.max(all_corners[:, 1])
        
        # Calculate canvas size with some padding
        padding = 50
        canvas_width = int(max_x - min_x) + 2 * padding
        canvas_height = int(max_y - min_y) + 2 * padding
        
        # Translation matrix to shift negative coordinates
        translation = np.array([
            [1, 0, -min_x + padding],
            [0, 1, -min_y + padding],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return canvas_width, canvas_height, translation

    def warp_image(
        self,
        image: np.ndarray,
        homography: np.ndarray,
        canvas_size: Tuple[int, int] = None
    ) -> np.ndarray:
        """
        Warp image using homography matrix.
        """
        if canvas_size is None:
            canvas_size = (self.config.output_width, self.config.output_height)
            
        warped = cv2.warpPerspective(
            image,
            homography,
            canvas_size,
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
            # Additive blending to preserve all camera data
            combined = np.zeros_like(warped_images[0], dtype=np.float32)
            pixel_count = np.zeros(warped_images[0].shape[:2], dtype=np.float32)
            
            for img in warped_images:
                # Create mask for non-black pixels (actual camera data)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mask = (gray > 5).astype(np.float32)  # Threshold to exclude black/noise
                
                # Add image data where cameras have actual content
                combined += img.astype(np.float32) * mask[..., np.newaxis]
                pixel_count += mask
            
            # Average overlapping areas, preserve unique areas
            pixel_count[pixel_count == 0] = 1  # Avoid division by zero
            combined = (combined / pixel_count[..., np.newaxis]).astype(np.uint8)
        
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
            
            # Use Camera 2 as fixed reference (center camera)
            reference_camera_idx = 2
            reference_pose = poses[reference_camera_idx] if reference_camera_idx < len(poses) else None
            
            # Check if reference camera has sufficient detection
            if (reference_pose and reference_pose.charuco_corners is not None and 
                reference_pose.charuco_ids is not None and 
                len(reference_pose.charuco_corners) >= self.config.min_corners_for_homography):
                
                reference_corner_count = len(reference_pose.charuco_corners)
                logger.info(f"🎯 Using Camera {reference_camera_idx} as fixed reference with {reference_corner_count} corners")
                
                # Log detection status for all cameras
                total_corners = 0
                cameras_with_detection = 0
                for i, pose in enumerate(poses):
                    if (pose.charuco_corners is not None and pose.charuco_ids is not None and 
                        len(pose.charuco_corners) >= self.config.min_corners_for_homography):
                        corner_count = len(pose.charuco_corners)
                        total_corners += corner_count
                        cameras_with_detection += 1
                        logger.info(f"   Camera {i}: {corner_count} corners")
                
                logger.info(f"📊 Total: {cameras_with_detection} cameras with detection, {total_corners} total corners")
            else:
                logger.info("⚠️ Reference camera (Camera 2) has insufficient corner detection")
                reference_camera_idx = None
                reference_pose = None
            
            # Compute homographies for each camera using CURRENT SYNCHRONIZED FRAME
            homographies = []
            valid_indices = []
            computed_homographies = 0
            
            for i, (image, pose) in enumerate(zip(images, poses[:len(images)])):
                H = np.eye(3)  # Default to identity
                
                if i == reference_camera_idx:
                    # Reference camera (Camera 2) gets identity matrix
                    logger.info(f"🎯 Camera {i}: Reference camera, using identity")
                    valid_indices.append(i)
                    
                elif (reference_pose is not None and 
                      pose.charuco_corners is not None and pose.charuco_ids is not None and 
                      len(pose.charuco_corners) >= self.config.min_corners_for_homography):
                    
                    # Each camera computes homography independently to Camera 2
                    H = self.compute_homography_from_charuco_corners(
                        pose.charuco_corners,
                        pose.charuco_ids,
                        None,
                        reference_pose.charuco_corners,
                        reference_pose.charuco_ids
                    )
                    
                    # Check if homography is valid (not identity)
                    if not np.allclose(H, np.eye(3), atol=1e-6):
                        logger.info(f"✅ Camera {i}: Computed independent homography to Camera 2")
                        valid_indices.append(i)
                        computed_homographies += 1
                    else:
                        logger.info(f"⚠️ Camera {i}: Insufficient correspondences with Camera 2")
                        # Use identity as fallback - camera can't be aligned in this frame
                        H = np.eye(3)
                
                elif reference_pose is None:
                    # Reference camera doesn't have detection - can't compute any homographies
                    logger.info(f"❌ Camera {i}: Cannot align (Camera 2 has no detection)")
                
                else:
                    # This camera doesn't have sufficient detection
                    logger.info(f"❌ Camera {i}: No ChAruco detection (insufficient corners < {self.config.min_corners_for_homography})")
                
                homographies.append(H)
            
            # Log homography computation status
            if computed_homographies > 0:
                logger.info(f"🔄 Computed {computed_homographies} new homographies from synchronized frame, {len(valid_indices)} total valid cameras")
            
            # Calculate optimal canvas size to contain all warped images
            canvas_width, canvas_height, translation_matrix = self.calculate_combined_canvas_size(
                images, homographies
            )
            
            logger.info(f"📐 Dynamic canvas size: {canvas_width}x{canvas_height} (vs fixed {self.config.output_width}x{self.config.output_height})")
            
            # Apply translation to homographies to handle negative coordinates
            adjusted_homographies = []
            for H in homographies:
                adjusted_H = translation_matrix @ H
                adjusted_homographies.append(adjusted_H)
            
            # Warp images using dynamic canvas size
            warped_images = []
            for i, (image, H) in enumerate(zip(images, adjusted_homographies)):
                if i in valid_indices or np.array_equal(homographies[i], np.eye(3)):  # Check original H for identity
                    warped = self.warp_image(image, H, (canvas_width, canvas_height))
                    warped_images.append(warped)
                else:
                    # Create blank canvas if no valid homography
                    blank = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
                    warped_images.append(blank)
            
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
                'reference_camera': reference_camera_idx
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
    
    # No caching methods needed for synchronized frame processing