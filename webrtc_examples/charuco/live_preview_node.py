"""
Live preview node that renders camera frames with ChAruco detection overlays.
Creates visual debug interface showing real-time detection results.
"""

from typing import Any, Dict, Optional, List
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
from charuco_detection_node import PoseResult, CharucoConfig

logger = logging.getLogger(__name__)


@dataclass
class LivePreviewConfig:
    """Configuration for live preview visualization."""
    preview_width: int = 320  # Width per camera preview
    preview_height: int = 240  # Height per camera preview
    grid_cols: int = 2  # Number of columns in preview grid
    show_corner_ids: bool = True
    show_pose_info: bool = True
    show_statistics: bool = True
    overlay_alpha: float = 0.8
    text_scale: float = 0.6
    line_thickness: int = 2


class LivePreviewNode(Node):
    """
    Creates live preview visualization of camera frames with ChAruco detection overlays.
    
    Input: Dict with 'frames' (List of images), 'poses' (List[PoseResult]), 
           'calibration_status', 'statistics'
    Output: Dict with 'preview_frame' (combined preview), 'individual_previews' (List)
    """
    
    def __init__(
        self,
        config: Optional[LivePreviewConfig] = None,
        charuco_config: Optional[CharucoConfig] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name or "LivePreview")
        self.config = config or LivePreviewConfig()
        self.charuco_config = charuco_config or CharucoConfig()
        
        # Statistics tracking
        self.frame_count = 0
        self.detection_history = {}  # camera_id -> [detection_counts]
        self.max_history = 30  # Keep 30 frames of history for averaging
        
        logger.info(f"Initialized live preview with {self.config.preview_width}x{self.config.preview_height} per camera")
    
    def create_camera_preview(
        self, 
        frame: np.ndarray, 
        pose: PoseResult, 
        camera_id: int,
        calibration_status: Dict[str, Any]
    ) -> np.ndarray:
        """Create preview visualization for a single camera."""
        
        # Resize frame to preview size
        preview = cv2.resize(frame, (self.config.preview_width, self.config.preview_height))
        
        # Create overlay for detection visualization
        overlay = preview.copy()
        
        # Draw ChAruco detection if available
        if pose.charuco_corners is not None and pose.charuco_ids is not None:
            corners = pose.charuco_corners
            ids = pose.charuco_ids
            
            # Scale corners to preview size
            scale_x = self.config.preview_width / frame.shape[1]
            scale_y = self.config.preview_height / frame.shape[0]
            
            scaled_corners = corners.copy()
            scaled_corners[:, :, 0] *= scale_x
            scaled_corners[:, :, 1] *= scale_y
            
            # Draw detected corners
            cv2.aruco.drawDetectedCornersCharuco(
                overlay, scaled_corners, ids, (0, 255, 0)
            )
            
            # Draw corner IDs if enabled
            if self.config.show_corner_ids:
                for i, corner_id in enumerate(ids.flatten()):
                    corner_pos = scaled_corners[i][0].astype(int)
                    cv2.putText(
                        overlay, str(corner_id),
                        (corner_pos[0] + 5, corner_pos[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1
                    )
        
        # Draw ArUco markers if no ChAruco but ArUco detected
        elif pose.aruco_corners is not None and pose.aruco_ids is not None:
            # Scale ArUco corners
            scale_x = self.config.preview_width / frame.shape[1]
            scale_y = self.config.preview_height / frame.shape[0]
            
            scaled_corners = []
            for corner_set in pose.aruco_corners:
                scaled_set = corner_set.copy()
                scaled_set[:, :, 0] *= scale_x
                scaled_set[:, :, 1] *= scale_y
                scaled_corners.append(scaled_set)
            
            # Draw ArUco markers
            cv2.aruco.drawDetectedMarkers(overlay, scaled_corners, pose.aruco_ids)
        
        # Add camera information overlay
        self._add_camera_info_overlay(overlay, pose, camera_id, calibration_status)
        
        # Blend overlay with original
        result = cv2.addWeighted(
            preview, 1 - self.config.overlay_alpha, 
            overlay, self.config.overlay_alpha, 0
        )
        
        return result
    
    def _add_camera_info_overlay(
        self, 
        image: np.ndarray, 
        pose: PoseResult, 
        camera_id: int,
        calibration_status: Dict[str, Any]
    ):
        """Add information overlay to camera preview."""
        
        # Camera ID and status
        status_color = (0, 255, 0) if pose.charuco_corners is not None else (0, 0, 255)
        cv2.putText(
            image, f"Cam {camera_id}",
            (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 
            self.config.text_scale, status_color, self.config.line_thickness
        )
        
        # Detection counts
        y_pos = 40
        if pose.aruco_ids is not None:
            aruco_count = len(pose.aruco_ids)
            cv2.putText(
                image, f"ArUco: {aruco_count}",
                (5, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                self.config.text_scale * 0.8, (255, 255, 0), 1
            )
            y_pos += 18
        
        if pose.charuco_corners is not None:
            charuco_count = len(pose.charuco_corners)
            expected_corners = (self.charuco_config.squares_x - 1) * (self.charuco_config.squares_y - 1)
            percentage = (charuco_count / expected_corners) * 100
            
            cv2.putText(
                image, f"ChAruco: {charuco_count}/{expected_corners}",
                (5, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                self.config.text_scale * 0.8, (0, 255, 255), 1
            )
            y_pos += 18
            
            cv2.putText(
                image, f"({percentage:.1f}%)",
                (5, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                self.config.text_scale * 0.7, (0, 255, 255), 1
            )
            y_pos += 18
        
        # Pose information
        if self.config.show_pose_info and pose.rvec is not None:
            distance = np.linalg.norm(pose.tvec)
            cv2.putText(
                image, f"Dist: {distance:.2f}m",
                (5, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                self.config.text_scale * 0.7, (255, 0, 255), 1
            )
            y_pos += 15
        
        # Calibration status
        is_calibrated = calibration_status.get('calibrated', False)
        cal_color = (0, 255, 0) if is_calibrated else (255, 255, 0)
        cal_text = "CAL ✓" if is_calibrated else "CAL ✗"
        cv2.putText(
            image, cal_text,
            (5, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
            self.config.text_scale * 0.8, cal_color, 1
        )
    
    def create_combined_preview(
        self, 
        individual_previews: List[np.ndarray],
        calibration_status: Dict[str, Any],
        statistics: Dict[str, Any]
    ) -> np.ndarray:
        """Create combined grid preview of all cameras."""
        
        num_cameras = len(individual_previews)
        if num_cameras == 0:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Calculate grid layout
        cols = min(self.config.grid_cols, num_cameras)
        rows = (num_cameras + cols - 1) // cols  # Ceiling division
        
        # Create grid
        grid_width = cols * self.config.preview_width
        grid_height = rows * self.config.preview_height
        
        # Add space for status bar
        status_bar_height = 80
        combined_height = grid_height + status_bar_height
        
        combined = np.zeros((combined_height, grid_width, 3), dtype=np.uint8)
        
        # Place individual previews in grid
        for i, preview in enumerate(individual_previews):
            row = i // cols
            col = i % cols
            
            y1 = row * self.config.preview_height
            y2 = y1 + self.config.preview_height
            x1 = col * self.config.preview_width
            x2 = x1 + self.config.preview_width
            
            combined[y1:y2, x1:x2] = preview
        
        # Add overall status bar
        self._add_status_bar(combined, calibration_status, statistics, grid_height)
        
        return combined
    
    def _add_status_bar(
        self, 
        image: np.ndarray, 
        calibration_status: Dict[str, Any],
        statistics: Dict[str, Any],
        y_start: int
    ):
        """Add status information bar at bottom of combined preview."""
        
        # Black background for status bar
        image[y_start:, :] = (40, 40, 40)
        
        y_pos = y_start + 25
        
        # Calibration status
        frames_collected = calibration_status.get('frames_collected', 0)
        frames_needed = calibration_status.get('frames_needed', 5)
        is_calibrated = calibration_status.get('calibrated', False)
        
        cal_text = f"Calibration: {'✅ COMPLETE' if is_calibrated else f'⏳ {frames_collected}/{frames_needed} frames'}"
        cv2.putText(
            image, cal_text,
            (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0) if is_calibrated else (255, 255, 0), 2
        )
        
        # Detection statistics
        y_pos += 25
        detection_rate = statistics.get('detection_rate', 0) * 100
        valid_poses = statistics.get('valid_poses_current', 0)
        frames_processed = statistics.get('frames_processed', 0)
        
        stats_text = f"Detection: {detection_rate:.1f}% | Valid poses: {valid_poses} | Frames: {frames_processed}"
        cv2.putText(
            image, stats_text,
            (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 1
        )
        
        # Frame counter and timestamp
        y_pos += 20
        frame_text = f"Frame #{self.frame_count}"
        cv2.putText(
            image, frame_text,
            (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (128, 128, 128), 1
        )
    
    def update_detection_history(self, camera_id: int, has_detection: bool):
        """Update detection history for averaging."""
        if camera_id not in self.detection_history:
            self.detection_history[camera_id] = []
        
        self.detection_history[camera_id].append(1 if has_detection else 0)
        
        # Keep only recent history
        if len(self.detection_history[camera_id]) > self.max_history:
            self.detection_history[camera_id].pop(0)
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process frames and poses to create live preview."""
        try:
            frames = data.get('frames', [])
            poses = data.get('poses', [])
            calibration_status = data.get('calibration_status', {})
            statistics = data.get('statistics', {})
            
            if not frames:
                logger.warning("No frames provided for live preview")
                return {'preview_frame': None, 'individual_previews': []}
            
            self.frame_count += 1
            
            # Create individual camera previews
            individual_previews = []
            
            for i, frame in enumerate(frames):
                pose = poses[i] if i < len(poses) else PoseResult()
                
                # Update detection history
                has_detection = pose.charuco_corners is not None
                self.update_detection_history(i, has_detection)
                
                # Create camera preview
                preview = self.create_camera_preview(
                    frame, pose, i, calibration_status
                )
                individual_previews.append(preview)
            
            # Create combined grid preview
            combined_preview = self.create_combined_preview(
                individual_previews, calibration_status, statistics
            )
            
            # Pass through all important data from previous nodes, adding our preview data
            output = data.copy()  # Start with all input data
            output.update({
                'preview_frame': combined_preview,
                'individual_previews': individual_previews,
                'detection_history': self.detection_history
            })
            return output
            
        except Exception as e:
            logger.error(f"Error in live preview: {e}")
            return {
                'preview_frame': None,
                'individual_previews': [],
                'error': str(e)
            }
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get detection statistics for all cameras."""
        stats = {}
        for camera_id, history in self.detection_history.items():
            if history:
                avg_detection = sum(history) / len(history)
                stats[f'camera_{camera_id}'] = {
                    'detection_rate': avg_detection,
                    'recent_frames': len(history),
                    'current_streak': self._get_current_streak(history)
                }
        return stats
    
    def _get_current_streak(self, history: List[int]) -> int:
        """Get current streak of consecutive detections/non-detections."""
        if not history:
            return 0
        
        current_value = history[-1]
        streak = 1
        
        for i in range(len(history) - 2, -1, -1):
            if history[i] == current_value:
                streak += 1
            else:
                break
        
        return streak if current_value == 1 else -streak  # Positive for detections, negative for non-detections