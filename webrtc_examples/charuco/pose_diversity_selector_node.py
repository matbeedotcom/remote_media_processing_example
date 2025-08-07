"""
Pose diversity selector node for selecting optimal calibration frames.
Maintains a set of N most diverse poses for robust calibration.
"""

from typing import Any, Dict, Optional, List, Tuple
import logging
import numpy as np
import cv2
from dataclasses import dataclass, field
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '..', 'remote_media_processing'))

from remotemedia.core.node import Node
from charuco_detection_node import PoseResult

logger = logging.getLogger(__name__)


@dataclass
class CalibrationFrame:
    """Stores a calibration frame with pose information."""
    frame_data: Dict[str, Any]  # Original frame data (images, etc.)
    poses: List[PoseResult]  # Pose results for each camera
    diversity_score: float = 0.0
    timestamp: Optional[float] = None


class PoseDiversitySelectorNode(Node):
    """
    Selects and maintains the most diverse set of calibration frames.
    Uses farthest-point sampling based on pose distance metrics.
    
    Input: Dict with 'frame_data' (original images), 'poses' (List[PoseResult])
    Output: Dict with 'selected_frames' (List[CalibrationFrame]), 'is_updated' (bool)
    """
    
    def __init__(
        self,
        max_frames: int = 10,
        rotation_weight: float = 1.0,
        translation_weight: float = 0.01,
        require_full_board: bool = True,
        name: Optional[str] = None
    ):
        super().__init__(name=name or "PoseDiversitySelector")
        self.max_frames = max_frames
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight
        self.require_full_board = require_full_board
        
        # Storage for selected calibration frames
        self.selected_frames: List[CalibrationFrame] = []
        
        logger.info(f"Initialized pose diversity selector with max_frames={max_frames}")
    
    def compute_pose_distance(self, pose1: PoseResult, pose2: PoseResult) -> float:
        """
        Compute distance between two poses using rotation and translation metrics.
        """
        if pose1.rvec is None or pose2.rvec is None or pose1.tvec is None or pose2.tvec is None:
            return float('inf')
        
        # Convert rotation vectors to matrices
        R1, _ = cv2.Rodrigues(pose1.rvec)
        R2, _ = cv2.Rodrigues(pose2.rvec)
        
        # Compute rotation distance (geodesic distance on SO(3))
        R_diff = R1.T @ R2
        cos_angle = np.clip((np.trace(R_diff) - 1) / 2, -1, 1)
        rotation_distance = np.arccos(cos_angle)
        
        # Compute translation distance
        translation_distance = np.linalg.norm(pose1.tvec - pose2.tvec)
        
        # Weighted combination
        total_distance = (self.rotation_weight * rotation_distance + 
                         self.translation_weight * translation_distance)
        
        return total_distance
    
    def compute_frame_distance(self, frame1: CalibrationFrame, frame2: CalibrationFrame) -> float:
        """
        Compute distance between two calibration frames.
        Uses minimum distance across all camera pairs.
        """
        min_distance = float('inf')
        
        # Compare poses for each camera
        for i, pose1 in enumerate(frame1.poses):
            if i < len(frame2.poses):
                pose2 = frame2.poses[i]
                distance = self.compute_pose_distance(pose1, pose2)
                min_distance = min(min_distance, distance)
        
        return min_distance
    
    def compute_diversity_score(self, candidate: CalibrationFrame) -> float:
        """
        Compute diversity score for a candidate frame.
        Score is the minimum distance to any selected frame.
        """
        if not self.selected_frames:
            return float('inf')
        
        min_distance = float('inf')
        for selected in self.selected_frames:
            distance = self.compute_frame_distance(candidate, selected)
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def should_add_frame(self, candidate: CalibrationFrame) -> bool:
        """
        Determine if a candidate frame should be added to the selection.
        """
        # Check if all cameras have valid poses
        valid_poses = sum(1 for pose in candidate.poses 
                          if pose.rvec is not None and pose.tvec is not None and
                          pose.rvec.size > 0 and pose.tvec.size > 0)
        
        if valid_poses == 0:
            return False
        
        # Check if full board is required
        if self.require_full_board:
            full_boards = sum(1 for pose in candidate.poses if pose.is_full_board)
            if full_boards == 0:
                return False
        
        # Compute diversity score
        candidate.diversity_score = self.compute_diversity_score(candidate)
        
        # Add if we haven't reached max frames
        if len(self.selected_frames) < self.max_frames:
            return True
        
        # Replace least diverse frame if candidate is better
        min_idx = min(range(len(self.selected_frames)),
                     key=lambda i: self.selected_frames[i].diversity_score)
        
        return candidate.diversity_score > self.selected_frames[min_idx].diversity_score
    
    def update_selection(self, candidate: CalibrationFrame) -> bool:
        """
        Update the selection with a new candidate frame.
        Returns True if the selection was updated.
        """
        if not self.should_add_frame(candidate):
            return False
        
        if len(self.selected_frames) < self.max_frames:
            # Add new frame
            self.selected_frames.append(candidate)
            logger.info(f"Added calibration frame {len(self.selected_frames)}/{self.max_frames} "
                       f"with diversity score {candidate.diversity_score:.3f}")
        else:
            # Replace least diverse frame
            min_idx = min(range(len(self.selected_frames)),
                         key=lambda i: self.selected_frames[i].diversity_score)
            old_score = self.selected_frames[min_idx].diversity_score
            self.selected_frames[min_idx] = candidate
            logger.info(f"Replaced calibration frame with diversity score "
                       f"{old_score:.3f} -> {candidate.diversity_score:.3f}")
        
        # Recompute all diversity scores after update
        for frame in self.selected_frames:
            if frame is not candidate:  # Use 'is' instead of '!=' to avoid array comparison
                frame.diversity_score = self.compute_diversity_score(frame)
        
        return True
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process new frame data and update selection if diverse enough."""
        try:
            frame_data = data.get('frame_data', {})
            poses = data.get('poses', [])
            timestamp = data.get('timestamp')
            
            # Create calibration frame
            candidate = CalibrationFrame(
                frame_data=frame_data,
                poses=poses,
                timestamp=timestamp
            )
            
            # Update selection
            is_updated = self.update_selection(candidate)
            
            # Log current selection status
            if is_updated:
                valid_count = sum(1 for f in self.selected_frames 
                                 for p in f.poses if p.rvec is not None and 
                                 hasattr(p.rvec, 'size') and p.rvec.size > 0)
                logger.info(f"Selection updated: {len(self.selected_frames)} frames, "
                           f"{valid_count} valid poses")
            
            return {
                'selected_frames': self.selected_frames.copy(),
                'is_updated': is_updated,
                'num_frames': len(self.selected_frames),
                'diversity_scores': [f.diversity_score for f in self.selected_frames]
            }
            
        except Exception as e:
            logger.error(f"Error in pose diversity selection: {e}")
            return {
                'selected_frames': self.selected_frames.copy(),
                'is_updated': False,
                'error': str(e)
            }
    
    def get_calibration_data(self) -> Dict[str, List]:
        """
        Extract calibration data from selected frames.
        Returns object points and image points for each camera.
        """
        calibration_data = {}
        
        for camera_idx in range(max(len(f.poses) for f in self.selected_frames)):
            object_points = []
            image_points = []
            
            for frame in self.selected_frames:
                if camera_idx < len(frame.poses):
                    pose = frame.poses[camera_idx]
                    if pose.charuco_corners is not None and pose.charuco_ids is not None:
                        # Get 3D object points for detected corners
                        obj_pts = []
                        for corner_id in pose.charuco_ids.flatten():
                            # Calculate 3D position based on board layout
                            # This is simplified - actual implementation would use board.chessboardCorners
                            obj_pts.append([0, 0, 0])  # Placeholder
                        
                        if obj_pts:
                            object_points.append(np.array(obj_pts, dtype=np.float32))
                            image_points.append(pose.charuco_corners)
            
            if object_points and image_points:
                calibration_data[f'camera_{camera_idx}'] = {
                    'object_points': object_points,
                    'image_points': image_points
                }
        
        return calibration_data
    
    def reset_selection(self):
        """Reset the selected frames."""
        self.selected_frames.clear()
        logger.info("Reset calibration frame selection")