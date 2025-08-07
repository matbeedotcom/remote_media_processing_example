"""
Multi-camera calibration node that orchestrates ChAruco detection, pose diversity selection,
and perspective warping for synchronized camera arrays.
"""

from typing import Any, Dict, Optional, List, Tuple
import logging
import numpy as np
import cv2
from dataclasses import dataclass, field
import asyncio
import sys
import os
import json
from datetime import datetime

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '..', 'remote_media_processing'))

from remotemedia.core.node import Node
from charuco_detection_node import CharucoDetectionNode, CharucoConfig, PoseResult
from pose_diversity_selector_node import PoseDiversitySelectorNode, CalibrationFrame
from perspective_warp_node import PerspectiveWarpNode, WarpConfig

logger = logging.getLogger(__name__)


@dataclass
class CameraCalibrationData:
    """Stores calibration data for a single camera."""
    camera_id: int
    camera_matrix: Optional[np.ndarray] = None
    dist_coeffs: Optional[np.ndarray] = None
    image_size: Optional[Tuple[int, int]] = None
    calibration_error: Optional[float] = None
    num_frames_used: int = 0
    last_updated: Optional[datetime] = None


@dataclass
class MultiCameraConfig:
    """Configuration for multi-camera calibration."""
    num_cameras: int = 4
    charuco_config: CharucoConfig = field(default_factory=CharucoConfig)
    warp_config: WarpConfig = field(default_factory=WarpConfig)
    max_calibration_frames: int = 10
    min_frames_for_calibration: int = 5
    auto_calibrate: bool = True
    calibration_file: Optional[str] = None
    enable_live_preview: bool = True


class MultiCameraCalibrationNode(Node):
    """
    Orchestrates multi-camera calibration and perspective warping.
    
    This node handles:
    1. Synchronized frame processing from multiple cameras
    2. ChAruco detection for each camera
    3. Pose diversity selection for optimal calibration frames
    4. Camera calibration when sufficient frames are collected
    5. Real-time perspective warping to align all cameras
    
    Input: Dict with 'frames' (List of images), 'timestamp', optional 'camera_calibrations'
    Output: Dict with 'warped_frames', 'combined_view', 'calibration_status', 'poses'
    """
    
    def __init__(
        self,
        config: Optional[MultiCameraConfig] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name or "MultiCameraCalibration")
        self.config = config or MultiCameraConfig()
        
        # Initialize sub-nodes
        self.detector = CharucoDetectionNode(config=self.config.charuco_config)
        self.diversity_selector = PoseDiversitySelectorNode(
            max_frames=self.config.max_calibration_frames,
            require_full_board=True
        )
        self.warp_node = PerspectiveWarpNode(config=self.config.warp_config)
        
        # Camera calibration data
        self.camera_calibrations: Dict[int, CameraCalibrationData] = {}
        for i in range(self.config.num_cameras):
            self.camera_calibrations[i] = CameraCalibrationData(camera_id=i)
        
        # Load existing calibration if available
        if self.config.calibration_file and os.path.exists(self.config.calibration_file):
            self.load_calibration(self.config.calibration_file)
        
        # Statistics
        self.frames_processed = 0
        self.successful_detections = 0
        self.calibration_performed = False
        
        logger.info(f"Initialized multi-camera calibration for {self.config.num_cameras} cameras")
    
    def load_calibration(self, filepath: str):
        """Load camera calibration from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            for cam_id_str, cam_data in data.items():
                cam_id = int(cam_id_str)
                if cam_id in self.camera_calibrations:
                    cal = self.camera_calibrations[cam_id]
                    cal.camera_matrix = np.array(cam_data['camera_matrix'])
                    cal.dist_coeffs = np.array(cam_data['dist_coeffs'])
                    cal.image_size = tuple(cam_data.get('image_size', []))
                    cal.calibration_error = cam_data.get('calibration_error')
                    cal.num_frames_used = cam_data.get('num_frames_used', 0)
            
            logger.info(f"Loaded calibration for {len(data)} cameras from {filepath}")
            self.calibration_performed = True
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
    
    def save_calibration(self, filepath: str):
        """Save camera calibration to file."""
        try:
            data = {}
            for cam_id, cal in self.camera_calibrations.items():
                if cal.camera_matrix is not None:
                    data[str(cam_id)] = {
                        'camera_matrix': cal.camera_matrix.tolist(),
                        'dist_coeffs': cal.dist_coeffs.tolist(),
                        'image_size': list(cal.image_size) if cal.image_size else [],
                        'calibration_error': cal.calibration_error,
                        'num_frames_used': cal.num_frames_used,
                        'last_updated': cal.last_updated.isoformat() if cal.last_updated else None
                    }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved calibration for {len(data)} cameras to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
    
    async def detect_charuco_parallel(
        self,
        frames: List[np.ndarray],
        timestamp: Optional[float] = None
    ) -> List[PoseResult]:
        """Detect ChAruco boards in multiple frames in parallel."""
        tasks = []
        
        for i, frame in enumerate(frames):
            cal = self.camera_calibrations.get(i)
            
            # Prepare detection input
            detection_input = {
                'image': frame,
                'camera_id': i,
                'timestamp': timestamp
            }
            
            # Add calibration if available
            if cal and cal.camera_matrix is not None:
                detection_input['camera_matrix'] = cal.camera_matrix
                detection_input['dist_coeffs'] = cal.dist_coeffs
            else:
                # Use default intrinsics based on image size
                h, w = frame.shape[:2]
                focal_length = max(w, h)
                detection_input['camera_matrix'] = np.array([
                    [focal_length, 0, w/2],
                    [0, focal_length, h/2],
                    [0, 0, 1]
                ], dtype=np.float32)
                detection_input['dist_coeffs'] = np.zeros(5, dtype=np.float32)
            
            # Create detection task
            task = self.detector.process(detection_input)
            tasks.append(task)
        
        # Run all detections in parallel
        poses = await asyncio.gather(*tasks)
        return poses
    
    def perform_camera_calibration(self, selected_frames: List[CalibrationFrame]):
        """
        Perform camera calibration using selected diverse frames.
        """
        logger.info(f"Performing calibration with {len(selected_frames)} frames")
        
        for cam_id in range(self.config.num_cameras):
            object_points = []
            image_points = []
            image_size = None
            
            # Collect calibration data for this camera
            for frame in selected_frames:
                if cam_id < len(frame.poses):
                    pose = frame.poses[cam_id]
                    
                    if pose.charuco_corners is not None and pose.charuco_ids is not None:
                        # Get object points from board
                        obj_pts = []
                        chessboard_corners = self.detector.board.getChessboardCorners()
                        for corner_id in pose.charuco_ids.flatten():
                            # Get 3D coordinates from ChAruco board
                            if corner_id < len(chessboard_corners):
                                corner_3d = chessboard_corners[corner_id]
                                obj_pts.append(corner_3d)
                        
                        if obj_pts:
                            object_points.append(np.array(obj_pts, dtype=np.float32))
                            image_points.append(pose.charuco_corners)
                            
                            # Get image size from frame data
                            if 'images' in frame.frame_data and cam_id < len(frame.frame_data['images']):
                                img = frame.frame_data['images'][cam_id]
                                image_size = (img.shape[1], img.shape[0])
            
            # Calibrate if we have enough data
            if len(object_points) >= self.config.min_frames_for_calibration and image_size:
                logger.info(f"Calibrating camera {cam_id} with {len(object_points)} frames")
                
                try:
                    # Perform calibration
                    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                        object_points,
                        image_points,
                        image_size,
                        None,
                        None
                    )
                    
                    if ret:
                        cal = self.camera_calibrations[cam_id]
                        cal.camera_matrix = camera_matrix
                        cal.dist_coeffs = dist_coeffs
                        cal.image_size = image_size
                        cal.calibration_error = ret
                        cal.num_frames_used = len(object_points)
                        cal.last_updated = datetime.now()
                        
                        logger.info(f"Camera {cam_id} calibrated: error={ret:.3f}, "
                                   f"focal=[{camera_matrix[0,0]:.1f}, {camera_matrix[1,1]:.1f}]")
                    else:
                        logger.warning(f"Calibration failed for camera {cam_id}")
                        
                except Exception as e:
                    logger.error(f"Error calibrating camera {cam_id}: {e}")
            else:
                logger.debug(f"Insufficient data for camera {cam_id}: "
                           f"{len(object_points)} frames")
        
        self.calibration_performed = True
        
        # Save calibration if configured
        if self.config.calibration_file:
            self.save_calibration(self.config.calibration_file)
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process synchronized frames from multiple cameras."""
        try:
            frames = data.get('frames', [])
            timestamp = data.get('timestamp')
            force_calibration = data.get('force_calibration', False)
            
            if not frames:
                logger.warning("No frames provided")
                return {'error': 'No frames provided'}
            
            self.frames_processed += 1
            
            # Detect ChAruco in all frames
            poses = await self.detect_charuco_parallel(frames, timestamp)
            
            # Count successful detections
            valid_poses = sum(1 for p in poses if p.rvec is not None and hasattr(p.rvec, 'size') and p.rvec.size > 0)
            if valid_poses > 0:
                self.successful_detections += 1
            
            # Update pose diversity selection
            diversity_result = await self.diversity_selector.process({
                'frame_data': {'images': frames},
                'poses': poses,
                'timestamp': timestamp
            })
            
            # Handle diversity selection errors
            if 'error' in diversity_result:
                logger.warning(f"Diversity selection error: {diversity_result['error']}")
                diversity_result = {
                    'selected_frames': [],
                    'is_updated': False,
                    'num_frames': 0,
                    'diversity_scores': []
                }
            
            # Perform calibration if needed
            if (self.config.auto_calibrate and 
                not self.calibration_performed and
                diversity_result['num_frames'] >= self.config.min_frames_for_calibration) or force_calibration:
                
                self.perform_camera_calibration(diversity_result['selected_frames'])
            
            # Prepare camera matrices for warping
            camera_matrices = []
            for i in range(len(frames)):
                cal = self.camera_calibrations.get(i)
                if cal and cal.camera_matrix is not None:
                    camera_matrices.append(cal.camera_matrix)
                else:
                    # Default matrix
                    h, w = frames[i].shape[:2]
                    focal = max(w, h)
                    K = np.array([[focal, 0, w/2],
                                 [0, focal, h/2],
                                 [0, 0, 1]], dtype=np.float32)
                    camera_matrices.append(K)
            
            # Perform perspective warping
            warp_result = await self.warp_node.process({
                'images': frames,
                'poses': poses,
                'camera_matrices': camera_matrices,
                'blend_mode': data.get('blend_mode', 'overlay'),
                'debug_visualization': data.get('debug_visualization', False)
            })
            
            # Prepare output
            output = {
                'warped_frames': warp_result['warped_images'],
                'combined_view': warp_result['combined_view'],
                'poses': poses,
                'homographies': warp_result['homographies'],
                'valid_cameras': warp_result['valid_cameras'],
                'calibration_status': {
                    'calibrated': self.calibration_performed,
                    'frames_collected': diversity_result['num_frames'],
                    'frames_needed': self.config.min_frames_for_calibration,
                    'diversity_scores': diversity_result.get('diversity_scores', []),
                    'cameras_calibrated': sum(1 for c in self.camera_calibrations.values() 
                                            if c.camera_matrix is not None)
                },
                'statistics': {
                    'frames_processed': self.frames_processed,
                    'successful_detections': self.successful_detections,
                    'detection_rate': self.successful_detections / max(1, self.frames_processed),
                    'valid_poses_current': valid_poses
                }
            }
            
            # Add individual camera calibration info if requested
            if data.get('include_calibration_details', False):
                output['camera_calibrations'] = {
                    i: {
                        'calibrated': cal.camera_matrix is not None,
                        'error': cal.calibration_error,
                        'num_frames': cal.num_frames_used,
                        'last_updated': cal.last_updated.isoformat() if cal.last_updated else None
                    }
                    for i, cal in self.camera_calibrations.items()
                }
            
            return output
            
        except Exception as e:
            logger.error(f"Error in multi-camera calibration: {e}")
            return {
                'error': str(e),
                'warped_frames': frames,
                'combined_view': frames[0] if frames else None
            }
    
    def reset_calibration(self):
        """Reset all calibration data."""
        for cal in self.camera_calibrations.values():
            cal.camera_matrix = None
            cal.dist_coeffs = None
            cal.calibration_error = None
            cal.num_frames_used = 0
            cal.last_updated = None
        
        self.diversity_selector.reset_selection()
        self.warp_node.reset_cache()
        self.calibration_performed = False
        self.frames_processed = 0
        self.successful_detections = 0
        
        logger.info("Reset all calibration data")
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get summary of current calibration status."""
        return {
            'num_cameras': self.config.num_cameras,
            'calibrated_cameras': sum(1 for c in self.camera_calibrations.values() 
                                     if c.camera_matrix is not None),
            'calibration_performed': self.calibration_performed,
            'frames_in_selection': len(self.diversity_selector.selected_frames),
            'max_frames': self.config.max_calibration_frames,
            'min_frames_required': self.config.min_frames_for_calibration,
            'board_config': {
                'squares': f"{self.config.charuco_config.squares_x}x{self.config.charuco_config.squares_y}",
                'square_length': self.config.charuco_config.square_length,
                'marker_length': self.config.charuco_config.marker_length,
                'dictionary': self.config.charuco_config.dictionary
            }
        }