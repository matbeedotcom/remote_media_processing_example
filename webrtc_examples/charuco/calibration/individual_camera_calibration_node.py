"""
Individual camera calibration node that performs full-screen ChAruco calibration 
for each camera independently before multi-camera calibration.
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
from enum import Enum

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '..', 'remote_media_processing'))

from remotemedia.core.node import Node
from ..nodes.charuco_detection_node import CharucoDetectionNode, CharucoConfig, PoseResult
from ..nodes.pose_diversity_selector_node import PoseDiversitySelectorNode, CalibrationFrame

logger = logging.getLogger(__name__)


class CalibrationState(Enum):
    """State of calibration for a camera."""
    NOT_STARTED = "not_started"
    COLLECTING = "collecting"
    CALIBRATING = "calibrating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class IndividualCameraCalibration:
    """Calibration data for a single camera."""
    camera_id: int
    state: CalibrationState = CalibrationState.NOT_STARTED
    camera_matrix: Optional[np.ndarray] = None
    dist_coeffs: Optional[np.ndarray] = None
    image_size: Optional[Tuple[int, int]] = None
    calibration_error: Optional[float] = None
    num_frames_collected: int = 0
    num_frames_used: int = 0
    last_updated: Optional[datetime] = None
    selected_frames: List[CalibrationFrame] = field(default_factory=list)
    coverage_percentage: float = 0.0  # Percentage of screen covered by detections
    full_board_detections: int = 0  # Number of frames with full board detected


@dataclass
class IndividualCalibrationConfig:
    """Configuration for individual camera calibration."""
    num_cameras: int = 4
    charuco_config: CharucoConfig = field(default_factory=CharucoConfig)
    frames_per_camera: int = 15  # More frames for better individual calibration
    min_frames_for_calibration: int = 10
    require_full_screen_coverage: bool = True
    min_coverage_percentage: float = 70.0  # Minimum screen coverage required
    calibration_file_prefix: str = "individual_calibration"
    enable_visualization: bool = True
    max_concurrent_calibrations: int = 2  # How many cameras to calibrate simultaneously


class IndividualCameraCalibrationNode(Node):
    """
    Performs individual full-screen ChAruco calibration for each camera.
    
    This node ensures each camera gets a proper individual calibration with
    full-screen coverage before proceeding to multi-camera calibration.
    
    Input: Dict with 'frames' (List of images), 'timestamp'
    Output: Dict with 'frames', 'calibrations', 'status', 'ready_for_multi_camera'
    """
    
    def __init__(
        self,
        config: Optional[IndividualCalibrationConfig] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name or "IndividualCameraCalibration")
        self.config = config or IndividualCalibrationConfig()
        
        # Initialize detector for each camera
        self.detector = CharucoDetectionNode(config=self.config.charuco_config)
        
        # Initialize diversity selectors for each camera
        self.diversity_selectors = {}
        for i in range(self.config.num_cameras):
            self.diversity_selectors[i] = PoseDiversitySelectorNode(
                max_frames=self.config.frames_per_camera,
                require_full_board=False  # Allow partial detections for better coverage
            )
        
        # Calibration data for each camera
        self.camera_calibrations: Dict[int, IndividualCameraCalibration] = {}
        for i in range(self.config.num_cameras):
            self.camera_calibrations[i] = IndividualCameraCalibration(camera_id=i)
        
        # Track overall progress
        self.current_calibrating_camera: Optional[int] = None
        self.calibration_order: List[int] = list(range(self.config.num_cameras))
        self.all_calibrations_complete = False
        
        # Statistics
        self.frames_processed = 0
        
        logger.info(f"Initialized individual camera calibration for {self.config.num_cameras} cameras")
        logger.info(f"Target: {self.config.frames_per_camera} frames per camera with >{self.config.min_coverage_percentage}% coverage")
    
    def get_active_camera(self) -> Optional[int]:
        """Get the camera that should currently be calibrated."""
        # Check if any camera is currently being calibrated
        for cam_id, cal in self.camera_calibrations.items():
            if cal.state == CalibrationState.COLLECTING:
                return cam_id
        
        # Find next camera that needs calibration
        for cam_id in self.calibration_order:
            cal = self.camera_calibrations[cam_id]
            if cal.state == CalibrationState.NOT_STARTED:
                return cam_id
        
        return None
    
    async def detect_charuco_for_camera(
        self,
        frame: np.ndarray,
        camera_id: int,
        timestamp: Optional[float] = None
    ) -> PoseResult:
        """Detect ChAruco board for a specific camera."""
        cal = self.camera_calibrations.get(camera_id)
        
        # Prepare detection input
        detection_input = {
            'image': frame,
            'camera_id': camera_id,
            'timestamp': timestamp
        }
        
        # Use existing calibration if available
        if cal and cal.camera_matrix is not None:
            detection_input['camera_matrix'] = cal.camera_matrix
            detection_input['dist_coeffs'] = cal.dist_coeffs
        else:
            # Use default intrinsics
            h, w = frame.shape[:2]
            focal_length = max(w, h)
            detection_input['camera_matrix'] = np.array([
                [focal_length, 0, w/2],
                [0, focal_length, h/2],
                [0, 0, 1]
            ], dtype=np.float32)
            detection_input['dist_coeffs'] = np.zeros(5, dtype=np.float32)
        
        return await self.detector.process(detection_input)
    
    def calculate_coverage(self, frames: List[CalibrationFrame]) -> float:
        """Calculate the screen coverage percentage from detected corners."""
        if not frames:
            return 0.0
        
        # Collect all detected corner positions across all frames
        all_corners = []
        image_size = None
        
        for frame in frames:
            for pose in frame.poses:
                if pose.charuco_corners is not None:
                    all_corners.extend(pose.charuco_corners.reshape(-1, 2))
                    
                    # Get image size from frame data
                    if image_size is None and 'images' in frame.frame_data:
                        img = frame.frame_data['images'][0]
                        image_size = (img.shape[1], img.shape[0])
        
        if not all_corners or image_size is None:
            return 0.0
        
        all_corners = np.array(all_corners)
        
        # Calculate bounding box of all detected corners
        min_x, min_y = np.min(all_corners, axis=0)
        max_x, max_y = np.max(all_corners, axis=0)
        
        # Calculate coverage percentage
        detected_area = (max_x - min_x) * (max_y - min_y)
        total_area = image_size[0] * image_size[1]
        coverage = (detected_area / total_area) * 100
        
        return min(coverage, 100.0)
    
    def perform_camera_calibration(self, camera_id: int):
        """Perform calibration for a single camera."""
        cal = self.camera_calibrations[camera_id]
        selector = self.diversity_selectors[camera_id]
        
        logger.info(f"📷 === CALIBRATING CAMERA {camera_id} ===")
        cal.state = CalibrationState.CALIBRATING
        
        selected_frames = selector.selected_frames
        if len(selected_frames) < self.config.min_frames_for_calibration:
            logger.warning(f"Insufficient frames for camera {camera_id}: {len(selected_frames)}")
            cal.state = CalibrationState.FAILED
            return
        
        # Calculate coverage
        coverage = self.calculate_coverage(selected_frames)
        cal.coverage_percentage = coverage
        
        logger.info(f"Screen coverage: {coverage:.1f}%")
        
        if self.config.require_full_screen_coverage and coverage < self.config.min_coverage_percentage:
            logger.warning(f"Insufficient screen coverage for camera {camera_id}: {coverage:.1f}%")
            logger.info(f"Need more diverse poses covering different screen areas")
            cal.state = CalibrationState.COLLECTING  # Continue collecting
            return
        
        # Collect calibration data
        object_points = []
        image_points = []
        image_size = None
        
        for frame in selected_frames:
            if camera_id < len(frame.poses):
                pose = frame.poses[camera_id]
                
                if pose.charuco_corners is not None and pose.charuco_ids is not None:
                    # Count full board detections
                    expected_corners = (self.config.charuco_config.squares_x - 1) * (self.config.charuco_config.squares_y - 1)
                    if len(pose.charuco_corners) == expected_corners:
                        cal.full_board_detections += 1
                    
                    # Get object points
                    obj_pts = []
                    chessboard_corners = self.detector.board.getChessboardCorners()
                    for corner_id in pose.charuco_ids.flatten():
                        if corner_id < len(chessboard_corners):
                            obj_pts.append(chessboard_corners[corner_id])
                    
                    if obj_pts:
                        object_points.append(np.array(obj_pts, dtype=np.float32))
                        image_points.append(pose.charuco_corners)
                        
                        # Get image size
                        if image_size is None and 'images' in frame.frame_data:
                            img = frame.frame_data['images'][camera_id]
                            image_size = (img.shape[1], img.shape[0])
        
        # Perform calibration
        if len(object_points) >= self.config.min_frames_for_calibration and image_size:
            logger.info(f"Calibrating with {len(object_points)} frames")
            logger.info(f"Full board detections: {cal.full_board_detections}/{len(selected_frames)}")
            
            try:
                # Use enhanced calibration flags for better accuracy
                calibration_flags = (
                    cv2.CALIB_RATIONAL_MODEL +        # 8-coefficient distortion model
                    cv2.CALIB_THIN_PRISM_MODEL +      # Thin prism distortion
                    cv2.CALIB_TILTED_MODEL            # Sensor tilt correction
                )
                
                ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                    object_points,
                    image_points,
                    image_size,
                    None,
                    None,
                    flags=calibration_flags,
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
                )
                
                if ret:
                    cal.camera_matrix = camera_matrix
                    cal.dist_coeffs = dist_coeffs
                    cal.image_size = image_size
                    cal.calibration_error = ret
                    cal.num_frames_used = len(object_points)
                    cal.last_updated = datetime.now()
                    cal.state = CalibrationState.COMPLETED
                    
                    logger.info(f"✅ Camera {camera_id} calibrated successfully!")
                    logger.info(f"   Calibration error: {ret:.4f} pixels")
                    logger.info(f"   Focal length: fx={camera_matrix[0,0]:.1f}, fy={camera_matrix[1,1]:.1f}")
                    logger.info(f"   Principal point: cx={camera_matrix[0,2]:.1f}, cy={camera_matrix[1,2]:.1f}")
                    logger.info(f"   Coverage: {coverage:.1f}%")
                    logger.info(f"   Full boards: {cal.full_board_detections}/{len(selected_frames)}")
                    
                    # Save individual calibration
                    self.save_individual_calibration(camera_id)
                else:
                    logger.error(f"Calibration failed for camera {camera_id}")
                    cal.state = CalibrationState.FAILED
                    
            except Exception as e:
                logger.error(f"Error calibrating camera {camera_id}: {e}")
                cal.state = CalibrationState.FAILED
        else:
            logger.warning(f"Insufficient data for camera {camera_id}")
            cal.state = CalibrationState.FAILED
    
    def save_individual_calibration(self, camera_id: int):
        """Save individual camera calibration to file."""
        cal = self.camera_calibrations[camera_id]
        if cal.camera_matrix is None:
            return
        
        filename = f"{self.config.calibration_file_prefix}_camera_{camera_id}.json"
        
        try:
            data = {
                'camera_id': camera_id,
                'camera_matrix': cal.camera_matrix.tolist(),
                'dist_coeffs': cal.dist_coeffs.tolist(),
                'image_size': list(cal.image_size) if cal.image_size else [],
                'calibration_error': cal.calibration_error,
                'num_frames_used': cal.num_frames_used,
                'coverage_percentage': cal.coverage_percentage,
                'full_board_detections': cal.full_board_detections,
                'last_updated': cal.last_updated.isoformat() if cal.last_updated else None
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"💾 Saved calibration for camera {camera_id} to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save calibration for camera {camera_id}: {e}")
    
    def get_combined_calibrations(self) -> Dict[int, Dict[str, Any]]:
        """Get all successful calibrations for multi-camera calibration."""
        calibrations = {}
        
        for cam_id, cal in self.camera_calibrations.items():
            if cal.state == CalibrationState.COMPLETED and cal.camera_matrix is not None:
                calibrations[cam_id] = {
                    'camera_matrix': cal.camera_matrix,
                    'dist_coeffs': cal.dist_coeffs,
                    'image_size': cal.image_size,
                    'calibration_error': cal.calibration_error,
                    'coverage_percentage': cal.coverage_percentage,
                    'full_board_detections': cal.full_board_detections
                }
        
        return calibrations
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process frames for individual camera calibration."""
        try:
            frames = data.get('frames', [])
            timestamp = data.get('timestamp')
            
            if not frames:
                return {'error': 'No frames provided'}
            
            self.frames_processed += 1
            
            # Check if all calibrations are complete
            completed_count = sum(1 for cal in self.camera_calibrations.values() 
                                 if cal.state == CalibrationState.COMPLETED)
            
            if completed_count == self.config.num_cameras:
                self.all_calibrations_complete = True
                
                # Return calibrations for multi-camera node
                return {
                    'frames': frames,
                    'individual_calibrations': self.get_combined_calibrations(),
                    'all_calibrations_complete': True,
                    'status': {
                        cam_id: cal.state.value 
                        for cam_id, cal in self.camera_calibrations.items()
                    },
                    'ready_for_multi_camera': True,
                    'message': f"All {self.config.num_cameras} cameras calibrated individually"
                }
            
            # Get current camera to calibrate
            active_camera = self.get_active_camera()
            
            if active_camera is None:
                # Check for failed calibrations
                failed_count = sum(1 for cal in self.camera_calibrations.values() 
                                  if cal.state == CalibrationState.FAILED)
                
                return {
                    'frames': frames,
                    'individual_calibrations': self.get_combined_calibrations(),
                    'all_calibrations_complete': False,
                    'status': {
                        cam_id: cal.state.value 
                        for cam_id, cal in self.camera_calibrations.items()
                    },
                    'ready_for_multi_camera': completed_count > 0,
                    'message': f"Calibration in progress: {completed_count}/{self.config.num_cameras} complete, {failed_count} failed"
                }
            
            # Start calibration for active camera if needed
            cal = self.camera_calibrations[active_camera]
            if cal.state == CalibrationState.NOT_STARTED:
                cal.state = CalibrationState.COLLECTING
                logger.info(f"🎯 Starting calibration for Camera {active_camera}")
                logger.info(f"   Please show ChAruco board to Camera {active_camera} only")
                logger.info(f"   Move board to cover entire screen area")
            
            # Detect ChAruco for active camera
            if active_camera < len(frames):
                pose = await self.detect_charuco_for_camera(
                    frames[active_camera], 
                    active_camera, 
                    timestamp
                )
                
                # Update diversity selector for this camera
                selector = self.diversity_selectors[active_camera]
                diversity_result = await selector.process({
                    'frame_data': {'images': [frames[active_camera]]},
                    'poses': [pose],
                    'timestamp': timestamp
                })
                
                cal.num_frames_collected = diversity_result.get('num_frames', 0)
                
                # Provide feedback
                if self.frames_processed % 30 == 0:  # Every second
                    if pose.charuco_corners is not None:
                        expected_corners = (self.config.charuco_config.squares_x - 1) * (self.config.charuco_config.squares_y - 1)
                        detected_corners = len(pose.charuco_corners)
                        detection_percentage = (detected_corners / expected_corners) * 100
                        
                        coverage = self.calculate_coverage(selector.selected_frames)
                        
                        logger.info(f"")
                        logger.info(f"📷 CAMERA {active_camera} CALIBRATION STATUS")
                        logger.info(f"   Detection: {detected_corners}/{expected_corners} corners ({detection_percentage:.1f}%)")
                        logger.info(f"   Frames collected: {cal.num_frames_collected}/{self.config.frames_per_camera}")
                        logger.info(f"   Screen coverage: {coverage:.1f}%")
                        
                        if coverage < 30:
                            logger.info(f"   💡 Move board to different screen areas")
                        elif coverage < 60:
                            logger.info(f"   💡 Continue covering more screen area")
                        else:
                            logger.info(f"   ✅ Good coverage! Keep collecting diverse poses")
                        
                        if pose.is_full_board:
                            logger.info(f"   🎉 Full board detected!")
                    else:
                        logger.info(f"⚠️  Camera {active_camera}: No detection - adjust board position")
                
                # Check if ready to calibrate
                if (cal.state == CalibrationState.COLLECTING and 
                    cal.num_frames_collected >= self.config.frames_per_camera):
                    
                    self.perform_camera_calibration(active_camera)
                    
                    # Move to next camera or complete
                    if cal.state == CalibrationState.COMPLETED:
                        next_camera = self.get_active_camera()
                        if next_camera is not None:
                            logger.info(f"")
                            logger.info(f"🎯 Next: Please show board to Camera {next_camera}")
                        else:
                            logger.info(f"")
                            logger.info(f"🎉 All individual calibrations complete!")
            
            # Create visualization if enabled
            visualization = None
            if self.config.enable_visualization:
                visualization = self.create_calibration_visualization(frames, active_camera)
            
            # Return current status
            return {
                'frames': frames,
                'visualization': visualization,
                'active_camera': active_camera,
                'individual_calibrations': self.get_combined_calibrations(),
                'all_calibrations_complete': self.all_calibrations_complete,
                'status': {
                    cam_id: {
                        'state': cal.state.value,
                        'frames_collected': cal.num_frames_collected,
                        'coverage': cal.coverage_percentage,
                        'error': cal.calibration_error
                    }
                    for cam_id, cal in self.camera_calibrations.items()
                },
                'ready_for_multi_camera': completed_count >= 2,  # Need at least 2 cameras
                'message': f"Calibrating Camera {active_camera}: {cal.num_frames_collected}/{self.config.frames_per_camera} frames"
            }
            
        except Exception as e:
            logger.error(f"Error in individual camera calibration: {e}")
            return {
                'error': str(e),
                'frames': frames
            }
    
    def create_calibration_visualization(self, frames: List[np.ndarray], active_camera: Optional[int]) -> np.ndarray:
        """Create a visualization showing calibration progress."""
        if not frames:
            return None
        
        # Create grid layout for all cameras
        rows = 2
        cols = (self.config.num_cameras + 1) // 2
        
        # Resize frames for display
        display_size = (640, 480)
        resized_frames = []
        
        for i, frame in enumerate(frames):
            if frame is not None:
                resized = cv2.resize(frame, display_size)
                
                # Add overlay information
                cal = self.camera_calibrations[i]
                
                # Highlight active camera
                if i == active_camera:
                    cv2.rectangle(resized, (0, 0), (display_size[0]-1, display_size[1]-1), 
                                (0, 255, 0), 3)
                
                # Add status text
                status_color = (0, 255, 0) if cal.state == CalibrationState.COMPLETED else \
                              (0, 255, 255) if i == active_camera else \
                              (128, 128, 128)
                
                status_text = f"Cam {i}: {cal.state.value}"
                cv2.putText(resized, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                if cal.state == CalibrationState.COLLECTING and i == active_camera:
                    progress_text = f"Frames: {cal.num_frames_collected}/{self.config.frames_per_camera}"
                    cv2.putText(resized, progress_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    coverage_text = f"Coverage: {cal.coverage_percentage:.1f}%"
                    cv2.putText(resized, coverage_text, (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                elif cal.state == CalibrationState.COMPLETED:
                    error_text = f"Error: {cal.calibration_error:.3f}px"
                    cv2.putText(resized, error_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                resized_frames.append(resized)
            else:
                # Create blank frame
                blank = np.zeros((display_size[1], display_size[0], 3), dtype=np.uint8)
                resized_frames.append(blank)
        
        # Pad with blank frames if needed
        while len(resized_frames) < rows * cols:
            blank = np.zeros((display_size[1], display_size[0], 3), dtype=np.uint8)
            resized_frames.append(blank)
        
        # Create grid
        grid_rows = []
        for r in range(rows):
            row_frames = resized_frames[r*cols:(r+1)*cols]
            grid_rows.append(np.hstack(row_frames))
        
        visualization = np.vstack(grid_rows)
        
        # Add overall status
        completed = sum(1 for cal in self.camera_calibrations.values() 
                       if cal.state == CalibrationState.COMPLETED)
        overall_text = f"Individual Calibration Progress: {completed}/{self.config.num_cameras} cameras complete"
        cv2.putText(visualization, overall_text, (20, visualization.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return visualization
    
    def reset_calibration(self, camera_id: Optional[int] = None):
        """Reset calibration for specific camera or all cameras."""
        if camera_id is not None:
            # Reset specific camera
            self.camera_calibrations[camera_id] = IndividualCameraCalibration(camera_id=camera_id)
            self.diversity_selectors[camera_id].reset_selection()
            logger.info(f"Reset calibration for camera {camera_id}")
        else:
            # Reset all cameras
            for cam_id in range(self.config.num_cameras):
                self.camera_calibrations[cam_id] = IndividualCameraCalibration(camera_id=cam_id)
                self.diversity_selectors[cam_id].reset_selection()
            
            self.all_calibrations_complete = False
            self.frames_processed = 0
            logger.info("Reset all individual camera calibrations")
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get summary of calibration status."""
        completed = sum(1 for cal in self.camera_calibrations.values() 
                       if cal.state == CalibrationState.COMPLETED)
        failed = sum(1 for cal in self.camera_calibrations.values() 
                    if cal.state == CalibrationState.FAILED)
        collecting = sum(1 for cal in self.camera_calibrations.values() 
                        if cal.state == CalibrationState.COLLECTING)
        
        camera_details = []
        for cam_id, cal in self.camera_calibrations.items():
            details = {
                'camera_id': cam_id,
                'state': cal.state.value,
                'frames_collected': cal.num_frames_collected,
                'frames_used': cal.num_frames_used,
                'coverage': cal.coverage_percentage,
                'full_boards': cal.full_board_detections,
                'error': cal.calibration_error
            }
            camera_details.append(details)
        
        return {
            'total_cameras': self.config.num_cameras,
            'completed': completed,
            'failed': failed,
            'collecting': collecting,
            'all_complete': self.all_calibrations_complete,
            'frames_processed': self.frames_processed,
            'camera_details': camera_details,
            'config': {
                'frames_per_camera': self.config.frames_per_camera,
                'min_coverage': self.config.min_coverage_percentage,
                'require_full_screen': self.config.require_full_screen_coverage
            }
        }