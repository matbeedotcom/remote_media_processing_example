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
import itertools

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '..', 'remote_media_processing'))

from remotemedia.core.node import Node
from charuco_detection_node import CharucoDetectionNode, CharucoConfig, PoseResult
from pose_diversity_selector_node import PoseDiversitySelectorNode, CalibrationFrame
from perspective_warp_node import PerspectiveWarpNode, WarpConfig
from sensor_config import SensorSpecifications, SensorDatabase, CameraSystemConfig

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
class StereoPairCalibration:
    """Stores stereo calibration data between two cameras."""
    camera1_id: int
    camera2_id: int
    rotation_matrix: Optional[np.ndarray] = None      # R: 3x3 rotation from cam1 to cam2
    translation_vector: Optional[np.ndarray] = None  # T: 3x1 translation from cam1 to cam2
    essential_matrix: Optional[np.ndarray] = None    # E: 3x3 essential matrix
    fundamental_matrix: Optional[np.ndarray] = None  # F: 3x3 fundamental matrix
    stereo_error: Optional[float] = None             # RMS reprojection error
    baseline_distance: Optional[float] = None        # Distance between camera centers (mm)
    convergence_angle: Optional[float] = None        # Angle between optical axes (degrees)
    last_updated: Optional[datetime] = None


@dataclass
class CameraPose:
    """6DOF camera pose relative to reference camera."""
    camera_id: int
    rotation_matrix: Optional[np.ndarray] = None     # 3x3 rotation relative to reference
    translation_vector: Optional[np.ndarray] = None # 3x1 translation relative to reference  
    baseline_distance: Optional[float] = None       # Distance from reference camera (mm)
    pose_error: Optional[float] = None              # Estimated pose accuracy
    reference_camera_id: int = 0                    # Reference camera (usually 0)


@dataclass
class MultiCameraConfig:
    """Configuration for multi-camera calibration."""
    num_cameras: int = 4
    charuco_config: CharucoConfig = field(default_factory=CharucoConfig)
    warp_config: WarpConfig = field(default_factory=WarpConfig)
    max_calibration_frames: int = 20  # Increased for better accuracy
    min_frames_for_calibration: int = 10  # More frames for sub-pixel accuracy
    auto_calibrate: bool = True
    calibration_file: Optional[str] = None
    enable_live_preview: bool = True
    force_fresh_calibration: bool = False  # Force new calibration even if file exists
    # Stereo calibration options
    enable_stereo_calibration: bool = True   # Perform stereo calibration between camera pairs
    stereo_calibration_file: Optional[str] = "stereo_calibration.json"  # File to save stereo data
    reference_camera_id: int = 0             # Reference camera for multi-camera poses
    # Sensor configuration
    sensor_name: Optional[str] = "OV9281"    # Default to user's OV9281 sensor
    sensor_config_file: Optional[str] = "sensor_database.json"  # Sensor database file
    camera_system_config_file: Optional[str] = "camera_system_config.json"  # System config
    focal_length_mm: Optional[float] = 2.8   # Default to user's 2.8mm lens focal length


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
            require_full_board=False  # Allow partial board detections for practical calibration
        )
        self.warp_node = PerspectiveWarpNode(config=self.config.warp_config)
        
        # Camera calibration data
        self.camera_calibrations: Dict[int, CameraCalibrationData] = {}
        for i in range(self.config.num_cameras):
            self.camera_calibrations[i] = CameraCalibrationData(camera_id=i)
            
        # Stereo calibration data
        self.stereo_calibrations: Dict[Tuple[int, int], StereoPairCalibration] = {}
        self.camera_poses: Dict[int, CameraPose] = {}
        self.stereo_calibration_performed = False
        
        # Sensor configuration
        self.sensor_database = SensorDatabase()
        self.sensor_specs: Optional[SensorSpecifications] = None
        self.camera_system: Optional[CameraSystemConfig] = None
        self._initialize_sensor_config()
        
        # Load existing calibration if available and not forcing fresh calibration
        if (self.config.calibration_file and 
            os.path.exists(self.config.calibration_file) and 
            not self.config.force_fresh_calibration):
            self.load_calibration(self.config.calibration_file)
        elif self.config.force_fresh_calibration:
            logger.info("🔄 Force fresh calibration enabled - will ignore existing calibration data")
        else:
            logger.info("📋 No existing calibration file found - will perform fresh calibration")
        
        # Statistics
        self.frames_processed = 0
        self.successful_detections = 0
        self.calibration_performed = False
        self.homographies_computed = False
        self.pipeline_should_stop = False
        
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
            
            logger.info(f"✅ Saved intrinsic calibration for {len(data)} cameras to {filepath}")
            for cam_id, cal_data in data.items():
                error = cal_data.get('calibration_error', 0)
                frames = cal_data.get('num_frames_used', 0)
                logger.info(f"   📷 Camera {cam_id}: error={error:.4f}, frames={frames}")
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
    
    def perform_stereo_calibration(self, selected_frames: List[CalibrationFrame]):
        """
        Perform stereo calibration between all camera pairs to determine relative poses.
        """
        logger.info(f"🔗 Performing stereo calibration with {len(selected_frames)} frames")
        
        # Get all camera pairs for stereo calibration
        camera_pairs = list(itertools.combinations(range(self.config.num_cameras), 2))
        logger.info(f"📊 Calibrating {len(camera_pairs)} camera pairs")
        
        for cam1_id, cam2_id in camera_pairs:
            # Check if both cameras are calibrated
            cam1 = self.camera_calibrations.get(cam1_id)
            cam2 = self.camera_calibrations.get(cam2_id)
            
            if (not cam1 or cam1.camera_matrix is None or 
                not cam2 or cam2.camera_matrix is None):
                logger.warning(f"⚠️ Skipping pair ({cam1_id}, {cam2_id}) - missing intrinsics")
                continue
                
            logger.info(f"🔗 Calibrating stereo pair: Camera {cam1_id} ↔ Camera {cam2_id}")
            
            # Collect matching ChAruco points for this camera pair
            object_points = []
            image_points1 = []
            image_points2 = []
            
            for frame in selected_frames:
                if (cam1_id < len(frame.poses) and cam2_id < len(frame.poses)):
                    pose1 = frame.poses[cam1_id]
                    pose2 = frame.poses[cam2_id]
                    
                    if (pose1.charuco_corners is not None and pose1.charuco_ids is not None and
                        pose2.charuco_corners is not None and pose2.charuco_ids is not None):
                        
                        # Find common ChAruco IDs between both cameras
                        common_ids = set(pose1.charuco_ids.flatten()) & set(pose2.charuco_ids.flatten())
                        
                        if len(common_ids) >= 4:  # Need at least 4 common points
                            # Extract matching points
                            obj_pts = []
                            img_pts1 = []
                            img_pts2 = []
                            
                            chessboard_corners = self.detector.board.getChessboardCorners()
                            
                            for corner_id in common_ids:
                                if corner_id < len(chessboard_corners):
                                    # Find indices in both poses
                                    idx1 = np.where(pose1.charuco_ids.flatten() == corner_id)[0]
                                    idx2 = np.where(pose2.charuco_ids.flatten() == corner_id)[0]
                                    
                                    if len(idx1) > 0 and len(idx2) > 0:
                                        obj_pts.append(chessboard_corners[corner_id])
                                        img_pts1.append(pose1.charuco_corners[idx1[0]])
                                        img_pts2.append(pose2.charuco_corners[idx2[0]])
                            
                            if len(obj_pts) >= 4:
                                object_points.append(np.array(obj_pts, dtype=np.float32))
                                image_points1.append(np.array(img_pts1, dtype=np.float32))
                                image_points2.append(np.array(img_pts2, dtype=np.float32))
            
            # Perform stereo calibration if we have enough data
            if len(object_points) >= 3:  # Need at least 3 frames
                try:
                    logger.info(f"   📐 Using {len(object_points)} frames with {sum(len(pts) for pts in object_points)} total points")
                    
                    # Perform stereo calibration
                    stereo_error, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
                        object_points,
                        image_points1,
                        image_points2,
                        cam1.camera_matrix,
                        cam1.dist_coeffs,
                        cam2.camera_matrix,
                        cam2.dist_coeffs,
                        cam1.image_size,
                        flags=cv2.CALIB_FIX_INTRINSIC  # Keep intrinsics fixed
                    )
                    
                    # Calculate baseline distance and convergence angle
                    baseline_distance = np.linalg.norm(T) * 1000  # Convert to mm if T is in meters
                    
                    # Convergence angle (angle between optical axes)
                    optical_axis1 = np.array([0, 0, 1])  # Z-axis in camera 1 frame
                    optical_axis2 = R @ optical_axis1    # Z-axis in camera 2 frame  
                    convergence_angle = np.degrees(np.arccos(np.clip(np.dot(optical_axis1, optical_axis2), -1, 1)))
                    
                    # Store stereo calibration
                    pair_key = (cam1_id, cam2_id)
                    self.stereo_calibrations[pair_key] = StereoPairCalibration(
                        camera1_id=cam1_id,
                        camera2_id=cam2_id,
                        rotation_matrix=R,
                        translation_vector=T,
                        essential_matrix=E,
                        fundamental_matrix=F,
                        stereo_error=stereo_error,
                        baseline_distance=baseline_distance,
                        convergence_angle=convergence_angle,
                        last_updated=datetime.now()
                    )
                    
                    logger.info(f"   ✅ Stereo calibration successful!")
                    logger.info(f"   📊 Stereo error: {stereo_error:.4f} pixels")
                    logger.info(f"   📏 Baseline distance: {baseline_distance:.2f} mm")  
                    logger.info(f"   🔄 Convergence angle: {convergence_angle:.2f}°")
                    
                except Exception as e:
                    logger.error(f"   ❌ Stereo calibration failed for pair ({cam1_id}, {cam2_id}): {e}")
            else:
                logger.warning(f"   ⚠️ Insufficient data for pair ({cam1_id}, {cam2_id}): {len(object_points)} frames")
        
        # Compute multi-camera poses relative to reference camera
        self.compute_multi_camera_poses()
        
        # Mark stereo calibration as complete
        self.stereo_calibration_performed = True
        
        # Save stereo calibration data
        if self.config.stereo_calibration_file:
            self.save_stereo_calibration(self.config.stereo_calibration_file)
            
        logger.info(f"🎉 === STEREO CALIBRATION COMPLETE ===")
        successful_pairs = len([s for s in self.stereo_calibrations.values() if s.rotation_matrix is not None])
        logger.info(f"🔗 Successfully calibrated {successful_pairs}/{len(camera_pairs)} stereo pairs")
    
    def compute_multi_camera_poses(self):
        """Compute poses of all cameras relative to reference camera."""
        ref_id = self.config.reference_camera_id
        logger.info(f"🎯 Computing camera poses relative to reference camera {ref_id}")
        
        # Reference camera has identity pose
        self.camera_poses[ref_id] = CameraPose(
            camera_id=ref_id,
            rotation_matrix=np.eye(3),
            translation_vector=np.zeros((3, 1)),
            baseline_distance=0.0,
            pose_error=0.0,
            reference_camera_id=ref_id
        )
        
        # For other cameras, find direct stereo pair with reference camera
        for cam_id in range(self.config.num_cameras):
            if cam_id == ref_id:
                continue
                
            # Look for direct stereo calibration with reference camera
            pair_key1 = (ref_id, cam_id)
            pair_key2 = (cam_id, ref_id)
            
            stereo_data = None
            inverse_transform = False
            
            if pair_key1 in self.stereo_calibrations:
                stereo_data = self.stereo_calibrations[pair_key1]
                inverse_transform = False
            elif pair_key2 in self.stereo_calibrations:
                stereo_data = self.stereo_calibrations[pair_key2] 
                inverse_transform = True
                
            if stereo_data and stereo_data.rotation_matrix is not None:
                if inverse_transform:
                    # Transform is from cam_id to ref_id, so invert it
                    R = stereo_data.rotation_matrix.T
                    T = -R @ stereo_data.translation_vector
                else:
                    # Transform is from ref_id to cam_id
                    R = stereo_data.rotation_matrix
                    T = stereo_data.translation_vector
                
                baseline_distance = np.linalg.norm(T) * 1000  # Convert to mm
                
                self.camera_poses[cam_id] = CameraPose(
                    camera_id=cam_id,
                    rotation_matrix=R,
                    translation_vector=T,
                    baseline_distance=baseline_distance,
                    pose_error=stereo_data.stereo_error,
                    reference_camera_id=ref_id
                )
                
                logger.info(f"   📷 Camera {cam_id}: baseline={baseline_distance:.2f}mm, error={stereo_data.stereo_error:.4f}px")
            else:
                logger.warning(f"   ⚠️ No direct stereo calibration found for camera {cam_id}")
    
    def save_stereo_calibration(self, filepath: str):
        """Save stereo calibration data to file."""
        try:
            data = {
                'stereo_pairs': {},
                'camera_poses': {},
                'reference_camera_id': self.config.reference_camera_id,
                'calibration_timestamp': datetime.now().isoformat()
            }
            
            # Save stereo pair data
            for (cam1_id, cam2_id), stereo_data in self.stereo_calibrations.items():
                pair_key = f"{cam1_id}_{cam2_id}"
                if stereo_data.rotation_matrix is not None:
                    data['stereo_pairs'][pair_key] = {
                        'camera1_id': cam1_id,
                        'camera2_id': cam2_id,
                        'rotation_matrix': stereo_data.rotation_matrix.tolist(),
                        'translation_vector': stereo_data.translation_vector.tolist(),
                        'essential_matrix': stereo_data.essential_matrix.tolist(),
                        'fundamental_matrix': stereo_data.fundamental_matrix.tolist(),
                        'stereo_error': stereo_data.stereo_error,
                        'baseline_distance': stereo_data.baseline_distance,
                        'convergence_angle': stereo_data.convergence_angle,
                        'last_updated': stereo_data.last_updated.isoformat()
                    }
            
            # Save camera poses
            for cam_id, pose_data in self.camera_poses.items():
                if pose_data.rotation_matrix is not None:
                    data['camera_poses'][str(cam_id)] = {
                        'camera_id': cam_id,
                        'rotation_matrix': pose_data.rotation_matrix.tolist(),
                        'translation_vector': pose_data.translation_vector.tolist(),
                        'baseline_distance': pose_data.baseline_distance,
                        'pose_error': pose_data.pose_error,
                        'reference_camera_id': pose_data.reference_camera_id
                    }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"✅ Saved stereo calibration for {len(data['stereo_pairs'])} pairs to {filepath}")
            logger.info(f"✅ Saved {len(data['camera_poses'])} camera poses to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save stereo calibration: {e}")
    
    def _initialize_sensor_config(self):
        """Initialize sensor configuration and load sensor database."""
        try:
            # Load sensor database from file if it exists
            if self.config.sensor_config_file and os.path.exists(self.config.sensor_config_file):
                self.sensor_database.load_from_file(self.config.sensor_config_file)
                logger.info(f"📷 Loaded sensor database from {self.config.sensor_config_file}")
            
            # Get sensor specifications
            if self.config.sensor_name:
                self.sensor_specs = self.sensor_database.get_sensor(self.config.sensor_name)
                if self.sensor_specs:
                    logger.info(f"📷 Using sensor: {self.sensor_specs.name}")
                    logger.info(f"   Sensor size: {self.sensor_specs.sensor_width_mm:.2f}x{self.sensor_specs.sensor_height_mm:.2f}mm")
                    logger.info(f"   Pixel pitch: {self.sensor_specs.pixel_pitch_um:.2f}μm")
                    logger.info(f"   Resolution: {self.sensor_specs.resolution_width}x{self.sensor_specs.resolution_height}")
                    
                    # Calculate astronomical parameters if focal length is provided
                    if self.config.focal_length_mm:
                        plate_scale = self.sensor_specs.plate_scale_arcsec_per_pixel(self.config.focal_length_mm)
                        fov = self.sensor_specs.field_of_view_deg(self.config.focal_length_mm)
                        sampling = self.sensor_specs.sampling_ratio(self.config.focal_length_mm, 2.0)
                        
                        logger.info(f"🔭 Optical system with {self.config.focal_length_mm}mm focal length:")
                        logger.info(f"   Plate scale: {plate_scale:.2f}\"/pixel")
                        logger.info(f"   Field of view: {fov[0]:.2f}° x {fov[1]:.2f}°")
                        logger.info(f"   Sampling @ 2\" seeing: {sampling:.2f}x")
                else:
                    logger.warning(f"⚠️ Sensor {self.config.sensor_name} not found in database")
            
            # Load camera system configuration
            if self.config.camera_system_config_file and os.path.exists(self.config.camera_system_config_file):
                with open(self.config.camera_system_config_file, 'r') as f:
                    system_data = json.load(f)
                    self.camera_system = CameraSystemConfig.from_dict(system_data)
                    logger.info(f"🎯 Loaded camera system configuration")
                    logger.info(f"   {self.camera_system.num_cameras} cameras in {self.camera_system.mounting_pattern} pattern")
                    logger.info(f"   Camera spacing: {self.camera_system.camera_spacing_mm}mm")
                    
                    # Calculate total system FOV
                    if self.sensor_specs:
                        total_fov = self.camera_system.calculate_system_fov(self.sensor_database)
                        if total_fov:
                            logger.info(f"   Total system FOV: {total_fov[0]:.2f}° x {total_fov[1]:.2f}°")
            
        except Exception as e:
            logger.error(f"Failed to initialize sensor configuration: {e}")
    
    def enhance_calibration_with_physical_units(self):
        """
        Enhance calibration data with physical measurements using sensor specifications.
        Converts pixel measurements to real-world units (mm, degrees, arcseconds).
        """
        if not self.sensor_specs:
            logger.warning("⚠️ No sensor specifications available for physical unit conversion")
            return
        
        logger.info("🔬 Enhancing calibration with physical measurements")
        
        # Enhance camera intrinsics with physical focal length
        for cam_id, cal in self.camera_calibrations.items():
            if cal.camera_matrix is not None:
                fx_pixels = cal.camera_matrix[0, 0]
                fy_pixels = cal.camera_matrix[1, 1]
                
                # Convert pixel focal length to mm
                fx_mm = fx_pixels * self.sensor_specs.pixel_width_um / 1000
                fy_mm = fy_pixels * self.sensor_specs.pixel_height_um / 1000
                focal_length_mm = (fx_mm + fy_mm) / 2
                
                # Calculate field of view for this camera
                fov_x = 2 * np.degrees(np.arctan(self.sensor_specs.sensor_width_mm / (2 * focal_length_mm)))
                fov_y = 2 * np.degrees(np.arctan(self.sensor_specs.sensor_height_mm / (2 * focal_length_mm)))
                
                # Calculate plate scale
                plate_scale = self.sensor_specs.plate_scale_arcsec_per_pixel(focal_length_mm)
                
                # Store physical parameters (extending the calibration data)
                cal.focal_length_mm = focal_length_mm
                cal.fov_degrees = (fov_x, fov_y)
                cal.plate_scale_arcsec_per_pixel = plate_scale
                
                logger.info(f"📷 Camera {cam_id} physical parameters:")
                logger.info(f"   Focal length: {focal_length_mm:.2f}mm")
                logger.info(f"   FOV: {fov_x:.2f}° x {fov_y:.2f}°")
                logger.info(f"   Plate scale: {plate_scale:.2f}\"/pixel")
        
        # Enhance stereo calibration with physical baseline measurements
        for (cam1_id, cam2_id), stereo_data in self.stereo_calibrations.items():
            if stereo_data.translation_vector is not None:
                # Translation vector is in ChAruco board units (typically mm)
                board_square_size = self.config.charuco_config.square_length * 1000  # Convert to mm
                
                # Scale translation to real-world units
                T_mm = stereo_data.translation_vector * board_square_size
                baseline_mm = np.linalg.norm(T_mm)
                
                # Update stereo data with physical units
                stereo_data.baseline_distance = baseline_mm
                stereo_data.translation_vector_mm = T_mm
                
                logger.info(f"🔗 Stereo pair ({cam1_id}, {cam2_id}) physical baseline: {baseline_mm:.2f}mm")
        
        # Calculate system-wide parameters
        if self.camera_poses:
            baselines = []
            for pose in self.camera_poses.values():
                if pose.baseline_distance and pose.baseline_distance > 0:
                    baselines.append(pose.baseline_distance)
            
            if baselines:
                avg_baseline = np.mean(baselines)
                max_baseline = np.max(baselines)
                
                logger.info(f"🎯 System baselines:")
                logger.info(f"   Average: {avg_baseline:.2f}mm")
                logger.info(f"   Maximum: {max_baseline:.2f}mm")
    
    def save_enhanced_calibration(self, filepath: str):
        """Save calibration with physical units to comprehensive file."""
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'sensor_info': self.sensor_specs.to_dict() if self.sensor_specs else None,
                'camera_intrinsics': {},
                'stereo_pairs': {},
                'camera_poses': {},
                'system_parameters': {}
            }
            
            # Save enhanced camera intrinsics
            for cam_id, cal in self.camera_calibrations.items():
                if cal.camera_matrix is not None:
                    data['camera_intrinsics'][str(cam_id)] = {
                        'camera_matrix': cal.camera_matrix.tolist(),
                        'dist_coeffs': cal.dist_coeffs.tolist(),
                        'image_size': list(cal.image_size) if cal.image_size else [],
                        'calibration_error': cal.calibration_error,
                        'focal_length_mm': getattr(cal, 'focal_length_mm', None),
                        'fov_degrees': getattr(cal, 'fov_degrees', None),
                        'plate_scale_arcsec_per_pixel': getattr(cal, 'plate_scale_arcsec_per_pixel', None)
                    }
            
            # Save enhanced stereo data
            for (cam1_id, cam2_id), stereo_data in self.stereo_calibrations.items():
                if stereo_data.rotation_matrix is not None:
                    pair_key = f"{cam1_id}_{cam2_id}"
                    data['stereo_pairs'][pair_key] = {
                        'rotation_matrix': stereo_data.rotation_matrix.tolist(),
                        'translation_vector': stereo_data.translation_vector.tolist(),
                        'baseline_distance_mm': stereo_data.baseline_distance,
                        'convergence_angle_deg': stereo_data.convergence_angle,
                        'stereo_error_pixels': stereo_data.stereo_error
                    }
            
            # Save system parameters
            if self.camera_system:
                data['system_parameters'] = self.camera_system.to_dict()
            
            # Write comprehensive calibration file
            output_path = filepath.replace('.json', '_enhanced.json')
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"✅ Saved enhanced calibration with physical units to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save enhanced calibration: {e}")
    
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
        logger.info(f"🎯 === STARTING CAMERA CALIBRATION ===")
        logger.info(f"📸 Performing calibration with {len(selected_frames)} diverse frames")
        
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
                logger.info(f"📷 Calibrating camera {cam_id} with {len(object_points)} frames, image_size={image_size}")
                
                # Log calibration data details
                total_corners = sum(len(pts) for pts in object_points)
                logger.info(f"   Total corners: {total_corners}, Frames: {len(object_points)}")
                
                try:
                    # Perform calibration with enhanced flags for sub-pixel accuracy
                    calibration_flags = (
                        cv2.CALIB_RATIONAL_MODEL +        # Use 8-coefficient distortion model
                        cv2.CALIB_THIN_PRISM_MODEL +      # Include thin prism distortion
                        cv2.CALIB_TILTED_MODEL            # Include sensor tilt correction
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
                        cal = self.camera_calibrations[cam_id]
                        cal.camera_matrix = camera_matrix
                        cal.dist_coeffs = dist_coeffs
                        cal.image_size = image_size
                        cal.calibration_error = ret
                        cal.num_frames_used = len(object_points)
                        cal.last_updated = datetime.now()
                        
                        logger.info(f"✅ Camera {cam_id} calibrated successfully!")
                        logger.info(f"   📊 Calibration error: {ret:.4f} pixels")
                        logger.info(f"   🔍 Focal length: fx={camera_matrix[0,0]:.1f}, fy={camera_matrix[1,1]:.1f}")
                        logger.info(f"   📐 Principal point: cx={camera_matrix[0,2]:.1f}, cy={camera_matrix[1,2]:.1f}")
                        logger.info(f"   🌊 Distortion coeffs: k1={dist_coeffs.flatten()[0]:.3f}, k2={dist_coeffs.flatten()[1]:.3f}")
                    else:
                        logger.warning(f"Calibration failed for camera {cam_id}")
                        
                except Exception as e:
                    logger.error(f"Error calibrating camera {cam_id}: {e}")
            else:
                logger.debug(f"Insufficient data for camera {cam_id}: "
                           f"{len(object_points)} frames")
        
        self.calibration_performed = True
        
        # Count successfully calibrated cameras
        calibrated_cameras = sum(1 for cal in self.camera_calibrations.values() if cal.camera_matrix is not None)
        logger.info(f"🎉 === CALIBRATION COMPLETE ===")
        logger.info(f"📷 Successfully calibrated {calibrated_cameras}/{self.config.num_cameras} cameras")
        
        # Save calibration if configured
        if self.config.calibration_file:
            logger.info(f"💾 Saving calibration data to: {self.config.calibration_file}")
            self.save_calibration(self.config.calibration_file)
        
        # Perform stereo calibration if enabled and we have enough cameras
        if (self.config.enable_stereo_calibration and 
            calibrated_cameras >= 2 and 
            selected_frames):
            logger.info(f"🔗 === STARTING STEREO CALIBRATION ===")
            self.perform_stereo_calibration(selected_frames)
        
        # Enhance calibration with physical measurements
        if self.sensor_specs:
            self.enhance_calibration_with_physical_units()
            
            # Save enhanced calibration file with physical units
            if self.config.calibration_file:
                self.save_enhanced_calibration(self.config.calibration_file)
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process synchronized frames from multiple cameras."""
        try:
            frames = data.get('frames', [])
            timestamp = data.get('timestamp')
            force_calibration = data.get('force_calibration', False)
            
            # If homographies are already computed, use the stored homographies for warped rendering
            if self.homographies_computed:
                logger.info("🎬 Homographies complete - continuing warped frame rendering for preview")
                
                # Use stored homographies from the last successful computation
                if hasattr(self, '_last_successful_warp_result') and self._last_successful_warp_result:
                    stored_homographies = self._last_successful_warp_result.get('homographies', [])
                    stored_valid_cameras = self._last_successful_warp_result.get('valid_cameras', [])
                    stored_reference_camera = self._last_successful_warp_result.get('reference_camera', 2)
                    
                    logger.info(f"🔍 Using stored homographies: {len(stored_homographies)} matrices, valid_cameras: {stored_valid_cameras}, reference: {stored_reference_camera}")
                    
                    # Create warped images manually using stored homographies
                    warped_images = []
                    for i, (frame, H) in enumerate(zip(frames, stored_homographies)):
                        if H is not None and i in stored_valid_cameras:
                            # Apply homography transformation
                            canvas_size = (self.config.warp_config.output_width, self.config.warp_config.output_height)
                            warped = cv2.warpPerspective(
                                frame, H, canvas_size,
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0)
                            )
                            warped_images.append(warped)
                        else:
                            # Create blank image for invalid cameras
                            blank = np.zeros((self.config.warp_config.output_height, 
                                            self.config.warp_config.output_width, 3), dtype=np.uint8)
                            warped_images.append(blank)
                    
                    # Create simple combined view by overlaying valid warped images
                    if warped_images:
                        combined = np.zeros_like(warped_images[0], dtype=np.float32)
                        pixel_count = np.zeros(warped_images[0].shape[:2], dtype=np.float32)
                        
                        for img in warped_images:
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            mask = (gray > 5).astype(np.float32)
                            combined += img.astype(np.float32) * mask[..., np.newaxis]
                            pixel_count += mask
                        
                        pixel_count[pixel_count == 0] = 1
                        combined_view = (combined / pixel_count[..., np.newaxis]).astype(np.uint8)
                    else:
                        combined_view = None
                    
                    return {
                        'frames': frames,
                        'warped_frames': warped_images,
                        'combined_view': combined_view,
                        'poses': [],
                        'homographies': stored_homographies,
                        'valid_cameras': stored_valid_cameras,
                        'reference_camera': stored_reference_camera,
                        'calibration_status': {
                            'calibrated': True,
                            'frames_collected': self.config.min_frames_for_calibration,
                            'frames_needed': self.config.min_frames_for_calibration,
                            'cameras_calibrated': len(self.camera_calibrations),
                            'homographies_available': True
                        },
                        'pipeline_should_stop': False,  # Keep pipeline running for warped preview
                        'homographies_computed': True,
                        'message': 'Homographies computed - warped preview active'
                    }
                else:
                    logger.warning("⚠️ No stored homographies available for warped rendering")
                    return {
                        'frames': frames,
                        'warped_frames': [],
                        'combined_view': None,
                        'poses': [],
                        'homographies': [],
                        'pipeline_should_stop': False,  # Keep pipeline running
                        'homographies_computed': True,
                        'message': 'No homographies available'
                    }
            
            if not frames:
                logger.warning("⚠️  No frames provided to multi-camera calibration")
                return {'error': 'No frames provided'}
            
            self.frames_processed += 1
            
            # Log periodic status
            if self.frames_processed % 60 == 0:  # Every 60 frames (~2 seconds)
                logger.info(f"🎬 Multi-camera processing: frame #{self.frames_processed}, {len(frames)} cameras")
                logger.info(f"   📊 Detection rate: {(self.successful_detections/self.frames_processed)*100:.1f}%")
                logger.info(f"   🎯 Calibration status: {'✅ DONE' if self.calibration_performed else '⏳ COLLECTING'}")
            
            # Detect ChAruco in all frames
            poses = await self.detect_charuco_parallel(frames, timestamp)
            
            # Count successful detections and analyze per camera
            valid_poses = sum(1 for p in poses if p.charuco_corners is not None and len(p.charuco_corners) >= 4)
            if valid_poses > 0:
                self.successful_detections += 1
            
            # Update pose diversity selection first
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
            
            # Unified camera status reporting every 30 frames (~1 second)
            if self.frames_processed % 30 == 0:
                logger.info(f"")  # Empty line for readability
                logger.info(f"🎬 === MULTI-CAMERA STATUS - Frame #{self.frames_processed} ===")
                
                camera_status = []
                total_aruco = 0
                total_charuco = 0
                cameras_with_charuco = 0
                
                for i, pose in enumerate(poses):
                    has_aruco = pose.aruco_ids is not None and len(pose.aruco_ids) > 0
                    has_charuco = pose.charuco_corners is not None and len(pose.charuco_corners) > 0
                    aruco_count = len(pose.aruco_ids) if has_aruco else 0
                    charuco_count = len(pose.charuco_corners) if has_charuco else 0
                    
                    total_aruco += aruco_count
                    total_charuco += charuco_count
                    if has_charuco:
                        cameras_with_charuco += 1
                    
                    # Individual camera status
                    if has_charuco:
                        status = f"✅ {aruco_count}ArUco→{charuco_count}ChAruco"
                        if pose.is_full_board:
                            status += " (FULL BOARD!)"
                        else:
                            expected_corners = (self.config.charuco_config.squares_x - 1) * (self.config.charuco_config.squares_y - 1)
                            percentage = (charuco_count / expected_corners) * 100
                            status += f" ({percentage:.1f}%)"
                    elif has_aruco:
                        status = f"🟡 {aruco_count}ArUco (no ChAruco)"
                    else:
                        status = f"❌ No markers"
                    
                    logger.info(f"   📷 Camera {i}: {status}")
                
                # Summary statistics
                detection_rate = (self.successful_detections/self.frames_processed)*100
                logger.info(f"")
                logger.info(f"📊 SUMMARY: {cameras_with_charuco}/{len(frames)} cameras with ChAruco")
                logger.info(f"   🎯 Total: {total_aruco} ArUco → {total_charuco} ChAruco corners")
                logger.info(f"   📈 Detection rate: {detection_rate:.1f}% ({self.successful_detections}/{self.frames_processed})")
                frames_collected = diversity_result.get("num_frames", 0)
                calibration_status = "✅ COMPLETE" if self.calibration_performed else f"⏳ Need {frames_collected}/5 diverse poses"
                logger.info(f"   🎯 Calibration: {calibration_status}")
                
                if total_aruco == 0:
                    logger.info(f"   💡 TIP: Show ChAruco board to cameras for calibration")
                elif total_charuco < 50:
                    logger.info(f"   💡 TIP: Improve lighting or move board closer for better detection")
                elif cameras_with_charuco > 0:
                    logger.info(f"   🎉 GREAT: ChAruco detection active on {cameras_with_charuco} cameras!")
                
                logger.info(f"========================================================")
                logger.info(f"")
            
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
            
            # Store successful warp result for later use
            if warp_result and warp_result.get('homographies') is not None and len(warp_result.get('homographies', [])) > 0:
                self._last_successful_warp_result = warp_result
                logger.info(f"💾 Stored warp result with {len(warp_result.get('homographies', []))} homographies for reuse")
            
            # Check if all homographies are computed (one identity reference, others with transforms)
            if not self.homographies_computed and self.calibration_performed:
                homographies = warp_result.get('homographies', [])
                if len(homographies) >= self.config.num_cameras:
                    identity_count = 0
                    transform_count = 0
                    
                    for H in homographies:
                        if np.allclose(H, np.eye(3), atol=1e-6):
                            identity_count += 1
                        else:
                            transform_count += 1
                    
                    # Check if we have exactly 1 reference camera (identity) and enough transforms
                    # Only require transforms for cameras that have sufficient detection in current frame
                    total_cameras_with_homography = identity_count + transform_count
                    
                    if identity_count == 1 and transform_count >= 2 and total_cameras_with_homography >= 3:
                        self.homographies_computed = True
                        logger.info(f"🎯 Sufficient homographies computed! Reference: 1, Transforms: {transform_count}")
                        
                        # Save homographies IMMEDIATELY while we have the correct data
                        if not hasattr(self, '_homographies_saved'):
                            self._save_homographies_now(warp_result, frames, poses)
                            self._homographies_saved = True
                        
                        logger.info(f"🛑 Homography computation complete - detection stopped, warped preview continues")
                        self.pipeline_should_stop = True  # Used internally to skip detection, but pipeline continues
            
            # Save calibration result images if calibration is complete
            if self.calibration_performed and not hasattr(self, '_calibration_images_saved'):
                self._save_calibration_images(warp_result, frames, poses)
                self._calibration_images_saved = True
            
            # Prepare output
            output = {
                'frames': frames,  # Add original frames for live preview
                'warped_frames': warp_result['warped_images'],
                'combined_view': warp_result['combined_view'],
                'poses': poses,
                'homographies': warp_result['homographies'],
                'valid_cameras': warp_result['valid_cameras'],
                'reference_camera': warp_result.get('reference_camera'),
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
                },
                'timestamp': timestamp,  # Also pass through timestamp
                'pipeline_should_stop': False,  # Keep pipeline running for warped preview
                'homographies_computed': self.homographies_computed
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
                'frames': frames,
                'warped_frames': frames,
                'combined_view': frames[0] if frames else None,
                'poses': [],
                'calibration_status': {'calibrated': False},
                'statistics': {'frames_processed': self.frames_processed}
            }
    
    def _save_homographies_now(self, warp_result: Dict[str, Any], frames: List[np.ndarray], poses: List[PoseResult]):
        """Save homography matrices immediately when computed."""
        try:
            import os
            from datetime import datetime
            import json
            
            # Create output directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"calibration_results_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
            
            logger.info(f"💾 Saving homographies to: {output_dir}/")
            
            # Save homography matrices using CURRENT frame homographies
            logger.info(f"🔍 Debug warp_result keys: {list(warp_result.keys())}")
            homographies = warp_result.get('homographies', [])
            valid_cameras = warp_result.get('valid_cameras', [])
            reference_camera = warp_result.get('reference_camera', 0)
            
            logger.info(f"🔍 Raw homographies type: {type(homographies)}, length: {len(homographies) if hasattr(homographies, '__len__') else 'N/A'}")
            logger.info(f"🔍 Raw valid_cameras: {valid_cameras}")
            logger.info(f"🔍 Raw reference_camera: {reference_camera}")
            
            homo_data = {}
            logger.info(f"📊 Processing homography matrices for saving:")
            logger.info(f"   Reference camera: {reference_camera}")
            logger.info(f"   Valid cameras: {valid_cameras}")
            
            for i, H in enumerate(homographies):
                if H is not None:
                    is_identity = np.allclose(H, np.eye(3), atol=1e-6)
                    logger.info(f"   Camera {i} homography debug:")
                    logger.info(f"     Matrix:\n{H}")
                    logger.info(f"     Is identity: {is_identity}")
                    logger.info(f"     In valid cameras: {i in valid_cameras}")
                    logger.info(f"     Is reference camera: {i == reference_camera}")
                    
                    # Save reference camera (identity) and cameras with valid transforms
                    if (is_identity and i == reference_camera) or (not is_identity and i in valid_cameras):
                        matrix_type = "Identity (reference)" if is_identity else "Transform"
                        logger.info(f"   Camera {i}: {matrix_type} matrix - SAVING")
                        homo_data[f'camera_{i}'] = H.tolist()
                    elif is_identity:
                        logger.info(f"   Camera {i}: Identity matrix (insufficient detection, not saved)")
                    else:
                        logger.info(f"   Camera {i}: Transform matrix (not in valid cameras, not saved)")
                        logger.warning(f"   Camera {i}: Valid cameras list: {valid_cameras}, Reference: {reference_camera}")
                else:
                    logger.info(f"   Camera {i}: No homography computed")
            
            if homo_data:
                homo_path = os.path.join(output_dir, "homographies.json")
                with open(homo_path, 'w') as f:
                    json.dump(homo_data, f, indent=2)
                logger.info(f"💾 Saved {len(homo_data)} homography matrices: homographies.json")
            else:
                logger.warning("⚠️  No homography matrices available to save")
                
        except Exception as e:
            logger.error(f"❌ Error saving homographies: {e}")
    
    def _save_calibration_images(self, warp_result: Dict[str, Any], frames: List[np.ndarray], poses: List[PoseResult]):
        """Save calibration result images for documentation."""
        try:
            import os
            from datetime import datetime
            
            # Create output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"calibration_results_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
            
            logger.info(f"📸 Saving calibration result images to: {output_dir}/")
            
            # Save combined/merged view
            combined_view = warp_result.get('combined_view')
            if combined_view is not None:
                combined_path = os.path.join(output_dir, "combined_view.jpg")
                cv2.imwrite(combined_path, combined_view)
                logger.info(f"💾 Saved combined view: combined_view.jpg ({combined_view.shape})")
            
            # Save individual warped frames with descriptive names
            warped_frames = warp_result.get('warped_images', [])
            for i, warped_frame in enumerate(warped_frames):
                if warped_frame is not None:
                    if i == 2:
                        # Reference camera - save as reference
                        warped_path = os.path.join(output_dir, f"cam_{i}_reference.jpg")
                        cv2.imwrite(warped_path, warped_frame)
                        logger.info(f"💾 Saved reference frame: cam_{i}_reference.jpg ({warped_frame.shape})")
                    else:
                        # Other cameras - save as aligned to cam 2
                        warped_path = os.path.join(output_dir, f"cam_{i}_to_cam2_warped.jpg")
                        cv2.imwrite(warped_path, warped_frame)
                        logger.info(f"💾 Saved aligned frame: cam_{i}_to_cam2_warped.jpg ({warped_frame.shape})")
            
            # Save original frames for comparison
            for i, frame in enumerate(frames):
                if frame is not None:
                    # Add ChAruco detection overlay if pose exists
                    display_frame = frame.copy()
                    if i < len(poses) and poses[i].charuco_corners is not None:
                        corners = poses[i].charuco_corners
                        ids = poses[i].charuco_ids
                        # Draw detected corners
                        cv2.aruco.drawDetectedCornersCharuco(display_frame, corners, ids, (0, 255, 0))
                        
                        # Add detection info overlay
                        corner_count = len(corners) if corners is not None else 0
                        expected_corners = (self.config.charuco_config.squares_x - 1) * (self.config.charuco_config.squares_y - 1)
                        info_text = f"Cam {i}: {corner_count}/{expected_corners} corners"
                        cv2.putText(display_frame, info_text, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        
                        if poses[i].rvec is not None:
                            distance = np.linalg.norm(poses[i].tvec)
                            pose_text = f"Distance: {distance:.3f}m"
                            cv2.putText(display_frame, pose_text, (10, 60), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    
                    original_path = os.path.join(output_dir, f"camera_{i}_original.jpg")
                    cv2.imwrite(original_path, display_frame)
                    logger.info(f"💾 Saved original frame {i}: camera_{i}_original.jpg ({frame.shape})")
            
            # Note: Homography matrices are now saved immediately when computed in _save_homographies_now()
            
            # Save summary info
            info_path = os.path.join(output_dir, "calibration_info.txt")
            with open(info_path, 'w') as f:
                f.write(f"ChAruco Multi-Camera Calibration Results\n")
                f.write(f"========================================\n\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Number of cameras: {len(frames)}\n")
                f.write(f"Board configuration: {self.config.charuco_config.squares_x}x{self.config.charuco_config.squares_y} squares, {self.config.charuco_config.square_length*1000:.1f}mm squares, {self.config.charuco_config.marker_length*1000:.1f}mm markers\n")
                f.write(f"Dictionary: {self.config.charuco_config.dictionary}\n\n")
                
                f.write(f"Detection Summary:\n")
                for i, pose in enumerate(poses):
                    if pose.charuco_corners is not None:
                        corner_count = len(pose.charuco_corners)
                        expected_corners = (self.config.charuco_config.squares_x - 1) * (self.config.charuco_config.squares_y - 1)
                        percentage = (corner_count / expected_corners) * 100
                        f.write(f"Camera {i}: {corner_count}/{expected_corners} corners ({percentage:.1f}%)")
                        if pose.rvec is not None:
                            distance = np.linalg.norm(pose.tvec)
                            f.write(f" - Distance: {distance:.3f}m")
                        f.write(f"\n")
                    else:
                        f.write(f"Camera {i}: No ChAruco detection\n")
                
                f.write(f"\nFiles saved:\n")
                f.write(f"- combined_view.jpg: Merged view of all cameras\n")
                f.write(f"- camera_X_warped.jpg: Individual warped frames\n") 
                f.write(f"- camera_X_original.jpg: Original frames with ChAruco overlay\n")
                f.write(f"- homographies.json: Perspective transformation matrices\n")
                f.write(f"- calibration_info.txt: This summary file\n")
            
            logger.info(f"📋 Saved calibration summary: calibration_info.txt")
            logger.info(f"🎉 Calibration results saved successfully to: {output_dir}/")
            
        except Exception as e:
            logger.error(f"❌ Error saving calibration images: {e}")

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