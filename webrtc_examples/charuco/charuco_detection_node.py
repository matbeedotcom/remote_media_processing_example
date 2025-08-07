"""
ChAruco detection and pose estimation node for camera calibration.
"""

from typing import Any, Dict, Optional, Tuple, List
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

logger = logging.getLogger(__name__)


@dataclass
class CharucoConfig:
    """Configuration for ChAruco board."""
    squares_x: int = 27
    squares_y: int = 17
    square_length: float = 0.0092
    marker_length: float = 0.006
    dictionary: str = "DICT_6X6_250"
    margins: float = 0.0058
    dpi: int = 227


@dataclass
class PoseResult:
    """Result of pose estimation."""
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    charuco_corners: Optional[np.ndarray] = None
    charuco_ids: Optional[np.ndarray] = None
    aruco_corners: Optional[List] = None
    aruco_ids: Optional[np.ndarray] = None
    is_full_board: bool = False
    camera_id: Optional[int] = None
    timestamp: Optional[float] = None


class CharucoDetectionNode(Node):
    """
    Detects ChAruco board markers and estimates camera pose.
    
    Input: Dict with 'image' (numpy array), 'camera_matrix', 'dist_coeffs', 
           optional 'camera_id' and 'timestamp'
    Output: PoseResult with detection and pose estimation results
    """
    
    def __init__(
        self,
        config: Optional[CharucoConfig] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name or "CharucoDetection")
        self.config = config or CharucoConfig()
        
        # Statistics tracking
        self.frame_count = 0
        self.detection_count = 0
        self.pose_estimation_count = 0
        self.last_log_time = 0
        self.log_interval = 30  # Log every 30 frames
        
        # Initialize ArUco dictionary and board
        dict_id = getattr(cv2.aruco, self.config.dictionary)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        
        # Create ChAruco board
        self.board = cv2.aruco.CharucoBoard(
            (self.config.squares_x, self.config.squares_y),
            self.config.square_length,
            self.config.marker_length,
            self.aruco_dict
        )
        
        # Initialize detector parameters
        self.detector_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
        
        # Initialize ChAruco detector (newer API)
        charuco_params = cv2.aruco.CharucoParameters()
        self.charuco_detector = cv2.aruco.CharucoDetector(self.board, charuco_params)
        
        # Calculate expected corners for full board detection
        self.expected_corners = (self.config.squares_x - 1) * (self.config.squares_y - 1)
        
        logger.info(f"📋 Initialized ChAruco detector:")
        logger.info(f"   📐 Board: {self.config.squares_x}x{self.config.squares_y} squares")
        logger.info(f"   📏 Square size: {self.config.square_length*1000:.1f}mm")
        logger.info(f"   🎯 Marker size: {self.config.marker_length*1000:.1f}mm")
        logger.info(f"   📖 Dictionary: {self.config.dictionary}")
        logger.info(f"   🔢 Expected corners: {self.expected_corners}")
    
    async def process(self, data: Dict[str, Any]) -> PoseResult:
        """Process image to detect ChAruco board and estimate pose."""
        import time
        
        self.frame_count += 1
        frame_start_time = time.time()
        
        try:
            # Extract input data
            image = data.get('image')
            camera_matrix = data.get('camera_matrix')
            dist_coeffs = data.get('dist_coeffs')
            camera_id = data.get('camera_id', 0)
            timestamp = data.get('timestamp')
            
            if image is None:
                logger.warning("⚠️  No image provided to ChAruco detector")
                return PoseResult(camera_id=camera_id, timestamp=timestamp)
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detect ArUco markers first
            corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
            
            result = PoseResult(
                camera_id=camera_id,
                timestamp=timestamp,
                aruco_corners=corners,
                aruco_ids=ids
            )
            
            aruco_count = len(ids) if ids is not None else 0
            charuco_corners_count = 0
            pose_success = False
            
            if ids is not None and len(ids) > 0:
                # Use the new ChAruco detector API
                charuco_corners, charuco_ids, aruco_corners, aruco_ids = self.charuco_detector.detectBoard(gray)
                
                if charuco_corners is not None and len(charuco_corners) > 3:
                    self.detection_count += 1
                    charuco_corners_count = len(charuco_corners)
                    
                    result.charuco_corners = charuco_corners
                    result.charuco_ids = charuco_ids
                    
                    # Check if full board is detected
                    result.is_full_board = (len(charuco_ids) == self.expected_corners)
                    
                    # Estimate pose if camera parameters are provided
                    if camera_matrix is not None and dist_coeffs is not None:
                        try:
                            # Use solvePnP for pose estimation with ChAruco corners
                            object_points = []
                            chessboard_corners = self.board.getChessboardCorners()
                            for corner_id in charuco_ids.flatten():
                                if corner_id < len(chessboard_corners):
                                    obj_pt = chessboard_corners[corner_id]
                                    object_points.append(obj_pt)
                            
                            if len(object_points) >= 4:
                                object_points = np.array(object_points, dtype=np.float32)
                                valid, rvec, tvec = cv2.solvePnP(
                                    object_points,
                                    charuco_corners,
                                    camera_matrix,
                                    dist_coeffs
                                )
                                
                                if valid:
                                    result.rvec = rvec
                                    result.tvec = tvec
                                    pose_success = True
                                    self.pose_estimation_count += 1
                                    
                                    # Log successful pose estimation
                                    if charuco_corners_count >= 20 or result.is_full_board:
                                        distance = np.linalg.norm(tvec)
                                        rotation_magnitude = np.linalg.norm(rvec)
                                        logger.info(f"🎯 ChAruco POSE detected! Cam {camera_id}: {charuco_corners_count}/{self.expected_corners} corners, "
                                                   f"distance={distance:.3f}m, rotation={rotation_magnitude:.3f}rad")
                                        if result.is_full_board:
                                            logger.info(f"✨ FULL BOARD detected! Perfect for calibration!")
                            else:
                                logger.debug(f"Insufficient object points for pose estimation: {len(object_points)}")
                        except Exception as pose_error:
                            logger.debug(f"Pose estimation error for camera {camera_id}: {pose_error}")
                    
                    # Log good detections
                    if charuco_corners_count >= 10:
                        logger.info(f"📋 ChAruco detected! Cam {camera_id}: {aruco_count} ArUco markers → {charuco_corners_count}/{self.expected_corners} corners "
                                   f"({'FULL BOARD' if result.is_full_board else f'{charuco_corners_count/self.expected_corners*100:.1f}%'})")
            
            # Periodic statistics logging
            processing_time = (time.time() - frame_start_time) * 1000
            should_log_stats = (self.frame_count % self.log_interval == 0 or 
                               (self.frame_count <= 5) or 
                               (aruco_count > 0))
            
            if should_log_stats:
                detection_rate = (self.detection_count / self.frame_count) * 100 if self.frame_count > 0 else 0
                pose_rate = (self.pose_estimation_count / self.frame_count) * 100 if self.frame_count > 0 else 0
                
                logger.info(f"📊 ChAruco Stats (Frame #{self.frame_count}):")
                logger.info(f"   🎯 Detection rate: {detection_rate:.1f}% ({self.detection_count}/{self.frame_count})")
                logger.info(f"   📐 Pose estimation rate: {pose_rate:.1f}% ({self.pose_estimation_count}/{self.frame_count})")
                logger.info(f"   ⚡ Processing: {processing_time:.1f}ms")
                logger.info(f"   🔍 Current: {aruco_count} ArUco → {charuco_corners_count} ChAruco corners")
                
                if aruco_count == 0:
                    logger.info(f"   💡 Tip: Show a ChAruco board to the camera for calibration")
                elif charuco_corners_count < 10:
                    logger.info(f"   💡 Tip: Move closer or improve lighting for better corner detection")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in ChAruco detection (frame #{self.frame_count}): {e}")
            return PoseResult(camera_id=camera_id, timestamp=timestamp)