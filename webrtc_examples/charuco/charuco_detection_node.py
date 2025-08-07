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
        
        logger.info(f"Initialized ChAruco detector with {self.config.squares_x}x{self.config.squares_y} board")
    
    async def process(self, data: Dict[str, Any]) -> PoseResult:
        """Process image to detect ChAruco board and estimate pose."""
        try:
            # Extract input data
            image = data.get('image')
            camera_matrix = data.get('camera_matrix')
            dist_coeffs = data.get('dist_coeffs')
            camera_id = data.get('camera_id')
            timestamp = data.get('timestamp')
            
            if image is None:
                logger.warning("No image provided")
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
            
            if ids is not None and len(ids) > 0:
                # Use the new ChAruco detector API
                charuco_corners, charuco_ids, aruco_corners, aruco_ids = self.charuco_detector.detectBoard(gray)
                
                if charuco_corners is not None and len(charuco_corners) > 3:
                    result.charuco_corners = charuco_corners
                    result.charuco_ids = charuco_ids
                    
                    # Check if full board is detected
                    expected_corners = (self.config.squares_x - 1) * (self.config.squares_y - 1)
                    result.is_full_board = (len(charuco_ids) == expected_corners)
                    
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
                                    logger.debug(f"Pose estimated for camera {camera_id}: rvec={rvec.flatten()}, tvec={tvec.flatten()}")
                                else:
                                    logger.debug(f"solvePnP failed for camera {camera_id}")
                            else:
                                logger.debug(f"Insufficient object points for pose estimation: {len(object_points)}")
                        except Exception as pose_error:
                            logger.debug(f"Pose estimation error for camera {camera_id}: {pose_error}")
                    else:
                        logger.debug("Camera parameters not provided, skipping pose estimation")
                else:
                    logger.debug(f"Insufficient ChAruco corners detected")
            else:
                logger.debug("No ArUco markers detected")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ChAruco detection: {e}")
            return PoseResult(camera_id=camera_id, timestamp=timestamp)