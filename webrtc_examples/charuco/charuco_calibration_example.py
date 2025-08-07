"""
Example usage of the ChAruco calibration and perspective warping nodes.
Demonstrates how to use the multi-camera calibration system.
"""

import asyncio
import cv2
import numpy as np
import logging
from typing import List
import os
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '..', 'remote_media_processing'))

from charuco_detection_node import CharucoConfig
from perspective_warp_node import WarpConfig
from multi_camera_calibration_node import (
    MultiCameraCalibrationNode,
    MultiCameraConfig
)


class MockCameraArray:
    """
    Mock camera array for testing the calibration system.
    Simulates multiple cameras viewing a ChAruco board.
    """
    
    def __init__(self, num_cameras: int = 4):
        self.num_cameras = num_cameras
        self.charuco_config = CharucoConfig()
        
        # Create ChAruco board for simulation
        dict_id = getattr(cv2.aruco, self.charuco_config.dictionary)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self.board = cv2.aruco.CharucoBoard(
            (self.charuco_config.squares_x, self.charuco_config.squares_y),
            self.charuco_config.square_length,
            self.charuco_config.marker_length,
            self.aruco_dict
        )
        
        # Generate board image
        self.board_image = self.board.generateImage(
            (2000, 1400), marginSize=int(self.charuco_config.margins * 1000)
        )
        
        # Setup camera parameters for each simulated camera
        self.setup_cameras()
    
    def setup_cameras(self):
        """Setup simulated camera parameters."""
        self.camera_configs = []
        
        # Base camera matrix
        base_focal = 1000
        image_width, image_height = 1920, 1080
        
        for i in range(self.num_cameras):
            # Vary camera positions and orientations
            angle = i * (2 * np.pi / self.num_cameras)
            distance = 1.0  # meters from board
            height = 0.3 + i * 0.1  # vary height
            
            # Camera position
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            z = height
            
            # Camera matrix (intrinsics)
            camera_matrix = np.array([
                [base_focal + i * 50, 0, image_width/2],
                [0, base_focal + i * 30, image_height/2],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Distortion coefficients
            dist_coeffs = np.array([0.1 - i*0.02, -0.05 + i*0.01, 0, 0, 0], dtype=np.float32)
            
            # Camera pose (extrinsics)
            tvec = np.array([[x], [y], [z]], dtype=np.float32)
            
            # Look at board (rotation)
            look_at = np.array([0, 0, 0]) - np.array([x, y, z])
            look_at = look_at / np.linalg.norm(look_at)
            
            # Simple rotation (this is simplified)
            rvec = np.array([[0], [angle], [0]], dtype=np.float32)
            
            self.camera_configs.append({
                'camera_matrix': camera_matrix,
                'dist_coeffs': dist_coeffs,
                'rvec': rvec,
                'tvec': tvec,
                'image_size': (image_width, image_height)
            })
    
    def capture_frames(self, pose_variation: float = 0.0) -> List[np.ndarray]:
        """
        Generate simulated camera frames showing the ChAruco board.
        
        Args:
            pose_variation: Amount of random pose variation to add
        """
        frames = []
        
        for i, config in enumerate(self.camera_configs):
            # Add some pose variation for diversity
            rvec = config['rvec'].copy()
            tvec = config['tvec'].copy()
            
            if pose_variation > 0:
                rvec += np.random.normal(0, pose_variation, rvec.shape).astype(np.float32)
                tvec += np.random.normal(0, pose_variation, tvec.shape).astype(np.float32)
            
            # Project board to camera view
            frame = self.project_board_to_camera(
                config['camera_matrix'],
                config['dist_coeffs'],
                rvec,
                tvec,
                config['image_size']
            )
            
            frames.append(frame)
        
        return frames
    
    def project_board_to_camera(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        image_size: tuple
    ) -> np.ndarray:
        """Project the ChAruco board to a camera view."""
        # Create blank image
        frame = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        
        # For simplicity, just warp the board image
        # In a real implementation, you'd properly project the 3D board
        try:
            # Simple perspective transformation
            # This is a simplified projection
            height, width = frame.shape[:2]
            board_height, board_width = self.board_image.shape[:2]
            
            # Define source points (board corners)
            src_points = np.float32([
                [0, 0],
                [board_width, 0],
                [board_width, board_height],
                [0, board_height]
            ])
            
            # Define destination points with some perspective
            offset_x = int(0.1 * width)
            offset_y = int(0.1 * height)
            dst_points = np.float32([
                [offset_x, offset_y],
                [width - offset_x, offset_y + 20],
                [width - offset_x - 10, height - offset_y],
                [offset_x + 10, height - offset_y - 20]
            ])
            
            # Apply perspective transformation
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            warped = cv2.warpPerspective(self.board_image, M, (width, height))
            
            # Convert to color
            if len(warped.shape) == 2:
                warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
            
            # Blend with background
            mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            mask = mask > 0
            frame[mask] = warped[mask]
            
            # Add some noise and background
            noise = np.random.randint(0, 50, frame.shape, dtype=np.uint8)
            frame = cv2.addWeighted(frame, 0.9, noise, 0.1, 0)
            
        except Exception as e:
            logger.warning(f"Error projecting board: {e}")
            # Just return a test pattern
            cv2.rectangle(frame, (100, 100), (width-100, height-100), (100, 100, 100), 2)
        
        return frame


async def main():
    """Main example function."""
    logger.info("Starting ChAruco calibration example")
    
    # Configuration
    charuco_config = CharucoConfig(
        squares_x=27,
        squares_y=17,
        square_length=0.0092,
        marker_length=0.006,
        dictionary="DICT_6X6_250",
        margins=0.0058,
        dpi=227
    )
    
    warp_config = WarpConfig(
        output_width=1920,
        output_height=1080,
        reference_camera=0
    )
    
    multi_camera_config = MultiCameraConfig(
        num_cameras=4,
        charuco_config=charuco_config,
        warp_config=warp_config,
        max_calibration_frames=10,
        min_frames_for_calibration=5,
        auto_calibrate=True,
        calibration_file="camera_calibration.json",
        enable_live_preview=True
    )
    
    # Create calibration node
    calibration_node = MultiCameraCalibrationNode(config=multi_camera_config)
    
    # Create mock camera array
    camera_array = MockCameraArray(num_cameras=4)
    
    # Simulate calibration process
    logger.info("Starting calibration data collection...")
    
    for frame_idx in range(20):  # Simulate 20 frames
        logger.info(f"Processing frame {frame_idx + 1}/20")
        
        # Generate frames with pose variation for diversity
        pose_variation = 0.1 + frame_idx * 0.01  # Increase variation over time
        frames = camera_array.capture_frames(pose_variation=pose_variation)
        
        # Process frames through calibration node
        result = await calibration_node.process({
            'frames': frames,
            'timestamp': frame_idx / 30.0,  # Simulate 30 FPS
            'blend_mode': 'overlay',
            'debug_visualization': True,
            'include_calibration_details': True
        })
        
        # Display results
        if frame_idx % 5 == 0:  # Every 5th frame
            if 'error' not in result:
                status = result.get('calibration_status', {})
                stats = result.get('statistics', {})
                
                logger.info(f"Frame {frame_idx}:")
                logger.info(f"  Calibrated: {status.get('calibrated', False)}")
                logger.info(f"  Frames collected: {status.get('frames_collected', 0)}/{status.get('frames_needed', 5)}")
                logger.info(f"  Cameras calibrated: {status.get('cameras_calibrated', 0)}")
                logger.info(f"  Detection rate: {stats.get('detection_rate', 0):.2%}")
                logger.info(f"  Valid poses: {stats.get('valid_poses_current', 0)}")
            else:
                logger.warning(f"Frame {frame_idx} failed: {result.get('error', 'Unknown error')}")
        
        # Save images for first few frames (instead of display)
        if frame_idx < 3 and result.get('combined_view') is not None:
            try:
                cv2.imwrite(f'combined_view_frame_{frame_idx}.png', result['combined_view'])
                cv2.imwrite(f'original_camera0_frame_{frame_idx}.png', frames[0])
                logger.info(f"Saved images for frame {frame_idx}")
            except Exception as e:
                logger.debug(f"Could not save images: {e}")
    
    # Final calibration summary
    summary = calibration_node.get_calibration_summary()
    logger.info("Final Calibration Summary:")
    logger.info(f"  Total cameras: {summary['num_cameras']}")
    logger.info(f"  Calibrated cameras: {summary['calibrated_cameras']}")
    logger.info(f"  Calibration performed: {summary['calibration_performed']}")
    logger.info(f"  Frames in selection: {summary['frames_in_selection']}")
    logger.info(f"  Board configuration: {summary['board_config']}")
    
    # Test perspective warping with calibrated cameras
    if summary['calibration_performed']:
        logger.info("Testing perspective warping with calibrated system...")
        
        test_frames = camera_array.capture_frames(pose_variation=0.05)
        warp_result = await calibration_node.process({
            'frames': test_frames,
            'timestamp': 999.0,
            'blend_mode': 'overlay',
            'debug_visualization': True
        })
        
        if warp_result.get('combined_view') is not None:
            try:
                cv2.imwrite('final_warped_view.png', warp_result['combined_view'])
                logger.info("Saved final warped view to final_warped_view.png")
            except Exception as e:
                logger.debug(f"Could not save final image: {e}")
    
    # cv2.destroyAllWindows()  # Skip GUI calls
    logger.info("Example completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())