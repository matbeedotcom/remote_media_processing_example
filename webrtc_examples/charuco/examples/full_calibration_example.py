#!/usr/bin/env python3
"""
Example script demonstrating the full ChAruco calibration pipeline.

This script shows how to:
1. Perform individual full-screen calibration for each camera
2. Perform multi-camera calibration for stereo/multi-view alignment
3. Save and visualize calibration results

Usage:
    python charuco_full_calibration_example.py
"""

import asyncio
import cv2
import numpy as np
import logging
import sys
import os
from datetime import datetime
from typing import List, Optional

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from charuco.calibration.full_calibration_pipeline import FullCalibrationPipeline, FullCalibrationConfig
from charuco.nodes.charuco_detection_node import CharucoConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CameraCapture:
    """Simple camera capture class for testing."""
    
    def __init__(self, camera_indices: List[int]):
        """Initialize camera capture with given indices."""
        self.cameras = []
        self.num_cameras = len(camera_indices)
        
        for idx in camera_indices:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                # Set camera resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cameras.append(cap)
                logger.info(f"Opened camera {idx}")
            else:
                logger.warning(f"Failed to open camera {idx}")
                self.cameras.append(None)
    
    def read_frames(self) -> Optional[List[np.ndarray]]:
        """Read frames from all cameras."""
        frames = []
        
        for i, cap in enumerate(self.cameras):
            if cap is not None:
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    logger.warning(f"Failed to read from camera {i}")
                    frames.append(None)
            else:
                frames.append(None)
        
        # Return None if no valid frames
        if all(f is None for f in frames):
            return None
        
        return frames
    
    def release(self):
        """Release all cameras."""
        for cap in self.cameras:
            if cap is not None:
                cap.release()


def generate_charuco_board(config: CharucoConfig, output_path: str = "charuco_board.png"):
    """Generate and save a ChAruco board image."""
    import cv2.aruco as aruco
    
    # Get ArUco dictionary
    dict_mapping = {
        "DICT_4X4_50": aruco.DICT_4X4_50,
        "DICT_4X4_100": aruco.DICT_4X4_100,
        "DICT_4X4_250": aruco.DICT_4X4_250,
        "DICT_4X4_1000": aruco.DICT_4X4_1000,
        "DICT_5X5_50": aruco.DICT_5X5_50,
        "DICT_5X5_100": aruco.DICT_5X5_100,
        "DICT_5X5_250": aruco.DICT_5X5_250,
        "DICT_5X5_1000": aruco.DICT_5X5_1000,
    }
    
    dict_id = dict_mapping.get(config.dictionary, aruco.DICT_4X4_50)
    aruco_dict = aruco.getPredefinedDictionary(dict_id)
    
    # Create ChAruco board
    board = aruco.CharucoBoard(
        (config.squares_x, config.squares_y),
        config.square_length,
        config.marker_length,
        aruco_dict
    )
    
    # Calculate image size
    # Assuming 200 DPI (dots per inch), convert from meters to pixels
    dpi = config.dpi
    inches_per_meter = 39.3701
    
    # Board dimensions in meters
    board_width = config.squares_x * config.square_length + 2 * config.margins
    board_height = config.squares_y * config.square_length + 2 * config.margins
    
    # Convert to pixels
    img_width = int(board_width * inches_per_meter * dpi)
    img_height = int(board_height * inches_per_meter * dpi)
    
    # Generate board image
    board_image = board.generateImage((img_width, img_height))
    
    # Save board image
    cv2.imwrite(output_path, board_image)
    logger.info(f"Generated ChAruco board: {output_path}")
    logger.info(f"Board size: {img_width}x{img_height} pixels")
    logger.info(f"Configuration: {config.squares_x}x{config.squares_y} squares")
    logger.info(f"Square size: {config.square_length*1000:.1f}mm, Marker size: {config.marker_length*1000:.1f}mm")
    
    return board_image


async def run_calibration_with_cameras(camera_indices: List[int]):
    """Run calibration pipeline with real cameras."""
    
    # Configure ChAruco board
    charuco_config = CharucoConfig(
        squares_x=7,
        squares_y=5,
        square_length=0.035,  # 35mm squares
        marker_length=0.025,   # 25mm markers
        dictionary="DICT_4X4_50"
    )
    
    # Generate and save ChAruco board
    board_image = generate_charuco_board(charuco_config, "charuco_board_7x5.png")
    
    # Configure calibration pipeline
    pipeline_config = FullCalibrationConfig(
        num_cameras=len(camera_indices),
        charuco_config=charuco_config,
        individual_frames_per_camera=25,
        individual_min_coverage=70.0,
        multi_camera_frames=20,
        enable_stereo_calibration=True,
        calibration_output_dir=f"calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        enable_live_visualization=True
    )
    
    # Create pipeline
    pipeline = FullCalibrationPipeline(pipeline_config)
    
    # Initialize cameras
    capture = CameraCapture(camera_indices)
    
    # Create visualization window
    cv2.namedWindow("Calibration Progress", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration Progress", 1280, 720)
    
    logger.info("=" * 60)
    logger.info("STARTING FULL CALIBRATION PIPELINE")
    logger.info(f"Number of cameras: {len(camera_indices)}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("INSTRUCTIONS:")
    logger.info("1. Individual Calibration Phase:")
    logger.info("   - Show ChAruco board to each camera individually")
    logger.info("   - Move board to cover entire screen area")
    logger.info("   - System will guide you through each camera")
    logger.info("")
    logger.info("2. Multi-Camera Calibration Phase:")
    logger.info("   - Show board visible to multiple cameras")
    logger.info("   - Move board to different positions")
    logger.info("   - Keep board in view of at least 2 cameras")
    logger.info("")
    logger.info("Press 'q' to quit, 's' to save current state")
    logger.info("=" * 60)
    
    frame_count = 0
    
    try:
        while True:
            # Read frames from cameras
            frames = capture.read_frames()
            
            if frames is None:
                logger.error("Failed to read from cameras")
                break
            
            # Process frames through pipeline
            result = await pipeline.process({
                'frames': frames,
                'timestamp': frame_count / 30.0  # Assuming 30 fps
            })
            
            frame_count += 1
            
            # Display visualization
            visualization = result.get('visualization')
            if visualization is None:
                # Create simple grid view if no visualization available
                combined_view = result.get('combined_view')
                if combined_view is not None:
                    visualization = combined_view
                else:
                    # Show individual frames in grid
                    display_frames = result.get('warped_frames', frames)
                    if display_frames:
                        # Create 2x2 grid
                        rows = []
                        for i in range(0, len(display_frames), 2):
                            row = display_frames[i:i+2]
                            while len(row) < 2:
                                row.append(np.zeros_like(display_frames[0]))
                            
                            # Resize for display
                            row = [cv2.resize(f, (640, 480)) if f is not None 
                                  else np.zeros((480, 640, 3), dtype=np.uint8) 
                                  for f in row]
                            rows.append(np.hstack(row))
                        
                        visualization = np.vstack(rows) if rows else display_frames[0]
            
            if visualization is not None:
                # Add status text
                phase = result.get('phase', 'unknown')
                message = result.get('message', '')
                
                status_text = f"Phase: {phase}"
                cv2.putText(visualization, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                if message:
                    cv2.putText(visualization, message, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.imshow("Calibration Progress", visualization)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                logger.info("User requested quit")
                break
            elif key == ord('s'):
                # Save current state
                pipeline.save_final_calibration_report()
                logger.info("Saved current calibration state")
            
            # Check if pipeline is complete
            if result.get('pipeline_complete'):
                logger.info("=" * 60)
                logger.info("CALIBRATION COMPLETE!")
                logger.info(f"Results saved to: {result.get('calibration_directory')}")
                logger.info("=" * 60)
                
                # Show final result for a few seconds
                cv2.waitKey(5000)
                break
            
            # Log progress periodically
            if frame_count % 30 == 0:  # Every second
                status = pipeline.get_pipeline_status()
                current_phase = status['current_phase']
                
                if current_phase == 'individual_calibration':
                    ind_status = status['individual_calibration']
                    logger.info(f"Individual calibration: {ind_status['completed']}/{ind_status['total_cameras']} cameras complete")
                elif current_phase == 'multi_camera_calibration':
                    multi_status = status['multi_camera_calibration']
                    if multi_status:
                        logger.info(f"Multi-camera calibration: {multi_status['frames_in_selection']}/{pipeline_config.multi_camera_frames} frames")
        
    except KeyboardInterrupt:
        logger.info("Calibration interrupted by user")
    
    finally:
        # Clean up
        capture.release()
        cv2.destroyAllWindows()
        
        # Save final state
        pipeline.save_final_calibration_report()
        
        # Print final summary
        status = pipeline.get_pipeline_status()
        logger.info("")
        logger.info("FINAL SUMMARY:")
        logger.info(f"Frames processed: {status['frames_processed']}")
        logger.info(f"Pipeline complete: {status['pipeline_complete']}")
        logger.info(f"Output directory: {status['output_directory']}")


async def run_calibration_with_test_data():
    """Run calibration pipeline with simulated test data."""
    
    logger.info("Running calibration with simulated test data...")
    
    # Configure pipeline
    pipeline_config = FullCalibrationConfig(
        num_cameras=4,
        individual_frames_per_camera=15,
        individual_min_coverage=60.0,
        multi_camera_frames=10,
        calibration_output_dir="test_calibration_results"
    )
    
    # Create pipeline
    pipeline = FullCalibrationPipeline(pipeline_config)
    
    # Simulate frames
    num_frames = 200
    for i in range(num_frames):
        # Create dummy frames (in real usage, these would be camera frames)
        frames = []
        for cam_id in range(4):
            # Create a test pattern
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            
            # Add some visual content
            cv2.putText(frame, f"Camera {cam_id}", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame {i}", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
            
            frames.append(frame)
        
        # Process frames
        result = await pipeline.process({
            'frames': frames,
            'timestamp': i / 30.0
        })
        
        # Log progress
        if i % 10 == 0:
            logger.info(f"Frame {i}/{num_frames}: Phase={result.get('phase')}, Message={result.get('message')}")
        
        # Check if complete
        if result.get('pipeline_complete'):
            logger.info(f"Calibration complete at frame {i}")
            break
    
    # Get final status
    status = pipeline.get_pipeline_status()
    logger.info(f"Final status: {status}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Full ChAruco calibration pipeline example")
    parser.add_argument('--cameras', type=int, nargs='+', default=[0, 1, 2, 3],
                       help='Camera indices to use (default: 0 1 2 3)')
    parser.add_argument('--test', action='store_true',
                       help='Run with simulated test data instead of real cameras')
    parser.add_argument('--generate-board', action='store_true',
                       help='Only generate ChAruco board image and exit')
    
    args = parser.parse_args()
    
    if args.generate_board:
        # Just generate board
        config = CharucoConfig(
            squares_x=7,
            squares_y=5,
            square_length=0.035,
            marker_length=0.025,
            dictionary="DICT_4X4_50"
        )
        generate_charuco_board(config, "charuco_board_7x5.png")
        logger.info("Board generated. Print at actual size for calibration.")
        return
    
    if args.test:
        # Run with test data
        asyncio.run(run_calibration_with_test_data())
    else:
        # Run with real cameras
        asyncio.run(run_calibration_with_cameras(args.cameras))


if __name__ == "__main__":
    main()