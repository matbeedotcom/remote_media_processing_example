"""
Full calibration pipeline that performs individual camera calibration first,
then multi-camera calibration for stereo/multi-view alignment.
"""

from typing import Any, Dict, Optional, List
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
from .individual_camera_calibration_node import IndividualCameraCalibrationNode, IndividualCalibrationConfig
from .multi_camera_calibration_node import MultiCameraCalibrationNode, MultiCameraConfig
from ..nodes.charuco_detection_node import CharucoConfig

logger = logging.getLogger(__name__)


class CalibrationPhase(Enum):
    """Current phase of the calibration pipeline."""
    INDIVIDUAL_CALIBRATION = "individual_calibration"
    MULTI_CAMERA_CALIBRATION = "multi_camera_calibration"
    COMPLETED = "completed"


@dataclass
class FullCalibrationConfig:
    """Configuration for the full calibration pipeline."""
    num_cameras: int = 4
    charuco_config: CharucoConfig = field(default_factory=CharucoConfig)
    
    # Individual calibration settings
    individual_frames_per_camera: int = 20
    individual_min_coverage: float = 75.0
    require_individual_calibration: bool = True
    
    # Multi-camera calibration settings
    multi_camera_frames: int = 15
    enable_stereo_calibration: bool = True
    
    # File paths
    calibration_output_dir: str = "calibration_results"
    save_intermediate_results: bool = True
    
    # Visualization
    enable_live_visualization: bool = True
    visualization_size: tuple = (1920, 1080)


class FullCalibrationPipeline(Node):
    """
    Complete calibration pipeline that handles both individual and multi-camera calibration.
    
    Phase 1: Individual camera calibration with full-screen coverage
    Phase 2: Multi-camera calibration for stereo/multi-view alignment
    
    Input: Dict with 'frames' (List of images), 'timestamp'
    Output: Dict with complete calibration data and status
    """
    
    def __init__(
        self,
        config: Optional[FullCalibrationConfig] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name or "FullCalibrationPipeline")
        self.config = config or FullCalibrationConfig()
        
        # Create output directory
        os.makedirs(self.config.calibration_output_dir, exist_ok=True)
        
        # Initialize calibration nodes
        individual_config = IndividualCalibrationConfig(
            num_cameras=self.config.num_cameras,
            charuco_config=self.config.charuco_config,
            frames_per_camera=self.config.individual_frames_per_camera,
            min_coverage_percentage=self.config.individual_min_coverage,
            calibration_file_prefix=os.path.join(
                self.config.calibration_output_dir, 
                "individual_calibration"
            )
        )
        self.individual_calibration_node = IndividualCameraCalibrationNode(individual_config)
        
        multi_config = MultiCameraConfig(
            num_cameras=self.config.num_cameras,
            charuco_config=self.config.charuco_config,
            max_calibration_frames=self.config.multi_camera_frames,
            enable_stereo_calibration=self.config.enable_stereo_calibration,
            calibration_file=os.path.join(
                self.config.calibration_output_dir,
                "multi_camera_calibration.json"
            ),
            stereo_calibration_file=os.path.join(
                self.config.calibration_output_dir,
                "stereo_calibration.json"
            )
        )
        self.multi_camera_node = MultiCameraCalibrationNode(multi_config)
        
        # Pipeline state
        self.current_phase = CalibrationPhase.INDIVIDUAL_CALIBRATION
        self.individual_calibrations_loaded = False
        self.frames_processed = 0
        self.pipeline_complete = False
        
        logger.info(f"Initialized full calibration pipeline for {self.config.num_cameras} cameras")
        logger.info(f"Output directory: {self.config.calibration_output_dir}")
        logger.info(f"Phase 1: Individual calibration ({self.config.individual_frames_per_camera} frames/camera)")
        logger.info(f"Phase 2: Multi-camera calibration ({self.config.multi_camera_frames} frames total)")
    
    def transition_to_multi_camera(self, individual_calibrations: Dict[int, Dict[str, Any]]):
        """Transition from individual to multi-camera calibration phase."""
        logger.info("=" * 60)
        logger.info("🎯 TRANSITIONING TO MULTI-CAMERA CALIBRATION PHASE")
        logger.info("=" * 60)
        
        # Load individual calibrations into multi-camera node
        for cam_id, cal_data in individual_calibrations.items():
            if cam_id in self.multi_camera_node.camera_calibrations:
                cal = self.multi_camera_node.camera_calibrations[cam_id]
                cal.camera_matrix = cal_data['camera_matrix']
                cal.dist_coeffs = cal_data['dist_coeffs']
                cal.image_size = cal_data['image_size']
                cal.calibration_error = cal_data['calibration_error']
                cal.num_frames_used = 0  # Will be updated during multi-camera phase
                cal.last_updated = datetime.now()
        
        # Mark individual calibrations as loaded
        self.multi_camera_node.calibration_performed = True
        self.individual_calibrations_loaded = True
        self.current_phase = CalibrationPhase.MULTI_CAMERA_CALIBRATION
        
        logger.info(f"Loaded {len(individual_calibrations)} individual camera calibrations")
        logger.info("Now collecting synchronized frames for stereo calibration...")
        logger.info("Please show ChAruco board visible to multiple cameras simultaneously")
    
    def create_combined_visualization(
        self, 
        frames: List[np.ndarray], 
        phase_data: Dict[str, Any]
    ) -> np.ndarray:
        """Create a combined visualization showing calibration progress."""
        if not frames or not self.config.enable_live_visualization:
            return None
        
        # Get visualization from active phase
        if self.current_phase == CalibrationPhase.INDIVIDUAL_CALIBRATION:
            return self.individual_calibration_node.create_calibration_visualization(
                frames, 
                phase_data.get('active_camera')
            )
        else:
            # For multi-camera phase, create grid view with status
            return self.create_multi_camera_visualization(frames, phase_data)
    
    def create_multi_camera_visualization(
        self, 
        frames: List[np.ndarray], 
        phase_data: Dict[str, Any]
    ) -> np.ndarray:
        """Create visualization for multi-camera calibration phase."""
        if not frames:
            return None
        
        # Use warped frames if available
        warped_frames = phase_data.get('warped_frames', frames)
        combined_view = phase_data.get('combined_view')
        
        if combined_view is not None:
            # Add status overlay to combined view
            viz = combined_view.copy()
            
            status = phase_data.get('calibration_status', {})
            frames_collected = status.get('frames_collected', 0)
            frames_needed = status.get('frames_needed', 15)
            
            status_text = f"Multi-Camera Calibration: {frames_collected}/{frames_needed} frames"
            cv2.putText(viz, status_text, (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
            if status.get('calibrated'):
                cv2.putText(viz, "✓ Calibration Complete!", (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            return viz
        else:
            # Create simple grid view
            rows = 2
            cols = (len(frames) + 1) // 2
            display_size = (640, 480)
            
            resized = []
            for frame in warped_frames:
                if frame is not None:
                    resized.append(cv2.resize(frame, display_size))
                else:
                    resized.append(np.zeros((display_size[1], display_size[0], 3), dtype=np.uint8))
            
            while len(resized) < rows * cols:
                resized.append(np.zeros((display_size[1], display_size[0], 3), dtype=np.uint8))
            
            grid_rows = []
            for r in range(rows):
                row_frames = resized[r*cols:(r+1)*cols]
                grid_rows.append(np.hstack(row_frames))
            
            return np.vstack(grid_rows)
    
    def save_final_calibration_report(self):
        """Save comprehensive calibration report."""
        report_path = os.path.join(self.config.calibration_output_dir, "calibration_report.json")
        
        try:
            # Gather all calibration data
            report = {
                'timestamp': datetime.now().isoformat(),
                'configuration': {
                    'num_cameras': self.config.num_cameras,
                    'charuco_board': {
                        'squares_x': self.config.charuco_config.squares_x,
                        'squares_y': self.config.charuco_config.squares_y,
                        'square_length': self.config.charuco_config.square_length,
                        'marker_length': self.config.charuco_config.marker_length,
                        'dictionary': self.config.charuco_config.dictionary
                    }
                },
                'individual_calibration': self.individual_calibration_node.get_calibration_summary(),
                'multi_camera_calibration': self.multi_camera_node.get_calibration_summary(),
                'frames_processed': self.frames_processed,
                'pipeline_complete': self.pipeline_complete
            }
            
            # Add stereo calibration data if available
            if self.multi_camera_node.stereo_calibrations:
                stereo_summary = []
                for (cam1, cam2), stereo_data in self.multi_camera_node.stereo_calibrations.items():
                    if stereo_data.rotation_matrix is not None:
                        stereo_summary.append({
                            'camera_pair': [cam1, cam2],
                            'baseline_mm': stereo_data.baseline_distance,
                            'convergence_angle_deg': stereo_data.convergence_angle,
                            'stereo_error': stereo_data.stereo_error
                        })
                report['stereo_calibration'] = stereo_summary
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"📊 Saved calibration report to {report_path}")
            
            # Also save a human-readable summary
            summary_path = os.path.join(self.config.calibration_output_dir, "calibration_summary.txt")
            with open(summary_path, 'w') as f:
                f.write("CAMERA CALIBRATION SUMMARY\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Number of cameras: {self.config.num_cameras}\n\n")
                
                f.write("INDIVIDUAL CALIBRATIONS:\n")
                f.write("-" * 40 + "\n")
                for cam_id, cal in self.individual_calibration_node.camera_calibrations.items():
                    if cal.camera_matrix is not None:
                        f.write(f"Camera {cam_id}:\n")
                        f.write(f"  Error: {cal.calibration_error:.4f} pixels\n")
                        f.write(f"  Coverage: {cal.coverage_percentage:.1f}%\n")
                        f.write(f"  Frames used: {cal.num_frames_used}\n")
                        f.write(f"  Full boards: {cal.full_board_detections}\n\n")
                
                f.write("MULTI-CAMERA CALIBRATION:\n")
                f.write("-" * 40 + "\n")
                if self.multi_camera_node.stereo_calibrations:
                    for (cam1, cam2), stereo_data in self.multi_camera_node.stereo_calibrations.items():
                        if stereo_data.rotation_matrix is not None:
                            f.write(f"Camera {cam1} ↔ Camera {cam2}:\n")
                            f.write(f"  Baseline: {stereo_data.baseline_distance:.2f} mm\n")
                            f.write(f"  Convergence: {stereo_data.convergence_angle:.2f}°\n")
                            f.write(f"  Error: {stereo_data.stereo_error:.4f} pixels\n\n")
            
            logger.info(f"📝 Saved calibration summary to {summary_path}")
            
        except Exception as e:
            logger.error(f"Failed to save calibration report: {e}")
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process frames through the calibration pipeline."""
        try:
            frames = data.get('frames', [])
            timestamp = data.get('timestamp')
            
            if not frames:
                return {'error': 'No frames provided'}
            
            self.frames_processed += 1
            
            # Check if pipeline is complete
            if self.pipeline_complete:
                return {
                    'frames': frames,
                    'phase': CalibrationPhase.COMPLETED.value,
                    'message': 'Calibration pipeline complete',
                    'calibration_directory': self.config.calibration_output_dir,
                    'pipeline_complete': True
                }
            
            # Process based on current phase
            if self.current_phase == CalibrationPhase.INDIVIDUAL_CALIBRATION:
                # Phase 1: Individual camera calibration
                result = await self.individual_calibration_node.process(data)
                
                # Check if individual calibration is complete
                if result.get('all_calibrations_complete'):
                    individual_calibrations = result.get('individual_calibrations', {})
                    
                    if len(individual_calibrations) >= 2:  # Need at least 2 cameras
                        # Transition to multi-camera phase
                        self.transition_to_multi_camera(individual_calibrations)
                        
                        # Process same frame through multi-camera node
                        multi_result = await self.multi_camera_node.process(data)
                        
                        return {
                            'frames': frames,
                            'phase': self.current_phase.value,
                            'individual_calibrations': individual_calibrations,
                            'multi_camera_data': multi_result,
                            'visualization': self.create_combined_visualization(frames, multi_result),
                            'message': 'Transitioned to multi-camera calibration phase'
                        }
                    else:
                        logger.error("Insufficient cameras calibrated for multi-camera phase")
                        self.pipeline_complete = True
                        self.save_final_calibration_report()
                        return {
                            'frames': frames,
                            'phase': CalibrationPhase.COMPLETED.value,
                            'error': 'Insufficient cameras calibrated',
                            'pipeline_complete': True
                        }
                
                # Still in individual calibration phase
                return {
                    'frames': frames,
                    'phase': self.current_phase.value,
                    'individual_status': result.get('status'),
                    'active_camera': result.get('active_camera'),
                    'visualization': result.get('visualization'),
                    'message': result.get('message')
                }
            
            elif self.current_phase == CalibrationPhase.MULTI_CAMERA_CALIBRATION:
                # Phase 2: Multi-camera calibration
                result = await self.multi_camera_node.process(data)
                
                # Check if multi-camera calibration is complete
                if (result.get('calibration_status', {}).get('calibrated') and 
                    self.multi_camera_node.stereo_calibration_performed):
                    
                    self.current_phase = CalibrationPhase.COMPLETED
                    self.pipeline_complete = True
                    
                    # Save final calibration report
                    self.save_final_calibration_report()
                    
                    logger.info("=" * 60)
                    logger.info("🎉 FULL CALIBRATION PIPELINE COMPLETE!")
                    logger.info(f"📁 Results saved to: {self.config.calibration_output_dir}")
                    logger.info("=" * 60)
                    
                    return {
                        'frames': frames,
                        'phase': self.current_phase.value,
                        'warped_frames': result.get('warped_frames'),
                        'combined_view': result.get('combined_view'),
                        'homographies': result.get('homographies'),
                        'calibration_directory': self.config.calibration_output_dir,
                        'visualization': self.create_combined_visualization(frames, result),
                        'pipeline_complete': True,
                        'message': 'Full calibration complete!'
                    }
                
                # Still in multi-camera calibration phase
                return {
                    'frames': frames,
                    'phase': self.current_phase.value,
                    'warped_frames': result.get('warped_frames'),
                    'combined_view': result.get('combined_view'),
                    'calibration_status': result.get('calibration_status'),
                    'visualization': self.create_combined_visualization(frames, result),
                    'message': f"Multi-camera calibration: {result.get('calibration_status', {}).get('frames_collected', 0)}/{self.config.multi_camera_frames} frames"
                }
            
        except Exception as e:
            logger.error(f"Error in calibration pipeline: {e}")
            import traceback
            traceback.print_exc()
            return {
                'error': str(e),
                'frames': frames,
                'phase': self.current_phase.value
            }
    
    def reset_pipeline(self):
        """Reset the entire calibration pipeline."""
        self.individual_calibration_node.reset_calibration()
        self.multi_camera_node.reset_calibration()
        
        self.current_phase = CalibrationPhase.INDIVIDUAL_CALIBRATION
        self.individual_calibrations_loaded = False
        self.frames_processed = 0
        self.pipeline_complete = False
        
        logger.info("Reset full calibration pipeline")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current status of the calibration pipeline."""
        return {
            'current_phase': self.current_phase.value,
            'frames_processed': self.frames_processed,
            'pipeline_complete': self.pipeline_complete,
            'individual_calibration': self.individual_calibration_node.get_calibration_summary(),
            'multi_camera_calibration': self.multi_camera_node.get_calibration_summary() if self.individual_calibrations_loaded else None,
            'output_directory': self.config.calibration_output_dir
        }


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_pipeline():
        """Test the full calibration pipeline."""
        
        # Configure pipeline
        config = FullCalibrationConfig(
            num_cameras=4,
            individual_frames_per_camera=20,
            multi_camera_frames=15,
            calibration_output_dir="test_calibration_results"
        )
        
        # Create pipeline
        pipeline = FullCalibrationPipeline(config)
        
        # Simulate frames (in real usage, these would come from cameras)
        dummy_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(4)]
        
        # Process frames
        for i in range(100):  # Simulate 100 frame batches
            result = await pipeline.process({
                'frames': dummy_frames,
                'timestamp': i / 30.0  # 30 fps
            })
            
            print(f"Frame {i}: Phase={result.get('phase')}, Message={result.get('message')}")
            
            if result.get('pipeline_complete'):
                print("Calibration complete!")
                break
        
        # Get final status
        status = pipeline.get_pipeline_status()
        print(f"Final status: {status}")
    
    # Run test
    asyncio.run(test_pipeline())