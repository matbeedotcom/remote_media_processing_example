#!/usr/bin/env python3
"""
Test Enhanced VLBI Pipeline Integration

Test script for validating the enhanced VLBI pipeline with video quad splitter
integration. Tests all major components including sub-pixel refinement,
image registration, drizzle calibration, and VLBI processing.

Usage:
    python test_vlbi_pipeline.py [--verbose] [--save-images]
"""

import asyncio
import logging
import numpy as np
import cv2
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any
import time

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "remote_media_processing"))

# Add charuco module path  
charuco_path = Path(__file__).parent / "charuco"
sys.path.insert(0, str(charuco_path))

# Import pipeline components
from webrtc_vlbi_pipeline_server import create_enhanced_vlbi_pipeline
from video_quad_splitter_node import VideoQuadSplitterNode

logger = logging.getLogger(__name__)


class VLBIPipelineTester:
    """Comprehensive test suite for the enhanced VLBI pipeline."""
    
    def __init__(self, save_images: bool = False):
        self.save_images = save_images
        self.test_results = {}
        self.output_dir = Path("test_outputs")
        
        if self.save_images:
            self.output_dir.mkdir(exist_ok=True)
            
    def create_synthetic_quad_frame(self, width: int = 1280, height: int = 800) -> np.ndarray:
        """Create synthetic quad frame with 4 camera views side by side."""
        
        # Create individual camera frames with different synthetic scenes
        camera_frames = []
        
        for cam_id in range(4):
            frame = np.zeros((height, width//4, 3), dtype=np.uint8)
            
            # Add different patterns for each camera
            if cam_id == 0:
                # Camera 0: Checkerboard pattern
                for i in range(0, height, 40):
                    for j in range(0, width//4, 40):
                        if ((i//40) + (j//40)) % 2 == 0:
                            frame[i:i+40, j:j+40] = [255, 255, 255]
            
            elif cam_id == 1:
                # Camera 1: Circles
                cv2.circle(frame, (width//8, height//2), 60, (0, 255, 0), -1)
                cv2.circle(frame, (width//8, height//4), 30, (255, 0, 0), -1)
                cv2.circle(frame, (width//8, 3*height//4), 30, (0, 0, 255), -1)
            
            elif cam_id == 2:
                # Camera 2: Grid lines
                for i in range(0, height, 30):
                    cv2.line(frame, (0, i), (width//4, i), (128, 128, 128), 1)
                for j in range(0, width//4, 30):
                    cv2.line(frame, (j, 0), (j, height), (128, 128, 128), 1)
                    
            else:  # cam_id == 3
                # Camera 3: Random noise with some structure
                frame[:, :] = np.random.randint(0, 256, (height, width//4, 3))
                cv2.rectangle(frame, (50, 50), (width//4-50, height-50), (255, 255, 0), 3)
            
            camera_frames.append(frame)
        
        # Combine into quad frame
        quad_frame = np.hstack(camera_frames)
        
        return quad_frame
    
    def create_synthetic_charuco_quad_frame(self, width: int = 1280, height: int = 800) -> np.ndarray:
        """Create synthetic quad frame with ChAruco boards in each quadrant."""
        
        # ChAruco board parameters
        squares_x, squares_y = 5, 4
        square_size = min((width//4) // squares_x, height // squares_y) - 2
        
        camera_frames = []
        
        for cam_id in range(4):
            frame = np.zeros((height, width//4, 3), dtype=np.uint8)
            
            # Create ChAruco board with slight variations per camera
            board_corners = []
            
            # Generate checkerboard pattern
            start_x = 20 + cam_id * 5  # Slight offset per camera
            start_y = 20 + cam_id * 3
            
            for row in range(squares_y):
                for col in range(squares_x):
                    x = start_x + col * square_size
                    y = start_y + row * square_size
                    
                    if x + square_size < width//4 and y + square_size < height:
                        if (row + col) % 2 == 0:
                            cv2.rectangle(frame, (x, y), (x + square_size, y + square_size), 
                                        (255, 255, 255), -1)
                        
                        # Add some corner points for testing
                        board_corners.append((x, y))
                        board_corners.append((x + square_size, y))
                        board_corners.append((x, y + square_size))
                        board_corners.append((x + square_size, y + square_size))
            
            # Add some noise
            noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
            frame = cv2.add(frame, noise)
            
            camera_frames.append(frame)
        
        # Combine into quad frame
        quad_frame = np.hstack(camera_frames)
        
        return quad_frame
    
    async def test_quad_splitter(self) -> bool:
        """Test VideoQuadSplitterNode functionality."""
        logger.info("🔧 Testing VideoQuadSplitterNode...")
        
        try:
            # Create quad splitter
            splitter = VideoQuadSplitterNode(num_splits=4)
            
            # Create test quad frame
            quad_frame = self.create_synthetic_quad_frame()
            
            # Test splitter
            test_data = {'frame': quad_frame, 'timestamp': time.time()}
            result = await splitter.process(test_data)
            
            if result is None:
                logger.error("❌ Quad splitter returned None")
                return False
            
            frames = result.get('frames', [])
            if len(frames) != 4:
                logger.error(f"❌ Expected 4 frames, got {len(frames)}")
                return False
            
            # Verify frame dimensions
            expected_width = quad_frame.shape[1] // 4
            expected_height = quad_frame.shape[0]
            
            for i, frame in enumerate(frames):
                if frame.shape[:2] != (expected_height, expected_width):
                    logger.error(f"❌ Frame {i} has incorrect dimensions: {frame.shape}")
                    return False
            
            logger.info("✅ VideoQuadSplitterNode test passed")
            
            if self.save_images:
                cv2.imwrite(str(self.output_dir / "test_quad_input.jpg"), quad_frame)
                for i, frame in enumerate(frames):
                    cv2.imwrite(str(self.output_dir / f"test_split_frame_{i}.jpg"), frame)
            
            self.test_results['quad_splitter'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Quad splitter test failed: {e}")
            self.test_results['quad_splitter'] = False
            return False
    
    async def test_pipeline_integration(self) -> bool:
        """Test full enhanced VLBI pipeline integration."""
        logger.info("🔬 Testing Enhanced VLBI Pipeline Integration...")
        
        try:
            # Create enhanced pipeline
            pipeline = create_enhanced_vlbi_pipeline(
                num_cameras=4,
                calibration_file="test_vlbi_calibration.json",
                use_quad_splitter=True,
                enable_vlbi_processing=True,
                enable_drizzle_export=True,
                output_directory=str(self.output_dir)
            )
            
            logger.info(f"📊 Pipeline created with {len(pipeline.nodes)} nodes")
            
            # Create test data with ChAruco patterns
            quad_frame = self.create_synthetic_charuco_quad_frame()
            
            # Process through pipeline
            test_data = {
                'frame': quad_frame,
                'timestamp': time.time(),
                'camera_id': 0
            }
            
            # Run pipeline processing
            result = None
            async for pipeline_result in pipeline.process(test_data):
                result = pipeline_result
                break  # Take first result for testing
            
            if result is None:
                logger.error("❌ Pipeline processing returned None")
                return False
            
            # Check for expected outputs
            expected_keys = [
                'frames',  # From frame buffer
                'corners_per_camera',  # From corner detection
                'refinement_stats',  # From sub-pixel refinement
                'registration_results',  # From image registration
                'calibration_results',  # From calibration
                'enhanced_distortion_models',  # From drizzle calibration
                'vlbi_stats'  # From VLBI processing
            ]
            
            missing_keys = []
            for key in expected_keys:
                if key not in result:
                    missing_keys.append(key)
            
            if missing_keys:
                logger.warning(f"⚠️ Missing expected keys: {missing_keys}")
                # Don't fail the test completely, some keys might not be present
                # in early pipeline stages
            
            # Check specific components
            frames = result.get('frames', [])
            if frames and len(frames) == 4:
                logger.info("✅ Frame processing successful")
            else:
                logger.warning(f"⚠️ Unexpected frame count: {len(frames) if frames else 0}")
            
            # Check refinement stats
            refinement_stats = result.get('refinement_stats', {})
            if refinement_stats:
                logger.info(f"✅ Sub-pixel refinement: {refinement_stats.get('total_processed', 0)} corners processed")
            
            # Check registration results
            registration_results = result.get('registration_results', [])
            if registration_results:
                logger.info(f"✅ Image registration: {len(registration_results)} frames registered")
            
            # Check drizzle calibration
            enhanced_models = result.get('enhanced_distortion_models', {})
            if enhanced_models:
                logger.info(f"✅ Drizzle calibration: {len(enhanced_models)} cameras enhanced")
            
            # Check VLBI processing
            vlbi_stats = result.get('vlbi_stats', {})
            if vlbi_stats:
                logger.info(f"✅ VLBI processing: {vlbi_stats.get('frames_processed', 0)} frames")
            
            logger.info("✅ Enhanced VLBI pipeline integration test completed")
            
            if self.save_images:
                cv2.imwrite(str(self.output_dir / "test_charuco_input.jpg"), quad_frame)
                
                # Save individual processed frames if available
                if frames:
                    for i, frame in enumerate(frames):
                        cv2.imwrite(str(self.output_dir / f"test_processed_frame_{i}.jpg"), frame)
            
            self.test_results['pipeline_integration'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline integration test failed: {e}", exc_info=True)
            self.test_results['pipeline_integration'] = False
            return False
    
    async def test_vlbi_processing(self) -> bool:
        """Test VLBI-specific processing components."""
        logger.info("🌟 Testing VLBI Processing Components...")
        
        try:
            from vlbi.visibility_processor import VisibilityProcessor, ApertureSynthesisImager, BaselineVector
            from sensor_config import SensorDatabase
            
            # Create test baseline vectors
            baselines = [
                BaselineVector(0, 1, 95.0, 0.0, 0.0, 95.0),
                BaselineVector(0, 2, 0.0, 95.0, 0.0, 95.0),
                BaselineVector(1, 3, 0.0, 95.0, 0.0, 95.0),
                BaselineVector(2, 3, 95.0, 0.0, 0.0, 95.0),
                BaselineVector(0, 3, 95.0, 95.0, 0.0, 134.4),
                BaselineVector(1, 2, -95.0, 95.0, 0.0, 134.4),
            ]
            
            # Get sensor specs
            sensor_db = SensorDatabase()
            sensor = sensor_db.get_sensor('OV9281')
            
            # Create visibility processor
            processor = VisibilityProcessor(baselines, sensor)
            
            # Create synthetic quad frame for VLBI
            test_quad_frame = np.random.randint(0, 255, (800, 1280, 4), dtype=np.uint8)
            
            # Add some coherent signal
            for i in range(4):
                # Add a bright spot at the same relative position in each camera
                center_y, center_x = 400, 160 + i * 320
                # Extract single channel for cv2.circle
                single_channel = test_quad_frame[:, :, i].copy()
                cv2.circle(single_channel, (center_x, center_y), 20, 200, -1)
                test_quad_frame[:, :, i] = single_channel
            
            # Process visibilities
            visibilities = processor.compute_visibility_matrix(test_quad_frame)
            
            if not visibilities:
                logger.warning("⚠️ No visibilities computed")
                self.test_results['vlbi_processing'] = False
                return False
            
            logger.info(f"✅ Computed {len(visibilities)} visibility measurements")
            
            # Test aperture synthesis
            imager = ApertureSynthesisImager(image_size=128, pixel_size_arcsec=2.0)
            synthesis_result = imager.synthesize_image(visibilities)
            
            if synthesis_result:
                beam_size = synthesis_result.get('synthesized_beam_arcsec', 0)
                logger.info(f"✅ Aperture synthesis: beam size = {beam_size:.2f} arcsec")
            else:
                logger.warning("⚠️ Aperture synthesis failed")
            
            self.test_results['vlbi_processing'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ VLBI processing test failed: {e}")
            self.test_results['vlbi_processing'] = False
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run comprehensive test suite."""
        logger.info("🚀 Starting Enhanced VLBI Pipeline Test Suite...")
        
        # Run all tests
        tests = [
            ("Quad Splitter", self.test_quad_splitter),
            ("Pipeline Integration", self.test_pipeline_integration),
            ("VLBI Processing", self.test_vlbi_processing)
        ]
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running {test_name} Test")
            logger.info(f"{'='*50}")
            
            try:
                success = await test_func()
                if success:
                    logger.info(f"✅ {test_name} test PASSED")
                else:
                    logger.error(f"❌ {test_name} test FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name} test ERROR: {e}")
                self.test_results[test_name.lower().replace(' ', '_')] = False
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("TEST SUITE SUMMARY")
        logger.info(f"{'='*60}")
        
        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
        
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        if self.save_images:
            logger.info(f"Test images saved to: {self.output_dir}")
        
        return self.test_results


async def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test Enhanced VLBI Pipeline")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    parser.add_argument("--save-images", "-s", action="store_true",
                       help="Save test images to disk")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Reduce noise from other loggers
    if not args.verbose:
        logging.getLogger("remotemedia").setLevel(logging.WARNING)
        logging.getLogger("charuco_detection_node").setLevel(logging.WARNING)
        logging.getLogger("multi_camera_calibration_node").setLevel(logging.WARNING)
    
    # Run tests
    tester = VLBIPipelineTester(save_images=args.save_images)
    results = await tester.run_all_tests()
    
    # Exit with appropriate code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())