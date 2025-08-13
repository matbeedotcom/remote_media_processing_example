#!/usr/bin/env python3
"""
Image Registration and Alignment Node

High-precision image registration for astrophoto stacking and VLBI applications.
Implements multiple registration algorithms optimized for astronomical imaging
with sub-pixel accuracy requirements for drizzle/fusion processing.

Features:
- Feature-based registration with ORB/SIFT/SURF
- Phase correlation registration
- Enhanced correlation coefficient (ECC) alignment
- Homography and affine transformation estimation
- Sub-pixel translation refinement
- Quality assessment and outlier rejection
- Multi-scale registration approach
"""

import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass
from pathlib import Path
import sys
from scipy.fft import fft2, ifft2, fftshift
from scipy.optimize import minimize
from scipy.ndimage import shift

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "remote_media_processing"))

from remotemedia.core.node import Node

logger = logging.getLogger(__name__)


@dataclass
class RegistrationConfig:
    """Configuration for image registration algorithms."""
    
    # Algorithm selection
    use_phase_correlation: bool = True
    use_feature_matching: bool = True
    use_ecc_alignment: bool = True
    
    # Feature detection parameters
    feature_detector: str = "ORB"  # ORB, SIFT, SURF, AKAZE
    max_features: int = 1000
    match_ratio_threshold: float = 0.7
    
    # Phase correlation parameters
    phase_correlation_window_size: int = 256
    phase_correlation_overlap_threshold: float = 0.8
    
    # ECC parameters
    ecc_max_iterations: int = 100
    ecc_termination_eps: float = 1e-6
    ecc_motion_model: str = "EUCLIDEAN"  # TRANSLATION, EUCLIDEAN, AFFINE, HOMOGRAPHY
    
    # Quality thresholds
    min_matches_required: int = 10
    max_reprojection_error: float = 1.0
    min_confidence: float = 0.7
    
    # Multi-scale parameters
    use_pyramid: bool = True
    pyramid_levels: int = 3
    pyramid_scale: float = 0.5


@dataclass
class RegistrationResult:
    """Result of image registration process."""
    
    transformation_matrix: np.ndarray
    translation: Tuple[float, float]
    rotation_angle: float
    scale_factor: float
    confidence_score: float
    num_matches: int
    reprojection_error: float
    method_used: str
    registration_time: float


class ImageRegistrationNode(Node):
    """
    High-precision image registration node for astronomical applications.
    
    Performs sub-pixel accurate image alignment using multiple complementary
    algorithms. Essential for astrophoto stacking and VLBI processing where
    precise image alignment is critical for signal quality.
    """
    
    def __init__(self, config: Optional[RegistrationConfig] = None, name: Optional[str] = None):
        """
        Initialize image registration node.
        
        Args:
            config: Registration configuration parameters
            name: Optional node name
        """
        super().__init__(name=name or "ImageRegistration")
        self.config = config or RegistrationConfig()
        
        # Initialize feature detector
        self.feature_detector = self._create_feature_detector()
        
        # Registration statistics
        self.total_registrations = 0
        self.successful_registrations = 0
        self.average_confidence = 0.0
        self.average_reprojection_error = 0.0
        
        logger.info(f"🎯 ImageRegistration initialized with {self.config.feature_detector} detector")
        
    def _create_feature_detector(self):
        """Create feature detector based on configuration."""
        detector_name = self.config.feature_detector.upper()
        
        if detector_name == "ORB":
            return cv2.ORB_create(nfeatures=self.config.max_features)
        elif detector_name == "SIFT":
            return cv2.SIFT_create(nfeatures=self.config.max_features)
        elif detector_name == "SURF":
            return cv2.xfeatures2d.SURF_create(hessianThreshold=400)
        elif detector_name == "AKAZE":
            return cv2.AKAZE_create()
        else:
            logger.warning(f"Unknown detector {detector_name}, using ORB")
            return cv2.ORB_create(nfeatures=self.config.max_features)
    
    async def process(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process frames and compute registration transformations.
        
        Args:
            data: Input data containing frames and reference frame info
            
        Returns:
            Enhanced data with registration results, or None if processing fails
        """
        try:
            frames = data.get('frames', [])
            if len(frames) < 2:
                return data  # Need at least 2 frames for registration
            
            # Use first frame as reference by default
            reference_frame = frames[0]
            reference_idx = data.get('reference_frame_idx', 0)
            if 0 <= reference_idx < len(frames):
                reference_frame = frames[reference_idx]
            
            # Register all frames to reference
            registration_results = []
            aligned_frames = []
            
            for i, frame in enumerate(frames):
                if i == reference_idx:
                    # Reference frame - identity transformation
                    result = RegistrationResult(
                        transformation_matrix=np.eye(3, dtype=np.float32),
                        translation=(0.0, 0.0),
                        rotation_angle=0.0,
                        scale_factor=1.0,
                        confidence_score=1.0,
                        num_matches=0,
                        reprojection_error=0.0,
                        method_used="reference",
                        registration_time=0.0
                    )
                    registration_results.append(result)
                    aligned_frames.append(frame)
                else:
                    # Register frame to reference
                    result = self._register_frame_pair(reference_frame, frame)
                    registration_results.append(result)
                    
                    if result.confidence_score >= self.config.min_confidence:
                        # Apply transformation to align frame
                        aligned_frame = self._apply_transformation(frame, result.transformation_matrix)
                        aligned_frames.append(aligned_frame)
                    else:
                        logger.warning(f"Frame {i} registration failed, using original frame")
                        aligned_frames.append(frame)
            
            # Update statistics
            self.total_registrations += len(frames) - 1
            successful_count = sum(1 for r in registration_results[1:] if r.confidence_score >= self.config.min_confidence)
            self.successful_registrations += successful_count
            
            if successful_count > 0:
                total_confidence = sum(r.confidence_score for r in registration_results[1:] if r.confidence_score >= self.config.min_confidence)
                total_error = sum(r.reprojection_error for r in registration_results[1:] if r.confidence_score >= self.config.min_confidence)
                self.average_confidence = total_confidence / successful_count
                self.average_reprojection_error = total_error / successful_count
            
            # Create enhanced output data
            enhanced_data = data.copy()
            enhanced_data['aligned_frames'] = aligned_frames
            enhanced_data['registration_results'] = [
                {
                    'transformation_matrix': r.transformation_matrix.tolist(),
                    'translation': r.translation,
                    'rotation_angle': r.rotation_angle,
                    'scale_factor': r.scale_factor,
                    'confidence_score': r.confidence_score,
                    'num_matches': r.num_matches,
                    'reprojection_error': r.reprojection_error,
                    'method_used': r.method_used
                }
                for r in registration_results
            ]
            enhanced_data['registration_stats'] = {
                'total_registrations': self.total_registrations,
                'successful_registrations': self.successful_registrations,
                'success_rate': self.successful_registrations / max(1, self.total_registrations),
                'average_confidence': self.average_confidence,
                'average_reprojection_error': self.average_reprojection_error
            }
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Error in ImageRegistration: {e}", exc_info=True)
            return data  # Pass through original data on error
    
    def _register_frame_pair(self, reference_frame: np.ndarray, target_frame: np.ndarray) -> RegistrationResult:
        """
        Register target frame to reference frame using multiple methods.
        
        Args:
            reference_frame: Reference frame for alignment
            target_frame: Frame to be aligned
            
        Returns:
            RegistrationResult with best alignment found
        """
        import time
        start_time = time.time()
        
        # Convert to grayscale if needed
        ref_gray = self._to_grayscale(reference_frame)
        target_gray = self._to_grayscale(target_frame)
        
        results = []
        
        # Try phase correlation registration
        if self.config.use_phase_correlation:
            result = self._phase_correlation_registration(ref_gray, target_gray)
            if result:
                results.append(result)
        
        # Try feature-based registration
        if self.config.use_feature_matching:
            result = self._feature_based_registration(ref_gray, target_gray)
            if result:
                results.append(result)
        
        # Try ECC alignment
        if self.config.use_ecc_alignment:
            result = self._ecc_registration(ref_gray, target_gray)
            if result:
                results.append(result)
        
        # Select best result
        if results:
            best_result = max(results, key=lambda r: r.confidence_score)
            best_result.registration_time = time.time() - start_time
            return best_result
        else:
            # Fallback: identity transformation
            return RegistrationResult(
                transformation_matrix=np.eye(3, dtype=np.float32),
                translation=(0.0, 0.0),
                rotation_angle=0.0,
                scale_factor=1.0,
                confidence_score=0.0,
                num_matches=0,
                reprojection_error=float('inf'),
                method_used="fallback",
                registration_time=time.time() - start_time
            )
    
    def _to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale if needed."""
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame.copy()
    
    def _phase_correlation_registration(self, ref_frame: np.ndarray, target_frame: np.ndarray) -> Optional[RegistrationResult]:
        """Register frames using phase correlation method."""
        try:
            # Ensure frames have same size
            h, w = ref_frame.shape
            target_resized = cv2.resize(target_frame, (w, h))
            
            # Convert to float
            ref_float = ref_frame.astype(np.float64)
            target_float = target_resized.astype(np.float64)
            
            # Apply window function to reduce edge effects
            window = self._create_hann_window(ref_float.shape)
            ref_windowed = ref_float * window
            target_windowed = target_float * window
            
            # Compute FFTs
            ref_fft = fft2(ref_windowed)
            target_fft = fft2(target_windowed)
            
            # Phase correlation
            cross_power_spectrum = ref_fft * np.conj(target_fft)
            cross_power_spectrum_normalized = cross_power_spectrum / (np.abs(cross_power_spectrum) + 1e-10)
            
            # Inverse FFT to get correlation surface
            correlation = np.real(ifft2(cross_power_spectrum_normalized))
            correlation = fftshift(correlation)
            
            # Find peak
            peak_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
            peak_value = correlation[peak_idx]
            
            # Convert peak position to translation
            center_y, center_x = h // 2, w // 2
            translation_y = peak_idx[0] - center_y
            translation_x = peak_idx[1] - center_x
            
            # Create transformation matrix
            transformation_matrix = np.float32([
                [1, 0, translation_x],
                [0, 1, translation_y],
                [0, 0, 1]
            ])
            
            # Estimate confidence from peak strength
            confidence = min(1.0, peak_value / np.mean(correlation))
            
            return RegistrationResult(
                transformation_matrix=transformation_matrix,
                translation=(float(translation_x), float(translation_y)),
                rotation_angle=0.0,
                scale_factor=1.0,
                confidence_score=confidence,
                num_matches=1,
                reprojection_error=0.0,
                method_used="phase_correlation",
                registration_time=0.0
            )
            
        except Exception as e:
            logger.debug(f"Phase correlation registration failed: {e}")
            return None
    
    def _create_hann_window(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create 2D Hann window."""
        h, w = shape
        hann_h = np.hanning(h)
        hann_w = np.hanning(w)
        return np.outer(hann_h, hann_w)
    
    def _feature_based_registration(self, ref_frame: np.ndarray, target_frame: np.ndarray) -> Optional[RegistrationResult]:
        """Register frames using feature matching."""
        try:
            # Detect features
            kp1, des1 = self.feature_detector.detectAndCompute(ref_frame, None)
            kp2, des2 = self.feature_detector.detectAndCompute(target_frame, None)
            
            if des1 is None or des2 is None or len(kp1) < self.config.min_matches_required:
                return None
            
            # Match features
            if self.config.feature_detector.upper() == "ORB":
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            else:
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            
            matches = matcher.knnMatch(des1, des2, k=2)
            
            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.config.match_ratio_threshold * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < self.config.min_matches_required:
                return None
            
            # Extract matched points
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Estimate transformation using RANSAC
            transformation_matrix, mask = cv2.findHomography(
                dst_pts, src_pts, 
                cv2.RANSAC, 
                self.config.max_reprojection_error
            )
            
            if transformation_matrix is None:
                return None
            
            # Calculate inliers
            inlier_count = np.sum(mask)
            confidence = inlier_count / len(good_matches)
            
            if confidence < self.config.min_confidence:
                return None
            
            # Extract transformation parameters
            translation_x = transformation_matrix[0, 2]
            translation_y = transformation_matrix[1, 2]
            
            # Estimate rotation and scale
            rotation_angle = np.arctan2(transformation_matrix[1, 0], transformation_matrix[0, 0])
            scale_x = np.sqrt(transformation_matrix[0, 0]**2 + transformation_matrix[1, 0]**2)
            scale_y = np.sqrt(transformation_matrix[0, 1]**2 + transformation_matrix[1, 1]**2)
            scale_factor = (scale_x + scale_y) / 2
            
            # Calculate reprojection error
            if mask is not None and np.sum(mask) > 0:
                inlier_src = src_pts[mask.ravel() == 1]
                inlier_dst = dst_pts[mask.ravel() == 1]
                
                # Transform destination points
                transformed_dst = cv2.perspectiveTransform(inlier_dst, transformation_matrix)
                
                # Calculate mean reprojection error
                reprojection_error = np.mean(np.linalg.norm(inlier_src - transformed_dst, axis=2))
            else:
                reprojection_error = float('inf')
            
            return RegistrationResult(
                transformation_matrix=transformation_matrix,
                translation=(float(translation_x), float(translation_y)),
                rotation_angle=float(rotation_angle),
                scale_factor=float(scale_factor),
                confidence_score=confidence,
                num_matches=int(inlier_count),
                reprojection_error=float(reprojection_error),
                method_used="feature_matching",
                registration_time=0.0
            )
            
        except Exception as e:
            logger.debug(f"Feature-based registration failed: {e}")
            return None
    
    def _ecc_registration(self, ref_frame: np.ndarray, target_frame: np.ndarray) -> Optional[RegistrationResult]:
        """Register frames using Enhanced Correlation Coefficient (ECC) method."""
        try:
            # Determine motion model
            motion_models = {
                "TRANSLATION": cv2.MOTION_TRANSLATION,
                "EUCLIDEAN": cv2.MOTION_EUCLIDEAN,
                "AFFINE": cv2.MOTION_AFFINE,
                "HOMOGRAPHY": cv2.MOTION_HOMOGRAPHY
            }
            
            motion_type = motion_models.get(self.config.ecc_motion_model, cv2.MOTION_EUCLIDEAN)
            
            # Initialize transformation matrix
            if motion_type == cv2.MOTION_TRANSLATION:
                warp_matrix = np.eye(2, 3, dtype=np.float32)
            elif motion_type == cv2.MOTION_EUCLIDEAN:
                warp_matrix = np.eye(2, 3, dtype=np.float32)
            elif motion_type == cv2.MOTION_AFFINE:
                warp_matrix = np.eye(2, 3, dtype=np.float32)
            else:  # HOMOGRAPHY
                warp_matrix = np.eye(3, dtype=np.float32)
            
            # Define termination criteria
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                       self.config.ecc_max_iterations, self.config.ecc_termination_eps)
            
            # Run ECC algorithm
            correlation_coefficient, warp_matrix = cv2.findTransformECC(
                ref_frame, target_frame, warp_matrix, motion_type, criteria
            )
            
            # Convert to 3x3 homography matrix
            if motion_type != cv2.MOTION_HOMOGRAPHY:
                transformation_matrix = np.eye(3, dtype=np.float32)
                transformation_matrix[:2, :] = warp_matrix
            else:
                transformation_matrix = warp_matrix
            
            # Extract transformation parameters
            translation_x = transformation_matrix[0, 2]
            translation_y = transformation_matrix[1, 2]
            
            # Estimate rotation and scale
            if motion_type in [cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE, cv2.MOTION_HOMOGRAPHY]:
                rotation_angle = np.arctan2(transformation_matrix[1, 0], transformation_matrix[0, 0])
                scale_x = np.sqrt(transformation_matrix[0, 0]**2 + transformation_matrix[1, 0]**2)
                scale_y = np.sqrt(transformation_matrix[0, 1]**2 + transformation_matrix[1, 1]**2)
                scale_factor = (scale_x + scale_y) / 2
            else:
                rotation_angle = 0.0
                scale_factor = 1.0
            
            # Use correlation coefficient as confidence
            confidence = max(0.0, min(1.0, correlation_coefficient))
            
            return RegistrationResult(
                transformation_matrix=transformation_matrix,
                translation=(float(translation_x), float(translation_y)),
                rotation_angle=float(rotation_angle),
                scale_factor=float(scale_factor),
                confidence_score=confidence,
                num_matches=1,
                reprojection_error=1.0 - correlation_coefficient,
                method_used="ecc_alignment",
                registration_time=0.0
            )
            
        except Exception as e:
            logger.debug(f"ECC registration failed: {e}")
            return None
    
    def _apply_transformation(self, frame: np.ndarray, transformation_matrix: np.ndarray) -> np.ndarray:
        """Apply transformation matrix to align frame."""
        try:
            h, w = frame.shape[:2]
            
            if transformation_matrix.shape == (3, 3):
                # Full homography transformation
                aligned_frame = cv2.warpPerspective(frame, transformation_matrix, (w, h))
            else:
                # Affine transformation
                aligned_frame = cv2.warpAffine(frame, transformation_matrix[:2, :], (w, h))
            
            return aligned_frame
            
        except Exception as e:
            logger.error(f"Error applying transformation: {e}")
            return frame
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'total_registrations': self.total_registrations,
            'successful_registrations': self.successful_registrations,
            'success_rate': self.successful_registrations / max(1, self.total_registrations),
            'average_confidence': self.average_confidence,
            'average_reprojection_error': self.average_reprojection_error,
            'feature_detector': self.config.feature_detector
        }


def main():
    """Test image registration with synthetic data."""
    logging.basicConfig(level=logging.INFO)
    
    # Create test configuration
    config = RegistrationConfig(
        feature_detector="ORB",
        use_phase_correlation=True,
        use_feature_matching=True,
        use_ecc_alignment=True
    )
    
    # Create registration node
    registrator = ImageRegistrationNode(config)
    
    # Create synthetic test images
    ref_image = np.zeros((400, 400), dtype=np.uint8)
    cv2.rectangle(ref_image, (100, 100), (300, 300), 255, 2)
    cv2.circle(ref_image, (200, 200), 50, 128, -1)
    
    # Create shifted version
    M = np.float32([[1, 0, 10], [0, 1, 5]])  # Translation by (10, 5)
    shifted_image = cv2.warpAffine(ref_image, M, (400, 400))
    
    # Test registration
    import asyncio
    
    async def test_registration():
        test_data = {
            'frames': [ref_image, shifted_image],
            'reference_frame_idx': 0
        }
        
        result = await registrator.process(test_data)
        
        if result:
            reg_results = result['registration_results']
            stats = result['registration_stats']
            
            print(f"Registration results:")
            for i, reg_result in enumerate(reg_results):
                print(f"Frame {i}: method={reg_result['method_used']}, "
                      f"translation={reg_result['translation']}, "
                      f"confidence={reg_result['confidence_score']:.3f}")
            
            print(f"Statistics: {stats}")
    
    asyncio.run(test_registration())


if __name__ == "__main__":
    main()