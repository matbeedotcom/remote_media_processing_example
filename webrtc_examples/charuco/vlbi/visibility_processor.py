"""
Optical VLBI Processing Engine

Core implementation of Very Long Baseline Interferometry for the 4-camera array.
Converts hardware-synchronized frames into complex visibility measurements and 
performs aperture synthesis imaging using the CLEAN algorithm.

This module implements the algorithms outlined in the Phase 2 project plan.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import logging
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.optimize import minimize_scalar
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensor_config import SensorSpecifications, SensorDatabase
import time

logger = logging.getLogger(__name__)


@dataclass
class BaselineVector:
    """Physical baseline between two cameras in the interferometric array."""
    
    camera1_id: int
    camera2_id: int
    baseline_x_mm: float  # East-West component
    baseline_y_mm: float  # North-South component  
    baseline_z_mm: float  # Vertical component
    baseline_length_mm: float  # Total baseline length
    
    def to_uv_coordinates(self, wavelength_nm: float = 550.0) -> Tuple[float, float]:
        """Convert baseline to (u,v) coordinates in units of wavelengths."""
        wavelength_mm = wavelength_nm * 1e-6  # Convert nm to mm
        u = self.baseline_x_mm / wavelength_mm
        v = self.baseline_y_mm / wavelength_mm
        return (u, v)


@dataclass 
class ComplexVisibility:
    """Complex visibility measurement from interferometric baseline."""
    
    baseline: BaselineVector
    amplitude: float      # Visibility amplitude
    phase: float         # Visibility phase (radians)
    u_coord: float       # u coordinate (wavelengths)
    v_coord: float       # v coordinate (wavelengths) 
    timestamp: float     # Time of measurement
    snr: float          # Signal-to-noise ratio
    
    @property
    def complex_value(self) -> complex:
        """Complex visibility as amplitude * exp(i * phase)."""
        return self.amplitude * np.exp(1j * self.phase)


class VisibilityProcessor:
    """Convert synchronized frames to interferometric visibilities."""
    
    def __init__(self, baseline_vectors: List[BaselineVector], 
                 sensor_specs: SensorSpecifications):
        """
        Initialize visibility processor.
        
        Args:
            baseline_vectors: Physical baselines between camera pairs
            sensor_specs: Sensor specifications for pixel-to-physical conversion
        """
        self.baselines = baseline_vectors
        self.sensor = sensor_specs
        self.correlation_window_size = 64  # Size of correlation window
        self.overlap_threshold = 0.1       # Minimum overlap fraction
        
        logger.info(f"🔬 Initialized VisibilityProcessor with {len(self.baselines)} baselines")
        
    def compute_visibility_matrix(self, quad_frames: np.ndarray) -> List[ComplexVisibility]:
        """
        Convert hardware-synchronized 4-camera frames to visibility measurements.
        
        Args:
            quad_frames: Hardware-synchronized frames [height, width, 4] from VideoQuadSplitterNode
            
        Returns:
            List of complex visibility measurements (6 baselines for 4 cameras)
        """
        timestamp = time.time()
        
        # Split quad frame into individual camera images
        individual_frames = self._split_quad_frame(quad_frames)
        
        if len(individual_frames) != 4:
            logger.error(f"Expected 4 camera frames, got {len(individual_frames)}")
            return []
        
        visibilities = []
        
        # Process all baseline pairs
        for baseline in self.baselines:
            cam1_id = baseline.camera1_id
            cam2_id = baseline.camera2_id
            
            if cam1_id >= len(individual_frames) or cam2_id >= len(individual_frames):
                logger.warning(f"Invalid camera IDs: {cam1_id}, {cam2_id}")
                continue
                
            frame1 = individual_frames[cam1_id]
            frame2 = individual_frames[cam2_id]
            
            # Compute complex visibility for this baseline
            visibility = self._compute_baseline_visibility(
                frame1, frame2, baseline, timestamp
            )
            
            if visibility:
                visibilities.append(visibility)
        
        logger.debug(f"📊 Computed {len(visibilities)} visibility measurements")
        return visibilities
    
    def _split_quad_frame(self, quad_frame: np.ndarray) -> List[np.ndarray]:
        """Split hardware-synchronized quad frame into individual camera frames."""
        
        if len(quad_frame.shape) == 3 and quad_frame.shape[2] == 4:
            # Format: [height, width, 4_cameras]
            return [quad_frame[:, :, i] for i in range(4)]
        elif len(quad_frame.shape) == 2:
            # Assume 2x2 quad layout: [2*height, 2*width]
            h, w = quad_frame.shape
            h_half, w_half = h // 2, w // 2
            
            frames = [
                quad_frame[:h_half, :w_half],          # Top-left
                quad_frame[:h_half, w_half:],          # Top-right  
                quad_frame[h_half:, :w_half],          # Bottom-left
                quad_frame[h_half:, w_half:]           # Bottom-right
            ]
            return frames
        else:
            logger.error(f"Unsupported quad frame format: {quad_frame.shape}")
            return []
    
    def _compute_baseline_visibility(self, frame1: np.ndarray, frame2: np.ndarray,
                                   baseline: BaselineVector, timestamp: float
                                   ) -> Optional[ComplexVisibility]:
        """Compute complex visibility for a single baseline pair."""
        
        try:
            # Find overlapping regions between the two frames
            overlap1, overlap2 = self._find_overlapping_regions(frame1, frame2, baseline)
            
            if overlap1 is None or overlap2 is None:
                logger.warning(f"No overlap found for baseline {baseline.camera1_id}-{baseline.camera2_id}")
                return None
            
            # Cross-correlate overlapping regions in Fourier domain
            cross_correlation = self._fourier_cross_correlation(overlap1, overlap2)
            
            # Apply geometric delay corrections
            corrected_correlation = self._apply_geometric_corrections(
                cross_correlation, baseline
            )
            
            # Extract amplitude and phase from peak correlation
            amplitude, phase = self._extract_visibility_from_correlation(corrected_correlation)
            
            # Calculate (u,v) coordinates
            u_coord, v_coord = baseline.to_uv_coordinates()
            
            # Estimate SNR from correlation peak
            snr = self._estimate_snr(corrected_correlation)
            
            visibility = ComplexVisibility(
                baseline=baseline,
                amplitude=amplitude,
                phase=phase,
                u_coord=u_coord,
                v_coord=v_coord,
                timestamp=timestamp,
                snr=snr
            )
            
            return visibility
            
        except Exception as e:
            logger.error(f"Error computing visibility for baseline {baseline.camera1_id}-{baseline.camera2_id}: {e}")
            return None
    
    def _find_overlapping_regions(self, frame1: np.ndarray, frame2: np.ndarray,
                                baseline: BaselineVector) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Find overlapping regions between two camera frames."""
        
        # For now, use simple center region extraction
        # In production, this should use calibration data to find precise overlaps
        
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        
        # Extract center regions of specified correlation window size
        center_h1, center_w1 = h1 // 2, w1 // 2
        center_h2, center_w2 = h2 // 2, w2 // 2
        
        half_window = self.correlation_window_size // 2
        
        # Ensure we don't exceed frame boundaries
        y1_start = max(0, center_h1 - half_window)
        y1_end = min(h1, center_h1 + half_window)
        x1_start = max(0, center_w1 - half_window)
        x1_end = min(w1, center_w1 + half_window)
        
        y2_start = max(0, center_h2 - half_window)
        y2_end = min(h2, center_h2 + half_window)
        x2_start = max(0, center_w2 - half_window) 
        x2_end = min(w2, center_w2 + half_window)
        
        overlap1 = frame1[y1_start:y1_end, x1_start:x1_end]
        overlap2 = frame2[y2_start:y2_end, x2_start:x2_end]
        
        # Ensure both regions have the same size
        min_h = min(overlap1.shape[0], overlap2.shape[0])
        min_w = min(overlap1.shape[1], overlap2.shape[1])
        
        overlap1 = overlap1[:min_h, :min_w]
        overlap2 = overlap2[:min_h, :min_w]
        
        if overlap1.size == 0 or overlap2.size == 0:
            return None, None
            
        return overlap1, overlap2
    
    def _fourier_cross_correlation(self, region1: np.ndarray, 
                                 region2: np.ndarray) -> np.ndarray:
        """Compute cross-correlation in Fourier domain."""
        
        # Convert to float and normalize
        r1 = region1.astype(np.float64)
        r2 = region2.astype(np.float64)
        
        # Remove DC component
        r1 = r1 - np.mean(r1)
        r2 = r2 - np.mean(r2)
        
        # Apply window function to reduce edge effects
        window = self._create_window_function(r1.shape)
        r1 = r1 * window
        r2 = r2 * window
        
        # Compute FFTs
        fft1 = fft2(r1)
        fft2 = fft2(r2)
        
        # Cross-correlation in frequency domain
        cross_spectrum = fft1 * np.conj(fft2)
        
        # Convert back to spatial domain
        cross_correlation = ifft2(cross_spectrum)
        
        # Shift zero frequency to center
        cross_correlation = fftshift(cross_correlation)
        
        return cross_correlation
    
    def _create_window_function(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create 2D Hann window function."""
        h, w = shape
        hann_h = np.hanning(h)
        hann_w = np.hanning(w)
        window_2d = np.outer(hann_h, hann_w)
        return window_2d
    
    def _apply_geometric_corrections(self, correlation: np.ndarray,
                                   baseline: BaselineVector) -> np.ndarray:
        """Apply geometric delay corrections based on baseline vector."""
        
        # For now, return correlation unchanged
        # In production, this should apply:
        # 1. Geometric delay corrections for Earth rotation
        # 2. Atmospheric delay corrections  
        # 3. Baseline projection corrections
        
        return correlation
    
    def _extract_visibility_from_correlation(self, correlation: np.ndarray) -> Tuple[float, float]:
        """Extract amplitude and phase from correlation peak."""
        
        # Find peak correlation
        peak_idx = np.unravel_index(np.argmax(np.abs(correlation)), correlation.shape)
        peak_value = correlation[peak_idx]
        
        # Extract amplitude and phase
        amplitude = np.abs(peak_value)
        phase = np.angle(peak_value)
        
        # Normalize amplitude by correlation window size
        amplitude = amplitude / (correlation.shape[0] * correlation.shape[1])
        
        return amplitude, phase
    
    def _estimate_snr(self, correlation: np.ndarray) -> float:
        """Estimate signal-to-noise ratio from correlation."""
        
        # Find peak value
        peak_value = np.max(np.abs(correlation))
        
        # Estimate noise from correlation values away from peak
        center_h, center_w = correlation.shape[0] // 2, correlation.shape[1] // 2
        noise_region = correlation.copy()
        
        # Mask out peak region
        mask_size = 5
        h_start = max(0, center_h - mask_size)
        h_end = min(correlation.shape[0], center_h + mask_size)
        w_start = max(0, center_w - mask_size)
        w_end = min(correlation.shape[1], center_w + mask_size)
        
        noise_region[h_start:h_end, w_start:w_end] = 0
        
        noise_level = np.std(np.abs(noise_region[noise_region != 0]))
        
        if noise_level > 0:
            snr = peak_value / noise_level
        else:
            snr = 0.0
            
        return snr


class ApertureSynthesisImager:
    """CLEAN algorithm for interferometric image reconstruction."""
    
    def __init__(self, image_size: int = 256, pixel_size_arcsec: float = 1.0):
        """
        Initialize aperture synthesis imager.
        
        Args:
            image_size: Output image size in pixels
            pixel_size_arcsec: Pixel size in arcseconds
        """
        self.image_size = image_size
        self.pixel_size_arcsec = pixel_size_arcsec
        self.gain = 0.1               # CLEAN gain parameter
        self.threshold_sigma = 3.0     # CLEAN threshold in sigma
        self.max_iterations = 1000     # Maximum CLEAN iterations
        
        logger.info(f"🖼️ Initialized ApertureSynthesisImager {image_size}x{image_size} pixels")
    
    def synthesize_image(self, visibilities: List[ComplexVisibility]) -> Dict:
        """
        Convert complex visibilities to high-resolution image using CLEAN algorithm.
        
        Args:
            visibilities: List of complex visibility measurements
            
        Returns:
            Dictionary containing:
            - dirty_image: Raw inverse FFT image  
            - dirty_beam: Point spread function
            - clean_image: Deconvolved image
            - clean_components: CLEAN component list
            - residual_image: Final residuals
        """
        
        if not visibilities:
            logger.error("No visibility data provided")
            return {}
        
        logger.info(f"🔄 Synthesizing image from {len(visibilities)} visibilities")
        
        # Grid visibilities in (u,v) plane
        uv_grid, weights_grid = self._grid_visibilities(visibilities)
        
        # Create dirty image via inverse FFT
        dirty_image = self._create_dirty_image(uv_grid)
        
        # Compute dirty beam from (u,v) sampling pattern
        dirty_beam = self._compute_dirty_beam(weights_grid)
        
        # Perform CLEAN deconvolution
        clean_components, residual_image = self._clean_deconvolution(
            dirty_image, dirty_beam
        )
        
        # Restore final image with clean beam
        clean_image = self._restore_clean_image(
            clean_components, residual_image, dirty_beam
        )
        
        result = {
            'dirty_image': dirty_image,
            'dirty_beam': dirty_beam, 
            'clean_image': clean_image,
            'clean_components': clean_components,
            'residual_image': residual_image,
            'num_visibilities': len(visibilities),
            'max_baseline_lambda': max(np.sqrt(v.u_coord**2 + v.v_coord**2) for v in visibilities),
            'synthesized_beam_arcsec': self._estimate_synthesized_beam_size(visibilities)
        }
        
        logger.info(f"✨ Image synthesis complete. Synthesized beam: {result['synthesized_beam_arcsec']:.2f} arcsec")
        
        return result
    
    def _grid_visibilities(self, visibilities: List[ComplexVisibility]) -> Tuple[np.ndarray, np.ndarray]:
        """Grid complex visibilities onto regular (u,v) grid."""
        
        # Create (u,v) coordinate grids
        max_u = max(abs(v.u_coord) for v in visibilities) * 1.1
        max_v = max(abs(v.v_coord) for v in visibilities) * 1.1
        
        grid_size = self.image_size
        u_coords = np.linspace(-max_u, max_u, grid_size)
        v_coords = np.linspace(-max_v, max_v, grid_size)
        
        # Initialize grids
        uv_grid = np.zeros((grid_size, grid_size), dtype=complex)
        weights_grid = np.zeros((grid_size, grid_size), dtype=float)
        
        # Grid each visibility
        for vis in visibilities:
            # Find nearest grid points
            u_idx = np.argmin(np.abs(u_coords - vis.u_coord))
            v_idx = np.argmin(np.abs(v_coords - vis.v_coord))
            
            # Weight by SNR
            weight = vis.snr if vis.snr > 0 else 1.0
            
            # Add visibility to grid
            uv_grid[v_idx, u_idx] += vis.complex_value * weight
            weights_grid[v_idx, u_idx] += weight
            
            # Add conjugate visibility for Hermitian symmetry
            if u_idx != grid_size // 2 or v_idx != grid_size // 2:
                conj_u_idx = grid_size - 1 - u_idx
                conj_v_idx = grid_size - 1 - v_idx
                uv_grid[conj_v_idx, conj_u_idx] += np.conj(vis.complex_value) * weight
                weights_grid[conj_v_idx, conj_u_idx] += weight
        
        # Normalize by weights
        mask = weights_grid > 0
        uv_grid[mask] = uv_grid[mask] / weights_grid[mask]
        
        return uv_grid, weights_grid
    
    def _create_dirty_image(self, uv_grid: np.ndarray) -> np.ndarray:
        """Create dirty image via inverse FFT of visibility grid."""
        
        # Inverse FFT to get dirty image
        dirty_complex = ifftshift(ifft2(fftshift(uv_grid)))
        
        # Take real part (image should be real for real sources)
        dirty_image = np.real(dirty_complex)
        
        # Normalize
        dirty_image = dirty_image / np.max(np.abs(dirty_image))
        
        return dirty_image
    
    def _compute_dirty_beam(self, weights_grid: np.ndarray) -> np.ndarray:
        """Compute dirty beam (point spread function) from sampling pattern."""
        
        # Create beam grid with same sampling pattern but unit weights
        beam_grid = (weights_grid > 0).astype(float)
        
        # Inverse FFT to get dirty beam
        beam_complex = ifftshift(ifft2(fftshift(beam_grid)))
        dirty_beam = np.real(beam_complex)
        
        # Normalize to peak of 1
        if np.max(np.abs(dirty_beam)) > 0:
            dirty_beam = dirty_beam / np.max(np.abs(dirty_beam))
        
        return dirty_beam
    
    def _clean_deconvolution(self, dirty_image: np.ndarray, 
                           dirty_beam: np.ndarray) -> Tuple[List, np.ndarray]:
        """Perform CLEAN deconvolution to find point sources."""
        
        clean_components = []
        residual = dirty_image.copy()
        
        # Estimate noise level for threshold
        noise_level = np.std(residual)
        threshold = self.threshold_sigma * noise_level
        
        logger.debug(f"CLEAN threshold: {threshold:.6f}, noise level: {noise_level:.6f}")
        
        for iteration in range(self.max_iterations):
            # Find peak in residual image
            peak_idx = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)
            peak_value = residual[peak_idx]
            
            # Check stopping criteria
            if abs(peak_value) < threshold:
                logger.debug(f"CLEAN converged at iteration {iteration}")
                break
            
            # Add CLEAN component
            clean_components.append({
                'position': peak_idx,
                'amplitude': peak_value * self.gain
            })
            
            # Subtract scaled dirty beam from residual
            self._subtract_dirty_beam(residual, dirty_beam, peak_idx, peak_value * self.gain)
        
        logger.info(f"🧹 CLEAN completed: {len(clean_components)} components, {iteration+1} iterations")
        
        return clean_components, residual
    
    def _subtract_dirty_beam(self, residual: np.ndarray, dirty_beam: np.ndarray,
                           peak_pos: Tuple[int, int], amplitude: float):
        """Subtract scaled dirty beam from residual image."""
        
        beam_center = (dirty_beam.shape[0] // 2, dirty_beam.shape[1] // 2)
        
        # Calculate offset from beam center to peak position
        offset_y = peak_pos[0] - beam_center[0]
        offset_x = peak_pos[1] - beam_center[1]
        
        # Determine overlapping region
        res_h, res_w = residual.shape
        beam_h, beam_w = dirty_beam.shape
        
        res_y_start = max(0, peak_pos[0] - beam_h // 2)
        res_y_end = min(res_h, peak_pos[0] + beam_h // 2 + 1)
        res_x_start = max(0, peak_pos[1] - beam_w // 2)
        res_x_end = min(res_w, peak_pos[1] + beam_w // 2 + 1)
        
        beam_y_start = res_y_start - (peak_pos[0] - beam_h // 2)
        beam_y_end = beam_y_start + (res_y_end - res_y_start)
        beam_x_start = res_x_start - (peak_pos[1] - beam_w // 2)
        beam_x_end = beam_x_start + (res_x_end - res_x_start)
        
        # Subtract scaled beam
        if beam_y_end > beam_y_start and beam_x_end > beam_x_start:
            residual[res_y_start:res_y_end, res_x_start:res_x_end] -= \
                amplitude * dirty_beam[beam_y_start:beam_y_end, beam_x_start:beam_x_end]
    
    def _restore_clean_image(self, clean_components: List, residual: np.ndarray,
                           dirty_beam: np.ndarray) -> np.ndarray:
        """Restore final clean image with Gaussian beam."""
        
        clean_image = residual.copy()
        
        # Create clean beam (Gaussian approximation)
        clean_beam_fwhm = self._estimate_clean_beam_fwhm(dirty_beam)
        
        # Add clean components convolved with clean beam
        for component in clean_components:
            pos = component['position']
            amplitude = component['amplitude']
            
            # For simplicity, just add point sources
            # In production, convolve with proper Gaussian beam
            clean_image[pos] += amplitude
        
        return clean_image
    
    def _estimate_clean_beam_fwhm(self, dirty_beam: np.ndarray) -> float:
        """Estimate FWHM of clean beam from dirty beam."""
        
        # Find FWHM of central lobe of dirty beam
        center = (dirty_beam.shape[0] // 2, dirty_beam.shape[1] // 2)
        center_value = dirty_beam[center]
        half_max = center_value / 2
        
        # Search radially outward for half-maximum
        for radius in range(1, min(dirty_beam.shape) // 4):
            # Sample points around circle
            angles = np.linspace(0, 2*np.pi, 16)
            values = []
            
            for angle in angles:
                y = int(center[0] + radius * np.sin(angle))
                x = int(center[1] + radius * np.cos(angle))
                if 0 <= y < dirty_beam.shape[0] and 0 <= x < dirty_beam.shape[1]:
                    values.append(dirty_beam[y, x])
            
            if values and np.mean(values) < half_max:
                return radius * 2  # FWHM = 2 * radius at half-max
        
        return 3.0  # Default fallback
    
    def _estimate_synthesized_beam_size(self, visibilities: List[ComplexVisibility]) -> float:
        """Estimate synthesized beam size in arcseconds."""
        
        if not visibilities:
            return 0.0
        
        # Find maximum baseline in wavelengths
        max_baseline = max(np.sqrt(v.u_coord**2 + v.v_coord**2) for v in visibilities)
        
        if max_baseline > 0:
            # Angular resolution ≈ λ/B = 1/B (in radians when B is in wavelengths)
            beam_size_rad = 1.0 / max_baseline
            beam_size_arcsec = np.degrees(beam_size_rad) * 3600
        else:
            beam_size_arcsec = 180.0  # Very large if no baselines
        
        return beam_size_arcsec


def create_baseline_vectors_from_calibration(stereo_calibrations: List) -> List[BaselineVector]:
    """Create baseline vectors from stereo calibration results."""
    
    baseline_vectors = []
    
    for i, stereo_cal in enumerate(stereo_calibrations):
        if stereo_cal.translation_vector is not None:
            # Convert from calibration units to physical units (mm)
            baseline_vector = BaselineVector(
                camera1_id=stereo_cal.camera1_id,
                camera2_id=stereo_cal.camera2_id,
                baseline_x_mm=stereo_cal.translation_vector[0] * 1000,  # Convert to mm
                baseline_y_mm=stereo_cal.translation_vector[1] * 1000,
                baseline_z_mm=stereo_cal.translation_vector[2] * 1000,
                baseline_length_mm=stereo_cal.baseline_distance * 1000 if stereo_cal.baseline_distance else 0
            )
            baseline_vectors.append(baseline_vector)
    
    return baseline_vectors


def main():
    """Test VLBI processing with simulated data."""
    
    # Create test baseline vectors
    baselines = [
        BaselineVector(0, 1, 95.0, 0.0, 0.0, 95.0),      # Horizontal baseline
        BaselineVector(0, 2, 0.0, 95.0, 0.0, 95.0),      # Vertical baseline  
        BaselineVector(1, 3, 0.0, 95.0, 0.0, 95.0),      # Vertical baseline
        BaselineVector(2, 3, 95.0, 0.0, 0.0, 95.0),      # Horizontal baseline
        BaselineVector(0, 3, 95.0, 95.0, 0.0, 134.4),    # Diagonal baseline
        BaselineVector(1, 2, -95.0, 95.0, 0.0, 134.4),   # Other diagonal
    ]
    
    # Get sensor specs
    sensor_db = SensorDatabase()
    sensor = sensor_db.get_sensor('OV9281')
    
    # Create processor
    processor = VisibilityProcessor(baselines, sensor)
    
    # Create test quad frame (simulated)
    test_quad_frame = np.random.randint(0, 255, (800, 1280, 4), dtype=np.uint8)
    
    # Process visibilities
    visibilities = processor.compute_visibility_matrix(test_quad_frame)
    
    print(f"Processed {len(visibilities)} visibility measurements")
    
    # Create imager and synthesize
    imager = ApertureSynthesisImager(image_size=128, pixel_size_arcsec=2.0)
    
    if visibilities:
        result = imager.synthesize_image(visibilities)
        print(f"Image synthesis result: {list(result.keys())}")
    else:
        print("No visibilities to process")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()