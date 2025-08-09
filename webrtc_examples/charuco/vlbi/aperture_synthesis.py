"""
Aperture Synthesis Imaging

Advanced CLEAN algorithm implementation for interferometric image reconstruction.
Separated from the main visibility processor for modularity and future enhancement
with GPU acceleration and advanced deconvolution techniques.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import logging

logger = logging.getLogger(__name__)


class CLEANImager:
    """Advanced CLEAN algorithm for interferometric image reconstruction."""
    
    def __init__(self, image_size: int = 256, pixel_size_arcsec: float = 1.0):
        """
        Initialize CLEAN imager.
        
        Args:
            image_size: Output image size in pixels
            pixel_size_arcsec: Pixel size in arcseconds
        """
        self.image_size = image_size
        self.pixel_size_arcsec = pixel_size_arcsec
        
        # CLEAN parameters
        self.gain = 0.1               # Conservative gain factor
        self.threshold_factor = 3.0   # Threshold in units of noise sigma
        self.max_iterations = 5000    # Maximum iterations
        self.min_patch_fraction = 0.01 # Minimum fractional peak for CLEAN
        
        # Advanced parameters
        self.multiscale_scales = [0, 2, 5, 10]  # Multi-scale CLEAN scales
        self.use_multiscale = False
        self.auto_threshold = True
        
        logger.info(f"🖼️ Initialized CLEAN imager: {image_size}x{image_size} @ {pixel_size_arcsec}\"/px")
    
    def clean_deconvolution(self, dirty_image: np.ndarray, 
                          dirty_beam: np.ndarray,
                          noise_rms: Optional[float] = None) -> Dict:
        """
        Perform CLEAN deconvolution on dirty image.
        
        Args:
            dirty_image: Inverse FFT of visibility data
            dirty_beam: Point spread function from (u,v) sampling
            noise_rms: Noise level estimate (auto-computed if None)
            
        Returns:
            Dictionary with CLEAN results
        """
        
        if noise_rms is None:
            noise_rms = self._estimate_noise_level(dirty_image)
        
        threshold = self.threshold_factor * noise_rms
        
        logger.info(f"🧹 Starting CLEAN: noise={noise_rms:.6f}, threshold={threshold:.6f}")
        
        # Initialize
        residual = dirty_image.copy()
        clean_components = []
        
        # Find beam peak for normalization
        beam_peak_pos = np.unravel_index(np.argmax(np.abs(dirty_beam)), dirty_beam.shape)
        beam_peak_value = dirty_beam[beam_peak_pos]
        
        if abs(beam_peak_value) < 1e-10:
            logger.error("Dirty beam peak is essentially zero")
            return self._create_clean_result(dirty_image, [], residual, noise_rms)
        
        # CLEAN iterations
        for iteration in range(self.max_iterations):
            # Find peak in residual
            peak_pos = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)
            peak_value = residual[peak_pos]
            
            # Check stopping criteria
            if abs(peak_value) < threshold:
                logger.debug(f"CLEAN converged at iteration {iteration} (threshold)")
                break
                
            if abs(peak_value) < abs(dirty_image.max()) * self.min_patch_fraction:
                logger.debug(f"CLEAN converged at iteration {iteration} (min fraction)")
                break
            
            # Scale factor for this component
            component_flux = peak_value * self.gain
            
            # Record CLEAN component
            clean_components.append({
                'x': peak_pos[1],
                'y': peak_pos[0], 
                'flux': component_flux,
                'iteration': iteration
            })
            
            # Subtract scaled dirty beam from residual
            self._subtract_dirty_beam(
                residual, dirty_beam, peak_pos, component_flux, beam_peak_value
            )
            
            # Progress logging
            if iteration % 500 == 0 and iteration > 0:
                current_peak = np.max(np.abs(residual))
                logger.debug(f"CLEAN iteration {iteration}: peak residual = {current_peak:.6f}")
        
        # Create final restored image
        restored_image = self._restore_image(clean_components, residual, dirty_beam)
        
        logger.info(f"✨ CLEAN complete: {len(clean_components)} components in {iteration+1} iterations")
        
        return self._create_clean_result(
            dirty_image, clean_components, residual, noise_rms, restored_image
        )
    
    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """Estimate noise level using robust statistics."""
        
        # Use median absolute deviation for robust noise estimate
        # Avoid central region which may contain sources
        h, w = image.shape
        border = min(h, w) // 6
        
        if border > 5:
            noise_region = np.concatenate([
                image[:border, :].flatten(),           # Top
                image[-border:, :].flatten(),          # Bottom  
                image[border:-border, :border].flatten(),    # Left
                image[border:-border, -border:].flatten()    # Right
            ])
        else:
            # For small images, use outer 25%
            mask = np.ones_like(image, dtype=bool)
            center_h, center_w = h//2, w//2
            mask[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4] = False
            noise_region = image[mask]
        
        # Median absolute deviation scaled to standard deviation
        mad = np.median(np.abs(noise_region - np.median(noise_region)))
        noise_rms = 1.4826 * mad  # Scale factor for Gaussian distribution
        
        return noise_rms
    
    def _subtract_dirty_beam(self, residual: np.ndarray, dirty_beam: np.ndarray,
                           peak_pos: Tuple[int, int], component_flux: float,
                           beam_peak_value: float):
        """Subtract scaled dirty beam from residual at peak position."""
        
        # Normalize dirty beam by its peak
        normalized_beam = dirty_beam / beam_peak_value
        
        # Calculate beam placement
        beam_center = (dirty_beam.shape[0] // 2, dirty_beam.shape[1] // 2)
        
        # Calculate regions for subtraction
        res_h, res_w = residual.shape
        beam_h, beam_w = dirty_beam.shape
        
        # Residual region
        res_y_start = max(0, peak_pos[0] - beam_center[0])
        res_y_end = min(res_h, peak_pos[0] - beam_center[0] + beam_h)
        res_x_start = max(0, peak_pos[1] - beam_center[1])
        res_x_end = min(res_w, peak_pos[1] - beam_center[1] + beam_w)
        
        # Corresponding beam region
        beam_y_start = max(0, beam_center[0] - peak_pos[0])
        beam_y_end = beam_y_start + (res_y_end - res_y_start)
        beam_x_start = max(0, beam_center[1] - peak_pos[1])
        beam_x_end = beam_x_start + (res_x_end - res_x_start)
        
        # Ensure valid regions
        if (beam_y_end > beam_y_start and beam_x_end > beam_x_start and
            res_y_end > res_y_start and res_x_end > res_x_start):
            
            residual[res_y_start:res_y_end, res_x_start:res_x_end] -= \
                component_flux * normalized_beam[beam_y_start:beam_y_end, beam_x_start:beam_x_end]
    
    def _restore_image(self, clean_components: List[Dict], residual: np.ndarray,
                      dirty_beam: np.ndarray) -> np.ndarray:
        """Restore final image by convolving CLEAN components with clean beam."""
        
        # Create model image from CLEAN components
        model_image = np.zeros_like(residual)
        
        for comp in clean_components:
            x, y = int(comp['x']), int(comp['y'])
            if 0 <= y < model_image.shape[0] and 0 <= x < model_image.shape[1]:
                model_image[y, x] += comp['flux']
        
        # Create clean beam (Gaussian approximation of dirty beam)
        clean_beam = self._create_clean_beam(dirty_beam)
        
        # Convolve model with clean beam
        from scipy.ndimage import convolve
        convolved_model = convolve(model_image, clean_beam, mode='constant')
        
        # Add residual
        restored_image = convolved_model + residual
        
        return restored_image
    
    def _create_clean_beam(self, dirty_beam: np.ndarray) -> np.ndarray:
        """Create clean beam as Gaussian fit to dirty beam main lobe."""
        
        # Find dirty beam parameters
        beam_center = (dirty_beam.shape[0] // 2, dirty_beam.shape[1] // 2)
        beam_peak = dirty_beam[beam_center]
        
        if abs(beam_peak) < 1e-10:
            # Fallback: simple Gaussian
            sigma = 2.0
            clean_beam = self._gaussian_kernel(dirty_beam.shape, sigma)
            return clean_beam / np.sum(clean_beam)
        
        # Estimate FWHM by finding half-maximum points
        half_max = abs(beam_peak) / 2
        
        # Search along major axes
        center_y, center_x = beam_center
        
        # Horizontal FWHM
        h_fwhm = 2.0  # Default
        for dx in range(1, min(20, dirty_beam.shape[1] // 4)):
            if (center_x + dx < dirty_beam.shape[1] and 
                abs(dirty_beam[center_y, center_x + dx]) < half_max):
                h_fwhm = 2 * dx
                break
        
        # Vertical FWHM  
        v_fwhm = 2.0  # Default
        for dy in range(1, min(20, dirty_beam.shape[0] // 4)):
            if (center_y + dy < dirty_beam.shape[0] and
                abs(dirty_beam[center_y + dy, center_x]) < half_max):
                v_fwhm = 2 * dy
                break
        
        # Create Gaussian clean beam
        sigma_x = h_fwhm / (2 * np.sqrt(2 * np.log(2)))
        sigma_y = v_fwhm / (2 * np.sqrt(2 * np.log(2)))
        
        clean_beam = self._elliptical_gaussian_kernel(
            dirty_beam.shape, sigma_x, sigma_y, center=(center_y, center_x)
        )
        
        # Normalize
        clean_beam = clean_beam / np.sum(clean_beam)
        
        return clean_beam
    
    def _gaussian_kernel(self, shape: Tuple[int, int], sigma: float) -> np.ndarray:
        """Create 2D Gaussian kernel."""
        
        h, w = shape
        center_y, center_x = h // 2, w // 2
        
        y, x = np.ogrid[:h, :w]
        kernel = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
        
        return kernel
    
    def _elliptical_gaussian_kernel(self, shape: Tuple[int, int], 
                                  sigma_x: float, sigma_y: float,
                                  center: Tuple[int, int]) -> np.ndarray:
        """Create elliptical Gaussian kernel."""
        
        h, w = shape
        center_y, center_x = center
        
        y, x = np.ogrid[:h, :w]
        kernel = np.exp(-((x - center_x)**2 / (2 * sigma_x**2) + 
                         (y - center_y)**2 / (2 * sigma_y**2)))
        
        return kernel
    
    def _create_clean_result(self, dirty_image: np.ndarray, clean_components: List[Dict],
                           residual: np.ndarray, noise_rms: float,
                           restored_image: Optional[np.ndarray] = None) -> Dict:
        """Create standardized CLEAN result dictionary."""
        
        if restored_image is None:
            restored_image = dirty_image
        
        # Calculate statistics
        total_clean_flux = sum(comp['flux'] for comp in clean_components)
        peak_residual = np.max(np.abs(residual))
        dynamic_range = abs(dirty_image.max()) / noise_rms if noise_rms > 0 else 0
        
        return {
            'restored_image': restored_image,
            'residual_image': residual,
            'clean_components': clean_components,
            'dirty_image': dirty_image,
            'num_components': len(clean_components),
            'total_clean_flux': total_clean_flux,
            'peak_residual': peak_residual,
            'noise_rms': noise_rms,
            'dynamic_range': dynamic_range,
            'convergence_ratio': peak_residual / noise_rms if noise_rms > 0 else 0
        }


class MultiscaleCLEAN(CLEANImager):
    """Multi-scale CLEAN for extended sources."""
    
    def __init__(self, image_size: int = 256, pixel_size_arcsec: float = 1.0):
        super().__init__(image_size, pixel_size_arcsec)
        self.scales = [0, 2, 5, 10, 20]  # Scale sizes in pixels
        self.scale_bias = 0.6  # Bias toward smaller scales
    
    def multiscale_clean(self, dirty_image: np.ndarray, dirty_beam: np.ndarray,
                        noise_rms: Optional[float] = None) -> Dict:
        """Multi-scale CLEAN deconvolution."""
        
        # Create scale kernels
        scale_kernels = self._create_scale_kernels()
        
        # Initialize
        residual = dirty_image.copy()
        clean_components = []
        
        if noise_rms is None:
            noise_rms = self._estimate_noise_level(dirty_image)
        
        threshold = self.threshold_factor * noise_rms
        
        for iteration in range(self.max_iterations):
            # Find best scale and position
            best_scale, best_pos, best_value = self._find_multiscale_peak(
                residual, scale_kernels
            )
            
            if abs(best_value) < threshold:
                break
            
            # Add component
            component_flux = best_value * self.gain
            clean_components.append({
                'x': best_pos[1],
                'y': best_pos[0],
                'flux': component_flux,
                'scale': best_scale,
                'iteration': iteration
            })
            
            # Subtract from residual
            self._subtract_multiscale_component(
                residual, dirty_beam, best_pos, component_flux, 
                scale_kernels[best_scale]
            )
        
        # Restore with appropriate scale convolution
        restored_image = self._restore_multiscale_image(
            clean_components, residual, dirty_beam, scale_kernels
        )
        
        return self._create_clean_result(
            dirty_image, clean_components, residual, noise_rms, restored_image
        )
    
    def _create_scale_kernels(self) -> Dict[int, np.ndarray]:
        """Create convolution kernels for each scale."""
        
        kernels = {}
        
        for scale in self.scales:
            if scale == 0:
                # Point source (delta function)
                kernel = np.zeros((1, 1))
                kernel[0, 0] = 1.0
            else:
                # Gaussian kernel
                kernel_size = int(6 * scale) + 1
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                kernel = self._gaussian_kernel((kernel_size, kernel_size), scale)
                kernel = kernel / np.sum(kernel)
            
            kernels[scale] = kernel
        
        return kernels
    
    def _find_multiscale_peak(self, residual: np.ndarray, 
                            scale_kernels: Dict[int, np.ndarray]) -> Tuple[int, Tuple[int, int], float]:
        """Find the best scale and position for next CLEAN component."""
        
        best_scale = 0
        best_pos = (0, 0)
        best_value = 0
        
        for scale, kernel in scale_kernels.items():
            # Convolve residual with scale kernel
            if kernel.size == 1:
                convolved = residual
            else:
                from scipy.ndimage import convolve
                convolved = convolve(residual, kernel, mode='constant')
            
            # Find peak
            peak_pos = np.unravel_index(np.argmax(np.abs(convolved)), convolved.shape)
            peak_value = convolved[peak_pos]
            
            # Apply scale bias (favor smaller scales)
            biased_value = abs(peak_value) * (1 - self.scale_bias * scale / max(self.scales))
            
            if biased_value > abs(best_value):
                best_scale = scale
                best_pos = peak_pos
                best_value = peak_value
        
        return best_scale, best_pos, best_value
    
    def _subtract_multiscale_component(self, residual: np.ndarray, dirty_beam: np.ndarray,
                                     position: Tuple[int, int], flux: float,
                                     scale_kernel: np.ndarray):
        """Subtract multi-scale component from residual."""
        
        # For now, use same subtraction as regular CLEAN
        # In full implementation, would convolve dirty beam with scale kernel
        beam_center = (dirty_beam.shape[0] // 2, dirty_beam.shape[1] // 2)
        beam_peak = dirty_beam[beam_center]
        
        if abs(beam_peak) > 1e-10:
            self._subtract_dirty_beam(residual, dirty_beam, position, flux, beam_peak)
    
    def _restore_multiscale_image(self, clean_components: List[Dict], residual: np.ndarray,
                                dirty_beam: np.ndarray, scale_kernels: Dict[int, np.ndarray]) -> np.ndarray:
        """Restore image with scale-appropriate convolution."""
        
        # Group components by scale
        scale_models = {}
        for scale in self.scales:
            scale_models[scale] = np.zeros_like(residual)
        
        for comp in clean_components:
            scale = comp.get('scale', 0)
            x, y = int(comp['x']), int(comp['y'])
            if 0 <= y < residual.shape[0] and 0 <= x < residual.shape[1]:
                scale_models[scale][y, x] += comp['flux']
        
        # Convolve each scale with clean beam
        restored_image = residual.copy()
        clean_beam = self._create_clean_beam(dirty_beam)
        
        for scale, model in scale_models.items():
            if np.sum(np.abs(model)) > 0:
                # Convolve with clean beam
                from scipy.ndimage import convolve
                if scale == 0:
                    convolved_model = convolve(model, clean_beam, mode='constant')
                else:
                    # For extended scales, convolve with both clean beam and scale kernel
                    scale_kernel = scale_kernels[scale]
                    temp = convolve(model, scale_kernel, mode='constant')
                    convolved_model = convolve(temp, clean_beam, mode='constant')
                
                restored_image += convolved_model
        
        return restored_image