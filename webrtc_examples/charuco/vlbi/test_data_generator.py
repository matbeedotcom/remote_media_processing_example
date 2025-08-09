"""
VLBI Test Data Generator

Creates simulated astronomical scenes for testing the optical VLBI system.
Generates point sources, extended objects, and realistic noise patterns
to validate the visibility processing and aperture synthesis algorithms.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy.ndimage import gaussian_filter
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensor_config import SensorSpecifications, SensorDatabase
import json

logger = logging.getLogger(__name__)


@dataclass
class PointSource:
    """Simulated point source (star) in the field."""
    
    ra_deg: float          # Right ascension in degrees
    dec_deg: float         # Declination in degrees
    magnitude: float       # Visual magnitude
    x_pixel: float         # Pixel x coordinate
    y_pixel: float         # Pixel y coordinate
    flux_electrons: float  # Flux in electrons per second


@dataclass
class ExtendedSource:
    """Simulated extended object (galaxy, nebula)."""
    
    ra_deg: float          # Center RA
    dec_deg: float         # Center Dec
    magnitude: float       # Integrated magnitude
    size_arcsec: float     # Angular size in arcseconds
    x_pixel: float         # Center x pixel
    y_pixel: float         # Center y pixel
    flux_electrons: float  # Total flux in electrons per second
    shape: str = "gaussian"  # "gaussian", "exponential", "uniform"


@dataclass
class AstronomicalField:
    """Complete simulated astronomical field with sources and noise."""
    
    point_sources: List[PointSource]
    extended_sources: List[ExtendedSource]
    field_center_ra_deg: float
    field_center_dec_deg: float
    field_size_deg: float
    exposure_time_sec: float
    pixel_scale_arcsec: float


class VLBITestDataGenerator:
    """Generate realistic test data for VLBI algorithm validation."""
    
    def __init__(self, sensor_specs: SensorSpecifications, focal_length_mm: float = 2.8):
        """
        Initialize test data generator.
        
        Args:
            sensor_specs: Camera sensor specifications
            focal_length_mm: Lens focal length
        """
        self.sensor = sensor_specs
        self.focal_length_mm = focal_length_mm
        self.pixel_scale_arcsec = sensor_specs.plate_scale_arcsec_per_pixel(focal_length_mm)
        
        # Noise parameters
        self.read_noise_electrons = sensor_specs.read_noise_e if sensor_specs.read_noise_e else 3.0
        self.dark_current_e_per_s = sensor_specs.dark_current_e_per_s if sensor_specs.dark_current_e_per_s else 0.05
        self.sky_background_mag_per_arcsec2 = 21.5  # Typical dark sky
        
        logger.info(f"🌟 Initialized test data generator. Pixel scale: {self.pixel_scale_arcsec:.2f}\"/px")
    
    def create_simulated_field(self, field_name: str = "test_field") -> AstronomicalField:
        """Create a simulated astronomical field with realistic sources."""
        
        # Define field parameters
        field_center_ra = 280.0   # RA in Sagittarius (rich star field)
        field_center_dec = -25.0  # Dec towards galactic center
        field_size_deg = self.sensor.field_of_view_deg(self.focal_length_mm)[0]
        
        # Create point sources (stars)
        point_sources = self._generate_star_field(
            field_center_ra, field_center_dec, field_size_deg
        )
        
        # Add extended sources (galaxies)
        extended_sources = self._generate_extended_sources(
            field_center_ra, field_center_dec, field_size_deg
        )
        
        # Create test binary star system
        binary_star = self._create_binary_star_system(
            field_center_ra, field_center_dec, separation_arcsec=2.0
        )
        point_sources.extend(binary_star)
        
        field = AstronomicalField(
            point_sources=point_sources,
            extended_sources=extended_sources,
            field_center_ra_deg=field_center_ra,
            field_center_dec_deg=field_center_dec,
            field_size_deg=field_size_deg,
            exposure_time_sec=30.0,
            pixel_scale_arcsec=self.pixel_scale_arcsec
        )
        
        logger.info(f"📊 Created field '{field_name}': "
                   f"{len(point_sources)} stars, {len(extended_sources)} extended objects")
        
        return field
    
    def render_camera_frame(self, field: AstronomicalField, 
                           camera_id: int = 0,
                           add_noise: bool = True) -> np.ndarray:
        """
        Render astronomical field as seen by one camera.
        
        Args:
            field: Astronomical field definition
            camera_id: Camera identifier (affects slight pointing variations)
            add_noise: Whether to add realistic noise
            
        Returns:
            Rendered frame as uint16 array
        """
        
        # Create base image
        height, width = self.sensor.resolution_height, self.sensor.resolution_width
        image = np.zeros((height, width), dtype=np.float64)
        
        # Add sky background
        sky_background_electrons = self._calculate_sky_background_flux(field.exposure_time_sec)
        image[:] = sky_background_electrons
        
        # Render point sources
        for source in field.point_sources:
            self._render_point_source(image, source, field.exposure_time_sec, camera_id)
        
        # Render extended sources
        for source in field.extended_sources:
            self._render_extended_source(image, source, field.exposure_time_sec, camera_id)
        
        # Add noise if requested
        if add_noise:
            image = self._add_realistic_noise(image, field.exposure_time_sec)
        
        # Convert to ADU and clip
        adu_per_electron = 1.0  # Assume unity gain
        bit_depth = self.sensor.bit_depth
        max_adu = (2 ** bit_depth) - 1
        
        image_adu = image * adu_per_electron
        image_adu = np.clip(image_adu, 0, max_adu)
        
        return image_adu.astype(np.uint16)
    
    def create_quad_frame_set(self, field: AstronomicalField,
                             add_noise: bool = True) -> np.ndarray:
        """
        Create synchronized 4-camera frame set.
        
        Args:
            field: Astronomical field to render
            add_noise: Whether to add noise
            
        Returns:
            Quad frame array [height, width, 4]
        """
        
        height, width = self.sensor.resolution_height, self.sensor.resolution_width
        quad_frames = np.zeros((height, width, 4), dtype=np.uint16)
        
        for camera_id in range(4):
            quad_frames[:, :, camera_id] = self.render_camera_frame(
                field, camera_id, add_noise
            )
        
        logger.info(f"📹 Generated quad frame set: {quad_frames.shape}")
        return quad_frames
    
    def _generate_star_field(self, center_ra: float, center_dec: float, 
                           field_size_deg: float) -> List[PointSource]:
        """Generate realistic stellar field with proper magnitude distribution."""
        
        # Stellar density: roughly 3000 stars per square degree to mag 15
        # Scale by field size
        field_area_sq_deg = field_size_deg ** 2
        expected_stars = int(3000 * field_area_sq_deg)
        
        # Limit to reasonable number for testing
        num_stars = min(expected_stars, 500)
        
        stars = []
        
        for i in range(num_stars):
            # Random position in field
            ra_offset = (np.random.random() - 0.5) * field_size_deg
            dec_offset = (np.random.random() - 0.5) * field_size_deg
            
            ra = center_ra + ra_offset / np.cos(np.radians(center_dec))
            dec = center_dec + dec_offset
            
            # Convert to pixel coordinates
            x_pixel, y_pixel = self._sky_to_pixel(ra, dec, center_ra, center_dec)
            
            # Skip if outside sensor area
            if (x_pixel < 0 or x_pixel >= self.sensor.resolution_width or
                y_pixel < 0 or y_pixel >= self.sensor.resolution_height):
                continue
            
            # Generate magnitude with realistic distribution
            # More faint stars than bright ones
            magnitude = self._generate_stellar_magnitude()
            
            # Convert magnitude to electron flux
            flux_electrons = self._magnitude_to_flux(magnitude)
            
            star = PointSource(
                ra_deg=ra,
                dec_deg=dec,
                magnitude=magnitude,
                x_pixel=x_pixel,
                y_pixel=y_pixel,
                flux_electrons=flux_electrons
            )
            stars.append(star)
        
        return stars
    
    def _generate_extended_sources(self, center_ra: float, center_dec: float,
                                 field_size_deg: float) -> List[ExtendedSource]:
        """Generate a few extended sources (galaxies)."""
        
        # Add 2-3 faint galaxies
        sources = []
        num_galaxies = np.random.randint(2, 4)
        
        for i in range(num_galaxies):
            # Random position
            ra_offset = (np.random.random() - 0.5) * field_size_deg * 0.8  # Keep away from edges
            dec_offset = (np.random.random() - 0.5) * field_size_deg * 0.8
            
            ra = center_ra + ra_offset / np.cos(np.radians(center_dec))
            dec = center_dec + dec_offset
            
            # Convert to pixels
            x_pixel, y_pixel = self._sky_to_pixel(ra, dec, center_ra, center_dec)
            
            # Skip if outside sensor
            if (x_pixel < 50 or x_pixel >= self.sensor.resolution_width - 50 or
                y_pixel < 50 or y_pixel >= self.sensor.resolution_height - 50):
                continue
            
            # Galaxy properties
            magnitude = np.random.uniform(16.0, 19.0)  # Faint galaxies
            size_arcsec = np.random.uniform(5.0, 30.0)  # 5-30 arcsec
            flux_electrons = self._magnitude_to_flux(magnitude)
            
            galaxy = ExtendedSource(
                ra_deg=ra,
                dec_deg=dec,
                magnitude=magnitude,
                size_arcsec=size_arcsec,
                x_pixel=x_pixel,
                y_pixel=y_pixel,
                flux_electrons=flux_electrons,
                shape="gaussian"
            )
            sources.append(galaxy)
        
        return sources
    
    def _create_binary_star_system(self, center_ra: float, center_dec: float,
                                 separation_arcsec: float = 2.0) -> List[PointSource]:
        """Create a test binary star system with known separation."""
        
        # Place binary near field center
        ra_primary = center_ra + 0.001  # Slight offset from center
        dec_primary = center_dec + 0.001
        
        # Secondary star position
        position_angle_deg = 45.0  # Northeast
        ra_secondary = ra_primary + (separation_arcsec / 3600.0) * np.sin(np.radians(position_angle_deg)) / np.cos(np.radians(dec_primary))
        dec_secondary = dec_primary + (separation_arcsec / 3600.0) * np.cos(np.radians(position_angle_deg))
        
        # Convert to pixels
        x1, y1 = self._sky_to_pixel(ra_primary, dec_primary, center_ra, center_dec)
        x2, y2 = self._sky_to_pixel(ra_secondary, dec_secondary, center_ra, center_dec)
        
        # Create stars
        primary = PointSource(
            ra_deg=ra_primary,
            dec_deg=dec_primary,
            magnitude=8.0,  # Bright primary
            x_pixel=x1,
            y_pixel=y1,
            flux_electrons=self._magnitude_to_flux(8.0)
        )
        
        secondary = PointSource(
            ra_deg=ra_secondary,
            dec_deg=dec_secondary,
            magnitude=10.0,  # Fainter secondary
            x_pixel=x2,
            y_pixel=y2,
            flux_electrons=self._magnitude_to_flux(10.0)
        )
        
        logger.info(f"⭐ Created binary star: separation = {separation_arcsec:.1f}\", "
                   f"pixel separation = {np.sqrt((x2-x1)**2 + (y2-y1)**2):.1f}px")
        
        return [primary, secondary]
    
    def _sky_to_pixel(self, ra: float, dec: float, 
                     center_ra: float, center_dec: float) -> Tuple[float, float]:
        """Convert sky coordinates to pixel coordinates."""
        
        # Simple gnomonic projection for small fields
        delta_ra = (ra - center_ra) * np.cos(np.radians(center_dec))
        delta_dec = dec - center_dec
        
        # Convert degrees to arcseconds
        delta_ra_arcsec = delta_ra * 3600
        delta_dec_arcsec = delta_dec * 3600
        
        # Convert to pixels
        x_pixel = self.sensor.resolution_width / 2 + delta_ra_arcsec / self.pixel_scale_arcsec
        y_pixel = self.sensor.resolution_height / 2 - delta_dec_arcsec / self.pixel_scale_arcsec  # Flip Y
        
        return x_pixel, y_pixel
    
    def _generate_stellar_magnitude(self) -> float:
        """Generate stellar magnitude with realistic distribution."""
        
        # More faint stars than bright ones
        # Roughly follows: N(m) ∝ 10^(0.6m) up to limiting magnitude
        
        # Limiting magnitude for our sensor and exposure
        limiting_mag = 15.0
        
        # Weighted random selection favoring fainter magnitudes
        magnitude = np.random.triangular(6.0, limiting_mag, limiting_mag)
        
        return magnitude
    
    def _magnitude_to_flux(self, magnitude: float) -> float:
        """Convert visual magnitude to electron flux (electrons/second)."""
        
        # Zero magnitude star flux (Vega): approximately 1000 electrons/sec/cm²
        # for typical CCD with V-band filter
        zero_mag_flux = 1000.0  # electrons/sec/cm²
        
        # Telescope aperture area (very small for our 2.8mm lens)
        aperture_area_cm2 = np.pi * (self.focal_length_mm / self.sensor.f_ratio / 2 / 10) ** 2
        
        # Flux = zero_magnitude_flux * 10^(-0.4 * magnitude) * aperture_area * quantum_efficiency
        flux = (zero_mag_flux * 
                10**(-0.4 * magnitude) * 
                aperture_area_cm2 * 
                self.sensor.quantum_efficiency)
        
        return max(flux, 0.01)  # Minimum flux for detectability
    
    def _calculate_sky_background_flux(self, exposure_time_sec: float) -> float:
        """Calculate sky background flux in electrons."""
        
        # Sky background in electrons/pixel/second
        sky_flux_per_arcsec2 = self._magnitude_to_flux(self.sky_background_mag_per_arcsec2)
        
        # Convert to flux per pixel
        pixel_area_arcsec2 = self.pixel_scale_arcsec ** 2
        sky_flux_per_pixel_per_sec = sky_flux_per_arcsec2 * pixel_area_arcsec2
        
        # Total for exposure
        sky_background_electrons = sky_flux_per_pixel_per_sec * exposure_time_sec
        
        return sky_background_electrons
    
    def _render_point_source(self, image: np.ndarray, source: PointSource,
                           exposure_time_sec: float, camera_id: int = 0):
        """Render a point source onto the image array."""
        
        x, y = source.x_pixel, source.y_pixel
        
        # Add small random offset per camera to simulate slight misalignment
        camera_offset_pixels = 0.2
        x += np.random.uniform(-camera_offset_pixels, camera_offset_pixels)
        y += np.random.uniform(-camera_offset_pixels, camera_offset_pixels)
        
        # Total flux for exposure
        total_flux = source.flux_electrons * exposure_time_sec
        
        # Simple PSF: Gaussian with FWHM ~ 2-3 pixels (seeing limited)
        psf_fwhm_pixels = 2.5
        psf_sigma = psf_fwhm_pixels / (2 * np.sqrt(2 * np.log(2)))
        
        # Create small PSF kernel
        kernel_size = int(6 * psf_sigma) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        kernel_center = kernel_size // 2
        y_indices, x_indices = np.ogrid[:kernel_size, :kernel_size]
        
        # Gaussian PSF
        psf_kernel = np.exp(-((x_indices - kernel_center)**2 + (y_indices - kernel_center)**2) / (2 * psf_sigma**2))
        psf_kernel = psf_kernel / np.sum(psf_kernel) * total_flux
        
        # Add to image
        x_int, y_int = int(x), int(y)
        x_start = x_int - kernel_center
        y_start = y_int - kernel_center
        
        # Handle boundaries
        for ky in range(kernel_size):
            for kx in range(kernel_size):
                img_y = y_start + ky
                img_x = x_start + kx
                
                if (0 <= img_y < image.shape[0] and 0 <= img_x < image.shape[1]):
                    image[img_y, img_x] += psf_kernel[ky, kx]
    
    def _render_extended_source(self, image: np.ndarray, source: ExtendedSource,
                              exposure_time_sec: float, camera_id: int = 0):
        """Render an extended source onto the image array."""
        
        x, y = source.x_pixel, source.y_pixel
        
        # Total flux
        total_flux = source.flux_electrons * exposure_time_sec
        
        # Size in pixels
        size_pixels = source.size_arcsec / self.pixel_scale_arcsec
        
        if source.shape == "gaussian":
            # Gaussian profile
            sigma_pixels = size_pixels / (2 * np.sqrt(2 * np.log(2)))  # FWHM to sigma
            
            # Create extended source kernel
            kernel_size = int(6 * sigma_pixels) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            kernel_center = kernel_size // 2
            y_indices, x_indices = np.ogrid[:kernel_size, :kernel_size]
            
            # Gaussian profile
            profile = np.exp(-((x_indices - kernel_center)**2 + (y_indices - kernel_center)**2) / (2 * sigma_pixels**2))
            profile = profile / np.sum(profile) * total_flux
            
            # Add to image
            x_int, y_int = int(x), int(y)
            x_start = x_int - kernel_center
            y_start = y_int - kernel_center
            
            # Handle boundaries
            for ky in range(kernel_size):
                for kx in range(kernel_size):
                    img_y = y_start + ky
                    img_x = x_start + kx
                    
                    if (0 <= img_y < image.shape[0] and 0 <= img_x < image.shape[1]):
                        image[img_y, img_x] += profile[ky, kx]
    
    def _add_realistic_noise(self, image: np.ndarray, exposure_time_sec: float) -> np.ndarray:
        """Add realistic noise sources to the image."""
        
        noisy_image = image.copy()
        
        # Add dark current
        dark_electrons = self.dark_current_e_per_s * exposure_time_sec
        noisy_image += dark_electrons
        
        # Add Poisson noise (photon shot noise)
        # Each pixel value represents electrons, so Poisson noise has variance = mean
        poisson_noise = np.random.poisson(np.maximum(noisy_image, 0)) - noisy_image
        noisy_image += poisson_noise
        
        # Add read noise (Gaussian)
        read_noise = np.random.normal(0, self.read_noise_electrons, image.shape)
        noisy_image += read_noise
        
        # Ensure non-negative
        noisy_image = np.maximum(noisy_image, 0)
        
        return noisy_image
    
    def save_test_dataset(self, dataset_name: str, num_frames: int = 10):
        """Generate and save a complete test dataset."""
        
        output_dir = f"test_data_{dataset_name}"
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Create astronomical field
        field = self.create_simulated_field(dataset_name)
        
        # Save field metadata
        field_metadata = {
            'field_name': dataset_name,
            'num_point_sources': len(field.point_sources),
            'num_extended_sources': len(field.extended_sources),
            'field_center_ra_deg': field.field_center_ra_deg,
            'field_center_dec_deg': field.field_center_dec_deg,
            'field_size_deg': field.field_size_deg,
            'exposure_time_sec': field.exposure_time_sec,
            'pixel_scale_arcsec': field.pixel_scale_arcsec,
            'sensor_specs': self.sensor.to_dict()
        }
        
        with open(f"{output_dir}/field_metadata.json", 'w') as f:
            json.dump(field_metadata, f, indent=2)
        
        # Generate and save frames
        for frame_id in range(num_frames):
            # Create quad frame
            quad_frame = self.create_quad_frame_set(field, add_noise=True)
            
            # Save as separate camera files
            for camera_id in range(4):
                filename = f"{output_dir}/frame_{frame_id:03d}_cam_{camera_id}.fits"
                
                # For now save as NPY (would use FITS in production)
                np.save(filename.replace('.fits', '.npy'), quad_frame[:, :, camera_id])
            
            # Also save combined quad frame
            quad_filename = f"{output_dir}/quad_frame_{frame_id:03d}.npy"
            np.save(quad_filename, quad_frame)
        
        logger.info(f"💾 Saved {num_frames} test frames to {output_dir}/")
        
        return output_dir


def main():
    """Generate test datasets for VLBI validation."""
    
    # Get sensor specifications
    sensor_db = SensorDatabase()
    sensor = sensor_db.get_sensor('OV9281')
    
    if not sensor:
        logger.error("OV9281 sensor not found in database")
        return
    
    # Create test data generator
    generator = VLBITestDataGenerator(sensor, focal_length_mm=2.8)
    
    # Generate several test datasets
    test_scenarios = [
        ("bright_stars", "Field with several bright stars"),
        ("binary_system", "Close binary star system"),
        ("faint_field", "Field with faint stars and galaxies"),
        ("crowded_field", "Dense stellar field")
    ]
    
    for scenario_name, description in test_scenarios:
        print(f"\n🎬 Generating '{scenario_name}': {description}")
        
        # Create field
        field = generator.create_simulated_field(scenario_name)
        
        # Render sample frame
        sample_frame = generator.render_camera_frame(field, camera_id=0, add_noise=True)
        
        # Show statistics
        print(f"  📊 Frame stats: min={sample_frame.min()}, max={sample_frame.max()}, "
              f"mean={sample_frame.mean():.1f}")
        
        print(f"  ⭐ Sources: {len(field.point_sources)} stars, "
              f"{len(field.extended_sources)} extended objects")
        
        # Save small test dataset
        generator.save_test_dataset(scenario_name, num_frames=5)
    
    print(f"\n✅ Test data generation complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()