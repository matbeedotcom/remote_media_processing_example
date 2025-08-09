"""
Atmospheric Correction for Optical VLBI

Implements atmospheric phase calibration using reference stars to correct
for turbulence-induced phase delays across interferometric baselines.
Essential for achieving coherent interferometry over long baselines.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import logging
from scipy.interpolate import griddata
from scipy.optimize import minimize
import time

logger = logging.getLogger(__name__)


@dataclass
class ReferenceSource:
    """Reference star for atmospheric calibration."""
    
    source_id: str          # Unique identifier
    ra_deg: float          # Right ascension
    dec_deg: float         # Declination  
    magnitude: float       # Visual magnitude
    x_pixel: float         # Expected pixel position
    y_pixel: float         # Expected pixel position
    is_calibrator: bool = True  # Whether to use for calibration


@dataclass
class PhaseDelay:
    """Measured atmospheric phase delay."""
    
    baseline_id: str       # Baseline identifier
    source_id: str         # Reference source
    phase_delay_rad: float # Phase delay in radians
    timestamp: float       # Time of measurement
    uncertainty_rad: float # Measurement uncertainty
    elevation_deg: float   # Source elevation
    azimuth_deg: float     # Source azimuth


class AtmosphericPhaseScreen:
    """2D atmospheric phase screen model."""
    
    def __init__(self, size_m: float = 1000.0, resolution_m: float = 1.0,
                 r0_m: float = 0.15, wind_speed_ms: float = 10.0):
        """
        Initialize phase screen model.
        
        Args:
            size_m: Screen size in meters
            resolution_m: Screen resolution in meters  
            r0_m: Fried parameter (atmospheric coherence length)
            wind_speed_ms: Wind speed in m/s
        """
        self.size_m = size_m
        self.resolution_m = resolution_m
        self.r0_m = r0_m
        self.wind_speed_ms = wind_speed_ms
        
        # Create coordinate grids
        self.num_points = int(size_m / resolution_m)
        self.x_coords = np.linspace(-size_m/2, size_m/2, self.num_points)
        self.y_coords = np.linspace(-size_m/2, size_m/2, self.num_points)
        
        # Initialize phase screen
        self.phase_screen = self._generate_kolmogorov_screen()
        self.last_update_time = time.time()
        
        logger.info(f"🌪️ Initialized atmospheric phase screen: {size_m}m @ {resolution_m}m resolution")
    
    def _generate_kolmogorov_screen(self) -> np.ndarray:
        """Generate Kolmogorov turbulence phase screen."""
        
        # Spatial frequency grid
        kx = np.fft.fftfreq(self.num_points, self.resolution_m)
        ky = np.fft.fftfreq(self.num_points, self.resolution_m)
        kx_grid, ky_grid = np.meshgrid(kx, ky)
        k_squared = kx_grid**2 + ky_grid**2
        
        # Avoid division by zero at DC
        k_squared[0, 0] = 1e-10
        
        # Kolmogorov power spectrum: Φ(k) ∝ k^(-11/3)
        # Phase structure function constant
        cn2 = 1.0  # Simplified - should be based on atmospheric conditions
        
        # Power spectral density
        psd = 0.023 * (2*np.pi)**(-2) * cn2 * self.r0_m**(-5/3) * k_squared**(-11/6)
        
        # Generate random phases
        random_phases = np.random.normal(0, 1, (self.num_points, self.num_points)) + \
                       1j * np.random.normal(0, 1, (self.num_points, self.num_points))
        
        # Apply power spectrum
        fourier_screen = np.sqrt(psd * self.num_points**2 * self.resolution_m**2) * random_phases
        
        # Convert to spatial domain
        phase_screen = np.real(np.fft.ifft2(fourier_screen))
        
        return phase_screen
    
    def get_phase_at_position(self, x_m: float, y_m: float) -> float:
        """Get atmospheric phase at specific position."""
        
        # Interpolate phase screen
        from scipy.interpolate import interp2d
        
        # Handle boundaries
        x_m = np.clip(x_m, -self.size_m/2, self.size_m/2)
        y_m = np.clip(y_m, -self.size_m/2, self.size_m/2)
        
        # Simple bilinear interpolation
        x_idx = int((x_m + self.size_m/2) / self.resolution_m)
        y_idx = int((y_m + self.size_m/2) / self.resolution_m)
        
        x_idx = np.clip(x_idx, 0, self.num_points - 1)
        y_idx = np.clip(y_idx, 0, self.num_points - 1)
        
        return self.phase_screen[y_idx, x_idx]
    
    def update_screen(self, time_step_sec: float):
        """Update phase screen for temporal evolution."""
        
        # Simple wind drift model
        current_time = time.time()
        dt = current_time - self.last_update_time
        
        # Translate screen by wind
        shift_m = self.wind_speed_ms * dt
        shift_pixels = int(shift_m / self.resolution_m)
        
        if shift_pixels > 0:
            # Shift existing screen
            self.phase_screen = np.roll(self.phase_screen, shift_pixels, axis=1)
            
            # Generate new strip to replace shifted-out region
            new_strip = self._generate_kolmogorov_screen()[:, :shift_pixels]
            self.phase_screen[:, -shift_pixels:] = new_strip
            
            self.last_update_time = current_time


class AtmosphericCalibrator:
    """Atmospheric phase calibration using reference stars."""
    
    def __init__(self, baseline_vectors: List, reference_catalog: List[ReferenceSource]):
        """
        Initialize atmospheric calibrator.
        
        Args:
            baseline_vectors: Physical baseline vectors
            reference_catalog: List of reference stars for calibration
        """
        self.baselines = baseline_vectors
        self.reference_sources = reference_catalog
        
        # Calibration parameters
        self.min_elevation_deg = 20.0  # Minimum elevation for calibrators
        self.max_separation_deg = 5.0   # Maximum calibrator separation from target
        self.phase_unwrap_threshold = np.pi  # Phase unwrapping threshold
        
        # Phase screen models (one per baseline)
        self.phase_screens = {}
        for baseline in self.baselines:
            screen_id = f"{baseline.camera1_id}_{baseline.camera2_id}"
            self.phase_screens[screen_id] = AtmosphericPhaseScreen()
        
        logger.info(f"🔭 Initialized atmospheric calibrator with {len(reference_catalog)} reference sources")
    
    def calibrate_atmospheric_phase(self, visibility_data: List, 
                                  target_position: Tuple[float, float]) -> List:
        """
        Calibrate atmospheric phase using reference star measurements.
        
        Args:
            visibility_data: Raw complex visibility measurements
            target_position: (RA, Dec) of target object in degrees
            
        Returns:
            Phase-corrected visibility measurements
        """
        
        # Find suitable reference sources
        calibrators = self._select_calibrators(target_position)
        
        if not calibrators:
            logger.warning("No suitable calibrators found - returning uncorrected data")
            return visibility_data
        
        # Measure phase delays from calibrators
        phase_delays = self._measure_calibrator_phases(visibility_data, calibrators)
        
        # Model atmospheric phase screens
        phase_corrections = self._model_atmospheric_corrections(phase_delays, target_position)
        
        # Apply corrections to visibility data
        corrected_visibilities = self._apply_phase_corrections(visibility_data, phase_corrections)
        
        logger.info(f"✅ Applied atmospheric correction using {len(calibrators)} calibrators")
        
        return corrected_visibilities
    
    def _select_calibrators(self, target_position: Tuple[float, float]) -> List[ReferenceSource]:
        """Select suitable reference sources for calibration."""
        
        target_ra, target_dec = target_position
        suitable_calibrators = []
        
        for source in self.reference_sources:
            if not source.is_calibrator:
                continue
            
            # Check angular separation from target
            separation = self._angular_separation(
                target_ra, target_dec, source.ra_deg, source.dec_deg
            )
            
            if separation > self.max_separation_deg:
                continue
            
            # Check elevation (simplified - assumes zenith pointing)
            elevation = 90.0 - abs(source.dec_deg)  # Simplified elevation
            
            if elevation < self.min_elevation_deg:
                continue
            
            # Check brightness (need sufficient SNR for phase measurement)
            if source.magnitude > 12.0:  # Too faint
                continue
            
            suitable_calibrators.append(source)
        
        # Sort by brightness (prefer brighter calibrators)
        suitable_calibrators.sort(key=lambda s: s.magnitude)
        
        logger.debug(f"Selected {len(suitable_calibrators)} suitable calibrators")
        return suitable_calibrators[:5]  # Use up to 5 calibrators
    
    def _angular_separation(self, ra1: float, dec1: float, 
                          ra2: float, dec2: float) -> float:
        """Calculate angular separation between two sky positions."""
        
        # Convert to radians
        ra1_rad, dec1_rad = np.radians([ra1, dec1])
        ra2_rad, dec2_rad = np.radians([ra2, dec2])
        
        # Spherical law of cosines
        cos_sep = (np.sin(dec1_rad) * np.sin(dec2_rad) + 
                   np.cos(dec1_rad) * np.cos(dec2_rad) * np.cos(ra1_rad - ra2_rad))
        
        # Handle numerical errors
        cos_sep = np.clip(cos_sep, -1, 1)
        separation_rad = np.arccos(cos_sep)
        
        return np.degrees(separation_rad)
    
    def _measure_calibrator_phases(self, visibility_data: List, 
                                 calibrators: List[ReferenceSource]) -> List[PhaseDelay]:
        """Measure atmospheric phase delays from calibrator sources."""
        
        phase_delays = []
        
        for calibrator in calibrators:
            # Find visibilities near calibrator position
            # In real implementation, would extract sub-region around calibrator
            
            for visibility in visibility_data:
                # Simulate phase measurement from calibrator
                # In reality, would cross-correlate visibility data at calibrator position
                
                # For now, extract phase directly (simplified)
                measured_phase = np.angle(visibility.complex_value)
                
                # Estimate theoretical phase (geometric delay)
                geometric_phase = self._calculate_geometric_phase(
                    visibility.baseline, calibrator
                )
                
                # Atmospheric phase = measured - geometric
                atmospheric_phase = measured_phase - geometric_phase
                
                # Unwrap phase
                atmospheric_phase = self._unwrap_phase(atmospheric_phase)
                
                phase_delay = PhaseDelay(
                    baseline_id=f"{visibility.baseline.camera1_id}_{visibility.baseline.camera2_id}",
                    source_id=calibrator.source_id,
                    phase_delay_rad=atmospheric_phase,
                    timestamp=visibility.timestamp,
                    uncertainty_rad=0.1,  # Estimated uncertainty
                    elevation_deg=90.0 - abs(calibrator.dec_deg),  # Simplified
                    azimuth_deg=calibrator.ra_deg  # Simplified
                )
                
                phase_delays.append(phase_delay)
        
        return phase_delays
    
    def _calculate_geometric_phase(self, baseline, source: ReferenceSource) -> float:
        """Calculate expected geometric phase delay."""
        
        # Simplified geometric delay calculation
        # In reality, would account for Earth rotation, source position, etc.
        
        # Project baseline onto source direction (simplified)
        # This is a placeholder - real implementation needs proper coordinates
        
        baseline_length_m = baseline.baseline_length_mm / 1000.0
        wavelength_m = 550e-9  # Assume 550nm
        
        # Geometric phase proportional to baseline projected onto source direction
        geometric_phase = 2 * np.pi * baseline_length_m / wavelength_m
        
        return geometric_phase % (2 * np.pi)
    
    def _unwrap_phase(self, phase: float) -> float:
        """Unwrap phase to [-π, π] range."""
        
        while phase > np.pi:
            phase -= 2 * np.pi
        while phase < -np.pi:
            phase += 2 * np.pi
            
        return phase
    
    def _model_atmospheric_corrections(self, phase_delays: List[PhaseDelay],
                                     target_position: Tuple[float, float]) -> Dict:
        """Model atmospheric phase corrections from measured delays."""
        
        corrections = {}
        
        # Group delays by baseline
        baseline_delays = {}
        for delay in phase_delays:
            if delay.baseline_id not in baseline_delays:
                baseline_delays[delay.baseline_id] = []
            baseline_delays[delay.baseline_id].append(delay)
        
        # For each baseline, fit atmospheric model
        for baseline_id, delays in baseline_delays.items():
            if len(delays) < 2:
                # Need multiple calibrators for interpolation
                corrections[baseline_id] = 0.0
                continue
            
            # Extract positions and phases
            positions = []
            phases = []
            
            for delay in delays:
                # Find corresponding calibrator
                calibrator = next((c for c in self.reference_sources 
                                 if c.source_id == delay.source_id), None)
                if calibrator:
                    positions.append([calibrator.ra_deg, calibrator.dec_deg])
                    phases.append(delay.phase_delay_rad)
            
            if len(positions) < 2:
                corrections[baseline_id] = 0.0
                continue
            
            # Interpolate phase at target position
            positions = np.array(positions)
            phases = np.array(phases)
            
            try:
                # Simple linear interpolation (in practice, would use more sophisticated model)
                target_phase = griddata(
                    positions, phases, target_position, 
                    method='linear', fill_value=0.0
                )
                corrections[baseline_id] = float(target_phase) if not np.isnan(target_phase) else 0.0
                
            except Exception as e:
                logger.warning(f"Failed to interpolate phase for {baseline_id}: {e}")
                corrections[baseline_id] = 0.0
        
        return corrections
    
    def _apply_phase_corrections(self, visibility_data: List, 
                               phase_corrections: Dict) -> List:
        """Apply atmospheric phase corrections to visibility data."""
        
        corrected_visibilities = []
        
        for visibility in visibility_data:
            baseline_id = f"{visibility.baseline.camera1_id}_{visibility.baseline.camera2_id}"
            
            # Get correction for this baseline
            phase_correction = phase_corrections.get(baseline_id, 0.0)
            
            # Apply correction by rotating complex visibility
            correction_factor = np.exp(-1j * phase_correction)
            corrected_complex = visibility.complex_value * correction_factor
            
            # Create corrected visibility
            corrected_visibility = visibility
            corrected_visibility.amplitude = np.abs(corrected_complex)
            corrected_visibility.phase = np.angle(corrected_complex)
            
            corrected_visibilities.append(corrected_visibility)
        
        return corrected_visibilities


def create_reference_catalog(field_center_ra: float, field_center_dec: float,
                           field_size_deg: float = 1.0) -> List[ReferenceSource]:
    """Create reference star catalog for atmospheric calibration."""
    
    # In practice, would query astronomical catalogs (Gaia, Hipparcos, etc.)
    # For now, create simulated bright stars
    
    reference_sources = []
    
    # Add a few bright reference stars around the field
    num_references = 8
    for i in range(num_references):
        # Distribute around field edge
        angle = 2 * np.pi * i / num_references
        radius = field_size_deg * 0.4
        
        ra_offset = radius * np.cos(angle) / np.cos(np.radians(field_center_dec))
        dec_offset = radius * np.sin(angle)
        
        ra = field_center_ra + ra_offset
        dec = field_center_dec + dec_offset
        
        # Simulate pixel position (would use astrometry in practice)
        x_pixel = 640 + ra_offset * 3600 / 2.0  # Rough conversion
        y_pixel = 400 + dec_offset * 3600 / 2.0
        
        reference = ReferenceSource(
            source_id=f"REF_{i:02d}",
            ra_deg=ra,
            dec_deg=dec,
            magnitude=7.0 + np.random.uniform(0, 2),  # Magnitude 7-9
            x_pixel=x_pixel,
            y_pixel=y_pixel,
            is_calibrator=True
        )
        
        reference_sources.append(reference)
    
    logger.info(f"🌟 Created reference catalog with {len(reference_sources)} sources")
    
    return reference_sources


def main():
    """Test atmospheric calibration system."""
    
    # Create test reference catalog
    field_center_ra, field_center_dec = 280.0, -25.0
    reference_catalog = create_reference_catalog(field_center_ra, field_center_dec)
    
    print(f"Created reference catalog:")
    for ref in reference_catalog[:3]:  # Show first few
        print(f"  {ref.source_id}: RA={ref.ra_deg:.3f}°, Dec={ref.dec_deg:.3f}°, mag={ref.magnitude:.1f}")
    
    # Test phase screen
    screen = AtmosphericPhaseScreen(size_m=100.0, r0_m=0.15)
    
    # Sample phase at different positions
    positions = [(0, 0), (10, 0), (0, 10), (10, 10)]
    
    print(f"\nPhase screen samples:")
    for x, y in positions:
        phase = screen.get_phase_at_position(x, y)
        print(f"  Position ({x:2d}, {y:2d}): phase = {phase:.3f} rad")
    
    print(f"\n✅ Atmospheric correction system test complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()