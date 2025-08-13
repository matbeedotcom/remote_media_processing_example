"""
Sensor Configuration System for Astrophotography

Provides physical sensor specifications for converting between pixel and real-world measurements.
Essential for astronomical plate scale calculations, field of view computations, and 
accurate multi-camera fusion algorithms.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import json
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class SensorSpecifications:
    """Physical specifications for camera sensors."""
    
    # Basic sensor info
    name: str                           # e.g., "IMX477", "IMX585", "ASI2600MC"
    manufacturer: str                   # e.g., "Sony", "ON Semiconductor"
    model: str                         # Full model designation
    
    # Physical dimensions
    sensor_width_mm: float             # Physical width in millimeters
    sensor_height_mm: float            # Physical height in millimeters
    
    # Pixel specifications
    pixel_width_um: float              # Pixel width in micrometers
    pixel_height_um: float             # Pixel height in micrometers  
    resolution_width: int              # Horizontal resolution in pixels
    resolution_height: int             # Vertical resolution in pixels
    
    # Computed dimensions (optional)
    diagonal_mm: Optional[float] = None # Diagonal size (computed if not specified)
    
    # Optical specifications
    quantum_efficiency: float = 0.7    # Peak QE (0-1)
    full_well_capacity: int = None     # Electrons per pixel
    read_noise_e: float = None         # Read noise in electrons
    dark_current_e_per_s: float = None # Dark current e-/pixel/second
    bit_depth: int = 12                # ADC bit depth
    
    # Color filter array
    bayer_pattern: str = None          # "RGGB", "GRBG", "GBRG", "BGGR", or None for mono
    is_color: bool = True              # Color vs monochrome
    
    # Additional metadata
    sensor_type: str = "CMOS"          # "CMOS" or "CCD"
    back_illuminated: bool = False     # BSI sensor
    
    def __post_init__(self):
        """Calculate derived properties."""
        # Calculate diagonal if not provided
        if self.diagonal_mm is None:
            self.diagonal_mm = np.sqrt(self.sensor_width_mm**2 + self.sensor_height_mm**2)
        
        # Validate pixel pitch matches sensor size
        calc_width = self.pixel_width_um * self.resolution_width / 1000
        calc_height = self.pixel_height_um * self.resolution_height / 1000
        
        width_error = abs(calc_width - self.sensor_width_mm)
        height_error = abs(calc_height - self.sensor_height_mm)
        
        if width_error > 0.1 or height_error > 0.1:
            logger.warning(f"Sensor {self.name}: Calculated size mismatch! "
                         f"Calculated: {calc_width:.2f}x{calc_height:.2f}mm, "
                         f"Specified: {self.sensor_width_mm:.2f}x{self.sensor_height_mm:.2f}mm")
    
    @property
    def pixel_pitch_um(self) -> float:
        """Average pixel pitch in micrometers."""
        return (self.pixel_width_um + self.pixel_height_um) / 2
    
    @property
    def pixel_area_um2(self) -> float:
        """Pixel area in square micrometers."""
        return self.pixel_width_um * self.pixel_height_um
    
    @property
    def aspect_ratio(self) -> float:
        """Sensor aspect ratio."""
        return self.resolution_width / self.resolution_height
    
    @property
    def crop_factor(self) -> float:
        """Crop factor relative to 35mm full frame (36x24mm)."""
        full_frame_diagonal = 43.27  # mm
        return full_frame_diagonal / self.diagonal_mm
    
    def plate_scale_arcsec_per_pixel(self, focal_length_mm: float) -> float:
        """
        Calculate plate scale in arcseconds per pixel.
        
        Args:
            focal_length_mm: Telescope/lens focal length in millimeters
            
        Returns:
            Plate scale in arcseconds per pixel
        """
        # Formula: plate_scale = 206265 * pixel_size_mm / focal_length_mm
        pixel_size_mm = self.pixel_pitch_um / 1000
        return 206265 * pixel_size_mm / focal_length_mm
    
    def field_of_view_deg(self, focal_length_mm: float) -> Tuple[float, float]:
        """
        Calculate field of view in degrees.
        
        Args:
            focal_length_mm: Telescope/lens focal length in millimeters
            
        Returns:
            (horizontal_fov, vertical_fov) in degrees
        """
        horizontal_fov = 2 * np.degrees(np.arctan(self.sensor_width_mm / (2 * focal_length_mm)))
        vertical_fov = 2 * np.degrees(np.arctan(self.sensor_height_mm / (2 * focal_length_mm)))
        return (horizontal_fov, vertical_fov)
    
    def sampling_ratio(self, focal_length_mm: float, seeing_arcsec: float = 2.0) -> float:
        """
        Calculate sampling ratio for astronomical imaging.
        
        Args:
            focal_length_mm: Telescope/lens focal length
            seeing_arcsec: Atmospheric seeing in arcseconds (default 2")
            
        Returns:
            Sampling ratio (>2 is oversampled, <2 is undersampled per Nyquist)
        """
        plate_scale = self.plate_scale_arcsec_per_pixel(focal_length_mm)
        return seeing_arcsec / plate_scale
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'manufacturer': self.manufacturer,
            'model': self.model,
            'sensor_width_mm': self.sensor_width_mm,
            'sensor_height_mm': self.sensor_height_mm,
            'diagonal_mm': self.diagonal_mm,
            'pixel_width_um': self.pixel_width_um,
            'pixel_height_um': self.pixel_height_um,
            'resolution_width': self.resolution_width,
            'resolution_height': self.resolution_height,
            'quantum_efficiency': self.quantum_efficiency,
            'full_well_capacity': self.full_well_capacity,
            'read_noise_e': self.read_noise_e,
            'dark_current_e_per_s': self.dark_current_e_per_s,
            'bit_depth': self.bit_depth,
            'bayer_pattern': self.bayer_pattern,
            'is_color': self.is_color,
            'sensor_type': self.sensor_type,
            'back_illuminated': self.back_illuminated
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SensorSpecifications':
        """Create from dictionary."""
        return cls(**data)


class SensorDatabase:
    """Database of common camera sensors for astrophotography."""
    
    def __init__(self):
        self.sensors: Dict[str, SensorSpecifications] = {}
        self._load_builtin_sensors()
    
    def _load_builtin_sensors(self):
        """Load built-in sensor definitions."""
        
        # Popular Raspberry Pi HQ Camera sensor
        self.sensors['IMX477'] = SensorSpecifications(
            name='IMX477',
            manufacturer='Sony',
            model='IMX477-AAQR',
            sensor_width_mm=7.857,
            sensor_height_mm=5.262,
            pixel_width_um=1.55,
            pixel_height_um=1.55,
            resolution_width=4056,
            resolution_height=3040,
            quantum_efficiency=0.84,
            full_well_capacity=12000,
            read_noise_e=2.8,
            dark_current_e_per_s=0.028,
            bit_depth=12,
            bayer_pattern='BGGR',
            is_color=True,
            sensor_type='CMOS',
            back_illuminated=True
        )
        
        # Popular astronomy camera sensor
        self.sensors['IMX585'] = SensorSpecifications(
            name='IMX585',
            manufacturer='Sony',
            model='IMX585',
            sensor_width_mm=12.98,
            sensor_height_mm=8.78,
            pixel_width_um=2.90,
            pixel_height_um=2.90,
            resolution_width=3856,
            resolution_height=2180,
            quantum_efficiency=0.91,
            full_well_capacity=50000,
            read_noise_e=1.0,
            dark_current_e_per_s=0.002,
            bit_depth=12,
            bayer_pattern='RGGB',
            is_color=True,
            sensor_type='CMOS',
            back_illuminated=True
        )
        
        # ASI2600MC Pro sensor (APS-C)
        self.sensors['IMX571'] = SensorSpecifications(
            name='IMX571',
            manufacturer='Sony',
            model='IMX571',
            sensor_width_mm=23.5,
            sensor_height_mm=15.7,
            pixel_width_um=3.76,
            pixel_height_um=3.76,
            resolution_width=6248,
            resolution_height=4176,
            quantum_efficiency=0.91,
            full_well_capacity=50000,
            read_noise_e=1.0,
            dark_current_e_per_s=0.0022,
            bit_depth=16,
            bayer_pattern='RGGB',
            is_color=True,
            sensor_type='CMOS',
            back_illuminated=True
        )
        
        # Full frame sensor (ASI6200MC Pro)
        self.sensors['IMX455'] = SensorSpecifications(
            name='IMX455',
            manufacturer='Sony',
            model='IMX455',
            sensor_width_mm=36.0,
            sensor_height_mm=24.0,
            pixel_width_um=3.76,
            pixel_height_um=3.76,
            resolution_width=9576,
            resolution_height=6388,
            quantum_efficiency=0.91,
            full_well_capacity=51000,
            read_noise_e=1.2,
            dark_current_e_per_s=0.003,
            bit_depth=16,
            bayer_pattern='RGGB',
            is_color=True,
            sensor_type='CMOS',
            back_illuminated=True
        )
        
        # Common webcam sensor (for testing)
        self.sensors['IMX219'] = SensorSpecifications(
            name='IMX219',
            manufacturer='Sony',
            model='IMX219PQ',
            sensor_width_mm=3.68,
            sensor_height_mm=2.76,
            pixel_width_um=1.12,
            pixel_height_um=1.12,
            resolution_width=3280,
            resolution_height=2464,
            quantum_efficiency=0.67,
            full_well_capacity=4500,
            read_noise_e=5.5,
            dark_current_e_per_s=0.1,
            bit_depth=10,
            bayer_pattern='BGGR',
            is_color=True,
            sensor_type='CMOS',
            back_illuminated=False
        )
        
        # Your specific OV9281 global shutter monochrome sensor (excellent for astrophotography)
        self.sensors['OV9281'] = SensorSpecifications(
            name='OV9281',
            manufacturer='OmniVision',
            model='OV9281',
            sensor_width_mm=3.84,  # 1280 x 3μm = 3.84mm
            sensor_height_mm=2.40,  # 800 x 3μm = 2.40mm
            diagonal_mm=4.52,      # 1/4" = 4.52mm diagonal
            pixel_width_um=3.0,
            pixel_height_um=3.0,
            resolution_width=1280,
            resolution_height=800,
            quantum_efficiency=0.75,  # Estimated for monochrome
            full_well_capacity=8000,  # Estimated for 3μm pixels
            read_noise_e=3.0,      # Global shutter typically lower noise
            dark_current_e_per_s=0.05,  # Good for 3μm pixels
            bit_depth=10,           # Typical for this sensor
            bayer_pattern=None,     # Monochrome sensor
            is_color=False,
            sensor_type='CMOS',
            back_illuminated=False
        )
        
        logger.info(f"📷 Loaded {len(self.sensors)} built-in sensor specifications")
    
    def add_sensor(self, sensor: SensorSpecifications):
        """Add a sensor to the database."""
        self.sensors[sensor.name] = sensor
        logger.info(f"Added sensor {sensor.name} to database")
    
    def get_sensor(self, name: str) -> Optional[SensorSpecifications]:
        """Get sensor by name."""
        return self.sensors.get(name)
    
    def list_sensors(self) -> List[str]:
        """List all available sensor names."""
        return list(self.sensors.keys())
    
    def save_to_file(self, filepath: str):
        """Save sensor database to JSON file."""
        data = {name: sensor.to_dict() for name, sensor in self.sensors.items()}
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(self.sensors)} sensors to {filepath}")
    
    def load_from_file(self, filepath: str):
        """Load sensor database from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for name, sensor_data in data.items():
            self.sensors[name] = SensorSpecifications.from_dict(sensor_data)
        
        logger.info(f"Loaded {len(data)} sensors from {filepath}")
    
    def find_sensor_by_resolution(self, width: int, height: int) -> Optional[str]:
        """Find sensor by resolution."""
        for name, sensor in self.sensors.items():
            if sensor.resolution_width == width and sensor.resolution_height == height:
                return name
        return None


@dataclass
class CameraSystemConfig:
    """Configuration for multi-camera astrophotography system."""
    
    # Camera array configuration
    num_cameras: int = 4
    sensor_name: str = "IMX477"  # Default to Raspberry Pi HQ camera
    
    # Optical system
    focal_length_mm: float = 50.0  # Lens/telescope focal length
    f_ratio: float = None          # f/ratio (computed from focal length and aperture)
    aperture_mm: float = None      # Aperture diameter in mm
    
    # Physical mounting
    camera_spacing_mm: float = 100.0  # Physical spacing between cameras
    mounting_pattern: str = "square"   # "linear", "square", "circular"
    
    # Environmental conditions
    typical_seeing_arcsec: float = 2.0  # Typical atmospheric seeing
    site_elevation_m: float = 0.0       # Observation site elevation
    
    # Processing parameters
    drizzle_scale: float = 0.5         # Drizzle output pixel scale (0.5 = 2x resolution)
    dither_pattern: str = "random"      # Dithering pattern for sub-pixel sampling
    
    def __post_init__(self):
        """Calculate derived properties."""
        if self.aperture_mm and not self.f_ratio:
            self.f_ratio = self.focal_length_mm / self.aperture_mm
        elif self.f_ratio and not self.aperture_mm:
            self.aperture_mm = self.focal_length_mm / self.f_ratio
    
    def get_sensor_specs(self, database: SensorDatabase) -> Optional[SensorSpecifications]:
        """Get sensor specifications from database."""
        return database.get_sensor(self.sensor_name)
    
    def calculate_system_fov(self, database: SensorDatabase) -> Optional[Tuple[float, float]]:
        """Calculate total system field of view."""
        sensor = self.get_sensor_specs(database)
        if not sensor:
            return None
        
        single_fov = sensor.field_of_view_deg(self.focal_length_mm)
        
        # Calculate combined FOV based on mounting pattern
        if self.mounting_pattern == "linear":
            total_h_fov = single_fov[0] * self.num_cameras
            total_v_fov = single_fov[1]
        elif self.mounting_pattern == "square":
            n = int(np.sqrt(self.num_cameras))
            total_h_fov = single_fov[0] * n
            total_v_fov = single_fov[1] * n
        else:  # circular or other
            total_h_fov = single_fov[0] * 2  # Approximate
            total_v_fov = single_fov[1] * 2
        
        return (total_h_fov, total_v_fov)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'num_cameras': self.num_cameras,
            'sensor_name': self.sensor_name,
            'focal_length_mm': self.focal_length_mm,
            'f_ratio': self.f_ratio,
            'aperture_mm': self.aperture_mm,
            'camera_spacing_mm': self.camera_spacing_mm,
            'mounting_pattern': self.mounting_pattern,
            'typical_seeing_arcsec': self.typical_seeing_arcsec,
            'site_elevation_m': self.site_elevation_m,
            'drizzle_scale': self.drizzle_scale,
            'dither_pattern': self.dither_pattern
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CameraSystemConfig':
        """Create from dictionary."""
        return cls(**data)


def calculate_optimal_focal_length(
    sensor: SensorSpecifications,
    target_sampling: float = 2.0,
    seeing_arcsec: float = 2.0
) -> float:
    """
    Calculate optimal focal length for a given sensor and seeing conditions.
    
    Args:
        sensor: Sensor specifications
        target_sampling: Target sampling ratio (2.0 = Nyquist sampling)
        seeing_arcsec: Expected seeing conditions
        
    Returns:
        Optimal focal length in mm
    """
    # From plate scale formula: f = 206265 * pixel_size_mm / (plate_scale_arcsec)
    # Where plate_scale = seeing / target_sampling
    
    pixel_size_mm = sensor.pixel_pitch_um / 1000
    target_plate_scale = seeing_arcsec / target_sampling
    optimal_focal_length = 206265 * pixel_size_mm / target_plate_scale
    
    return optimal_focal_length


def main():
    """Example usage and testing."""
    
    # Create sensor database
    db = SensorDatabase()
    
    # List available sensors
    print("Available sensors:")
    for name in db.list_sensors():
        sensor = db.get_sensor(name)
        print(f"  - {name}: {sensor.sensor_width_mm:.2f}x{sensor.sensor_height_mm:.2f}mm, "
              f"{sensor.resolution_width}x{sensor.resolution_height}px")
    
    # Example: Calculate parameters for IMX477 with 50mm lens
    sensor = db.get_sensor('IMX477')
    focal_length = 50.0  # mm
    
    print(f"\n{sensor.name} with {focal_length}mm lens:")
    print(f"  Plate scale: {sensor.plate_scale_arcsec_per_pixel(focal_length):.2f}\"/pixel")
    fov = sensor.field_of_view_deg(focal_length)
    print(f"  Field of view: {fov[0]:.2f}° x {fov[1]:.2f}°")
    print(f"  Sampling @ 2\" seeing: {sensor.sampling_ratio(focal_length, 2.0):.2f}x")
    
    # Calculate optimal focal length
    optimal_fl = calculate_optimal_focal_length(sensor, target_sampling=2.0, seeing_arcsec=2.0)
    print(f"  Optimal focal length for 2\" seeing: {optimal_fl:.0f}mm")
    
    # Create camera system configuration
    system = CameraSystemConfig(
        num_cameras=4,
        sensor_name='IMX477',
        focal_length_mm=50.0,
        f_ratio=2.8,
        camera_spacing_mm=100.0,
        mounting_pattern='square'
    )
    
    total_fov = system.calculate_system_fov(db)
    print(f"\n4-camera array total FOV: {total_fov[0]:.2f}° x {total_fov[1]:.2f}°")
    
    # Save configuration
    db.save_to_file('sensor_database.json')
    
    with open('camera_system_config.json', 'w') as f:
        json.dump(system.to_dict(), f, indent=2)
    print("\nSaved sensor database and system configuration")


if __name__ == "__main__":
    main()