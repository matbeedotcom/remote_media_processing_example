"""
Optical VLBI System

Very Long Baseline Interferometry implementation for the 4-camera array.
Provides complex visibility computation and aperture synthesis imaging
for sub-milliarcsecond angular resolution astrophotography.

Components:
- visibility_processor: Convert synchronized frames to complex visibilities
- aperture_synthesis: CLEAN algorithm for image reconstruction  
- atmospheric_correction: Phase calibration using reference stars
- fringe_tracker: Real-time fringe tracking and coherence monitoring
- test_data_generator: Simulated astronomical scenes for validation
"""

__version__ = "1.0.0"
__author__ = "OpticalVLBI-4K Project Team"

from .visibility_processor import (
    VisibilityProcessor, 
    ApertureSynthesisImager,
    BaselineVector,
    ComplexVisibility,
    create_baseline_vectors_from_calibration
)

from .test_data_generator import (
    VLBITestDataGenerator,
    PointSource,
    ExtendedSource,
    AstronomicalField
)

__all__ = [
    'VisibilityProcessor',
    'ApertureSynthesisImager', 
    'BaselineVector',
    'ComplexVisibility',
    'create_baseline_vectors_from_calibration',
    'VLBITestDataGenerator',
    'PointSource',
    'ExtendedSource',
    'AstronomicalField'
]