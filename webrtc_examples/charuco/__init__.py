"""
ChAruco Camera Calibration System

A comprehensive multi-camera calibration system using ChAruco boards for precise 
camera intrinsic and extrinsic calibration.
"""

__version__ = "1.0.0"

# Import main components for easier access
from .calibration import (
    FullCalibrationPipeline,
    FullCalibrationConfig,
    IndividualCameraCalibrationNode,
    IndividualCalibrationConfig,
    MultiCameraCalibrationNode,
    MultiCameraConfig,
    CameraCalibrationData,
    StereoPairCalibration,
    CameraPose,
    CalibrationState,
    CalibrationPhase,
)

from .nodes import (
    CharucoDetectionNode,
    CharucoConfig,
    PoseResult,
    PoseDiversitySelectorNode,
    CalibrationFrame,
    PerspectiveWarpNode,
    WarpConfig,
)

from .utils import (
    SensorSpecifications,
    SensorDatabase,
    CameraSystemConfig,
)

__all__ = [
    # Version
    '__version__',
    
    # Calibration pipeline
    'FullCalibrationPipeline',
    'FullCalibrationConfig',
    'IndividualCameraCalibrationNode',
    'IndividualCalibrationConfig',
    'MultiCameraCalibrationNode',
    'MultiCameraConfig',
    'CameraCalibrationData',
    'StereoPairCalibration',
    'CameraPose',
    'CalibrationState',
    'CalibrationPhase',
    
    # Core nodes
    'CharucoDetectionNode',
    'CharucoConfig',
    'PoseResult',
    'PoseDiversitySelectorNode',
    'CalibrationFrame',
    'PerspectiveWarpNode',
    'WarpConfig',
    
    # Utils
    'SensorSpecifications',
    'SensorDatabase',
    'CameraSystemConfig',
]