"""
Raspberry Pi WebRTC Client Package

A Python-based WebRTC client package for Raspberry Pi that streams camera feeds
to the ChAruco calibration server. Supports multiple cameras and picamera2.
"""

from .client import RaspberryPiWebRTCClient
from .camera_manager import CameraManager, CameraInfo
from .video_track import CameraVideoTrack

__version__ = "1.0.0"
__author__ = "RemoteMedia Processing"
__description__ = "Raspberry Pi WebRTC Client for Multi-Camera ChAruco Calibration"

__all__ = [
    "RaspberryPiWebRTCClient",
    "CameraManager", 
    "CameraInfo",
    "CameraVideoTrack"
]