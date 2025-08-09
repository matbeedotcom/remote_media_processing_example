#!/usr/bin/env python3
"""
WebRTC ChAruco Calibration Server

Real-time multi-camera ChAruco calibration and perspective warping server.
Processes synchronized video streams from multiple cameras to:
1. Detect ChAruco boards in real-time
2. Collect diverse calibration frames automatically
3. Calibrate cameras when sufficient frames are available
4. Provide real-time perspective-warped composite views

Features:
- Multi-camera ChAruco detection with pose estimation
- Intelligent pose diversity selection for robust calibration
- Real-time perspective warping and image alignment
- WebRTC streaming for low-latency video processing
- Persistent calibration storage and loading
- Web-based client interface

Usage:
    python webrtc_charuco_pipeline_server.py [options]
    python webrtc_charuco_pipeline_server.py --host 0.0.0.0 --port 8081 --cameras 4
    python webrtc_charuco_pipeline_server.py --calibration-file my_calibration.json
    SERVER_HOST=192.168.1.100 python webrtc_charuco_pipeline_server.py

Command line options:
    --host HOST              Server host address (default: 0.0.0.0)
    --port, -p PORT          Server port (default: 8081)
    --cameras, -c COUNT      Number of expected cameras (default: 4)
    --calibration-file, -f FILE  Calibration storage file (default: charuco_calibration.json)
    --output-width, -w WIDTH     Warped output width in pixels (default: 1920)
    --output-height, -H HEIGHT   Warped output height in pixels (default: 1080)

Environment variables (used as fallbacks):
    SERVER_HOST=hostname (default: 0.0.0.0) - Server host
    SERVER_PORT=port (default: 8081) - Server port  
    NUM_CAMERAS=count (default: 4) - Number of expected cameras
    CALIBRATION_FILE=path (default: charuco_calibration.json) - Calibration storage
    OUTPUT_WIDTH=pixels (default: 1920) - Warped output width
    OUTPUT_HEIGHT=pixels (default: 1080) - Warped output height
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
import cv2
import numpy as np
from video_stream_analyzer import VideoStreamAnalyzer

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "remote_media_processing"))

# Add charuco module path
charuco_path = Path(__file__).parent / "charuco"
sys.path.insert(0, str(charuco_path))

# Import RemoteMedia components
from remotemedia.core.pipeline import Pipeline
from remotemedia.core.node import Node
from remotemedia.webrtc import WebRTCServer, WebRTCConfig
from remotemedia.nodes import PassThroughNode

# Import ChAruco nodes
from charuco_detection_node import CharucoDetectionNode, CharucoConfig
from pose_diversity_selector_node import PoseDiversitySelectorNode
from perspective_warp_node import PerspectiveWarpNode, WarpConfig
from multi_camera_calibration_node import MultiCameraCalibrationNode, MultiCameraConfig
from video_quad_splitter_node import VideoQuadSplitterNode, VideoQuadMergerNode
from live_preview_node import LivePreviewNode, LivePreviewConfig
from desktop_preview_node import DesktopPreviewNode, DesktopPreviewConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce noise from most loggers - only keep charuco_detection_node at INFO level
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)
logging.getLogger("remotemedia.core.pipeline").setLevel(logging.WARNING)
logging.getLogger("remotemedia.webrtc.pipeline_processor").setLevel(logging.WARNING)

# Set all other nodes to WARNING level except charuco_detection_node and multi_camera_calibration_node
logging.getLogger("video_stream_analyzer").setLevel(logging.WARNING)
logging.getLogger("video_quad_splitter_node").setLevel(logging.WARNING) 
logging.getLogger("pose_diversity_selector_node").setLevel(logging.WARNING)
logging.getLogger("perspective_warp_node").setLevel(logging.INFO)
logging.getLogger("desktop_preview_node").setLevel(logging.INFO)  # Enable desktop preview debugging
logging.getLogger(__name__).setLevel(logging.WARNING)  # Main server logs
logging.getLogger("Pipeline").setLevel(logging.WARNING)

# Reduce charuco detection spam - only show multi-camera status from calibration node
logging.getLogger("charuco_detection_node").setLevel(logging.WARNING)
# logging.getLogger("multi_camera_calibration_node").setLevel(logging.INFO)  # Already INFO by default


class VideoFrameBuffer(Node):
    """
    Buffers and synchronizes video frames from multiple cameras.
    Collects frames until all cameras have provided a frame for the same timestamp.
    """
    
    def __init__(self, num_cameras: int = 4, name: Optional[str] = None):
        super().__init__(name=name or "VideoFrameBuffer")
        self.num_cameras = num_cameras
        self.frame_buffer: Dict[float, Dict[int, np.ndarray]] = {}
        self.max_buffer_size = 10  # Keep only recent frames
        self.frame_counter = 0
        self.processed_frames = 0
        
        logger.info(f"📹 FrameBuffer initialized for {num_cameras} cameras")
        
    async def process(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Buffer frames and output when all cameras have frames for a timestamp."""
        try:
            # Check if we received an array of frames (from VideoQuadSplitterNode)
            frames_array = data.get('frames')
            if frames_array is not None and isinstance(frames_array, list):
                # Handle array of frames from splitter
                timestamp = data.get('timestamp', self.frame_counter)
                self.frame_counter += 1
                
                # Debug logging for first few frames
                if self.frame_counter <= 5 or self.frame_counter % 60 == 0:
                    logger.info(f"📥 FrameBuffer received {len(frames_array)} frames from splitter "
                               f"(frame #{self.frame_counter})")
                
                # Add all frames to buffer with appropriate camera IDs
                if timestamp not in self.frame_buffer:
                    self.frame_buffer[timestamp] = {}
                
                for i, frame in enumerate(frames_array[:self.num_cameras]):
                    self.frame_buffer[timestamp][i] = frame
                
                # Log if we received more/fewer frames than expected
                if len(frames_array) != self.num_cameras:
                    logger.warning(f"⚠️ Expected {self.num_cameras} frames, got {len(frames_array)}")
            
            else:
                # Handle single frame (original behavior)
                frame = data.get('frame')
                camera_id = data.get('camera_id', 0)
                timestamp = data.get('timestamp', self.frame_counter)
                
                self.frame_counter += 1
                
                # Debug logging for first few frames
                if self.frame_counter <= 5 or self.frame_counter % 60 == 0:
                    logger.info(f"📥 FrameBuffer received frame #{self.frame_counter}: camera_id={camera_id}, "
                               f"timestamp={timestamp}, frame_shape={frame.shape if frame is not None else None}")
                
                if frame is None:
                    logger.warning("⚠️  FrameBuffer received None frame")
                    return None
                
                # Add frame to buffer
                if timestamp not in self.frame_buffer:
                    self.frame_buffer[timestamp] = {}
                
                self.frame_buffer[timestamp][camera_id] = frame
            
            # Check if we have frames from all cameras for this timestamp
            # OR if we only have 1 camera, pass frames through immediately
            buffer_ready = (len(self.frame_buffer[timestamp]) >= self.num_cameras or 
                           (self.num_cameras == 1 and len(self.frame_buffer[timestamp]) >= 1))
            
            if buffer_ready:
                self.processed_frames += 1
                
                # Extract synchronized frame set
                frames = []
                for cam_id in range(self.num_cameras):
                    if cam_id in self.frame_buffer[timestamp]:
                        frames.append(self.frame_buffer[timestamp][cam_id])
                    else:
                        # Use a blank frame if camera is missing
                        h, w = 480, 640  # Default frame size
                        if frames:
                            h, w = frames[0].shape[:2]
                        blank_frame = np.zeros((h, w, 3), dtype=np.uint8)
                        frames.append(blank_frame)
                
                # Log successful frame passing
                if self.processed_frames <= 5 or self.processed_frames % 60 == 0:
                    available_cameras = len(self.frame_buffer[timestamp])
                    logger.info(f"✅ FrameBuffer passing {len(frames)} frames to ChAruco pipeline "
                               f"(frame #{self.processed_frames}, {available_cameras}/{self.num_cameras} cameras available)")
                
                # Clean up old frames
                self._cleanup_buffer()
                
                return {
                    'frames': frames,
                    'timestamp': timestamp,
                    'num_cameras': self.num_cameras
                }
            
            # Clean up periodically
            if len(self.frame_buffer) > self.max_buffer_size:
                self._cleanup_buffer()
            
            return None
            
        except Exception as e:
            logger.error(f"Error in video frame buffer: {e}")
            return None
    
    def _cleanup_buffer(self):
        """Remove old frames to prevent memory buildup."""
        if len(self.frame_buffer) <= self.max_buffer_size:
            return
        
        # Keep only the most recent frames
        timestamps = sorted(self.frame_buffer.keys())
        for old_timestamp in timestamps[:-self.max_buffer_size]:
            del self.frame_buffer[old_timestamp]


class VideoOutputFormatter(Node):
    """
    Formats video output for WebRTC streaming.
    Converts calibration results into streamable video frames.
    Can switch between combined warped view and live preview mode.
    """
    
    def __init__(self, show_live_preview: bool = True, name: Optional[str] = None):
        super().__init__(name=name or "VideoOutputFormatter")
        self.frame_counter = 0
        self.show_live_preview = show_live_preview
        
    async def process(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Format calibration output for video streaming."""
        try:
            # Check if we have preview frame (from live preview mode)
            preview_frame = data.get('preview_frame')
            combined_view = data.get('combined_view')
            warped_frames = data.get('warped_frames', [])
            calibration_status = data.get('calibration_status', {})
            
            output_frame = None
            
            # Prefer live preview if available and enabled
            if self.show_live_preview and preview_frame is not None:
                output_frame = preview_frame
                self.frame_counter += 1
                
                return {
                    'frame': output_frame,
                    'timestamp': self.frame_counter,
                    'calibration_status': calibration_status,
                    'mode': 'live_preview'
                }
            
            # Fallback to combined warped view
            elif combined_view is not None:
                # Add calibration status overlay
                output_frame = self._add_status_overlay(combined_view, calibration_status)
                self.frame_counter += 1
                
                return {
                    'frame': output_frame,
                    'timestamp': self.frame_counter,
                    'calibration_status': calibration_status,
                    'warped_frames': warped_frames,
                    'mode': 'combined_view'
                }
            
            # If no frames available, create a status frame
            status_frame = self._create_status_frame(calibration_status)
            return {
                'frame': status_frame,
                'timestamp': self.frame_counter,
                'calibration_status': calibration_status,
                'mode': 'status_only'
            }
            
        except Exception as e:
            logger.error(f"Error in video output formatter: {e}")
            return None
    
    def _add_status_overlay(self, frame: np.ndarray, status: Dict[str, Any]) -> np.ndarray:
        """Add calibration status overlay to the frame."""
        overlay = frame.copy()
        
        # Status text
        calibrated = status.get('calibrated', False)
        frames_collected = status.get('frames_collected', 0)
        frames_needed = status.get('frames_needed', 5)
        cameras_calibrated = status.get('cameras_calibrated', 0)
        
        # Status colors
        color = (0, 255, 0) if calibrated else (0, 165, 255)  # Green if calibrated, orange if not
        
        # Add text overlays
        cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)  # Background
        cv2.rectangle(overlay, (10, 10), (350, 120), color, 2)  # Border
        
        # Status text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay, f"Calibration: {'COMPLETE' if calibrated else 'IN PROGRESS'}", 
                   (20, 35), font, 0.6, color, 2)
        cv2.putText(overlay, f"Frames: {frames_collected}/{frames_needed}", 
                   (20, 55), font, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay, f"Cameras: {cameras_calibrated}/4", 
                   (20, 75), font, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay, "ChAruco Multi-Camera Calibration", 
                   (20, 95), font, 0.4, (200, 200, 200), 1)
        
        return overlay
    
    def _create_status_frame(self, status: Dict[str, Any]) -> np.ndarray:
        """Create a status frame when no video is available."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw status information
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "ChAruco Calibration", (150, 200), font, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, "Waiting for camera input...", (180, 250), font, 0.6, (200, 200, 200), 1)
        
        calibrated = status.get('calibrated', False)
        if calibrated:
            cv2.putText(frame, "System Calibrated!", (200, 300), font, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Show ChAruco board to cameras", (140, 300), font, 0.6, (0, 165, 255), 1)
        
        return frame
    
    def _create_completion_frame(self) -> np.ndarray:
        """Create a completion frame when processing is done."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw completion message
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Calibration Complete!", (120, 180), font, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, "Homographies Computed", (160, 220), font, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Processing Stopped", (190, 260), font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Check saved files in calibration_results/", (90, 300), font, 0.5, (200, 200, 200), 1)
        
        return frame


def create_charuco_pipeline(
    num_cameras: int = 1,
    calibration_file: str = "charuco_calibration.json",
    output_width: int = 1920,
    output_height: int = 1080,
    use_quad_splitter: bool = False,
    charuco_config_file: Optional[str] = None,
    enable_live_preview: bool = True,
    enable_desktop_preview: bool = False,
    min_corners_for_homography: int = 8,
    force_fresh_calibration: bool = False
) -> Pipeline:
    """
    Create a ChAruco calibration and perspective warping pipeline.
    
    Pipeline flow:
    1. VideoQuadSplitterNode (optional) - Splits quad layout into separate frames
    2. VideoFrameBuffer - Synchronizes frames from multiple cameras  
    3. MultiCameraCalibrationNode - Handles ChAruco detection, calibration, and warping
    4. LivePreviewNode - Creates visual debugging interface with detection overlays
    5. VideoOutputFormatter - Formats output for WebRTC streaming (live preview or warped view)
    """
    pipeline = Pipeline()
    # Video stream analyzer for detailed frame logging
    # analyzer = VideoStreamAnalyzer(name="FrameAnalyzer", log_interval=30)  # Log every 30 frames
    # pipeline.add_node(analyzer)
    
    # Optional quad splitter for composite input frames
    if use_quad_splitter:
        quad_splitter = VideoQuadSplitterNode(name="QuadSplitter", num_splits=4)
        pipeline.add_node(quad_splitter)
        # When using quad splitter, we expect 4 cameras from the split
        actual_num_cameras = 4
    else:
        actual_num_cameras = num_cameras
    
    # Video frame synchronization
    frame_buffer = VideoFrameBuffer(num_cameras=actual_num_cameras, name="FrameBuffer")
    pipeline.add_node(frame_buffer)
    
    # ChAruco calibration configuration
    if charuco_config_file and os.path.exists(charuco_config_file):
        logger.info(f"📋 Loading ChAruco configuration from: {charuco_config_file}")
        charuco_config = CharucoConfig.from_json_file(charuco_config_file)
    else:
        # Use default 5x4 configuration
        charuco_config = CharucoConfig(
            squares_x=5,
            squares_y=4,
            square_length=0.03,
            marker_length=0.015,
            dictionary="DICT_4X4_50",
            margins=0.005,
            dpi=200
        )
    
    warp_config = WarpConfig(
        output_width=output_width,
        output_height=output_height,
        reference_camera=0,
        min_corners_for_homography=min_corners_for_homography
    )
    
    multi_camera_config = MultiCameraConfig(
        num_cameras=actual_num_cameras,
        charuco_config=charuco_config,
        warp_config=warp_config,
        max_calibration_frames=10,
        min_frames_for_calibration=5,
        auto_calibrate=True,
        calibration_file=calibration_file,
        enable_live_preview=True,
        force_fresh_calibration=force_fresh_calibration
    )
    
    # Main calibration node
    calibration_node = MultiCameraCalibrationNode(
        config=multi_camera_config,
        name="CharucoCalibration"
    )
    pipeline.add_node(calibration_node)
    
    # Live preview node for visual debugging (optional)
    if enable_live_preview:
        live_preview_config = LivePreviewConfig(
            preview_width=320,
            preview_height=240,
            grid_cols=2,
            show_corner_ids=True,
            show_pose_info=True,
            show_statistics=True
        )
        
        live_preview_node = LivePreviewNode(
            config=live_preview_config,
            charuco_config=charuco_config,
            name="LivePreview"
        )
        pipeline.add_node(live_preview_node)
    
    # Desktop preview node for saving images to disk (optional)
    if enable_desktop_preview:
        desktop_preview_config = DesktopPreviewConfig(
            enable_pygame_display=True,  # Use pygame for direct screen rendering
            enable_opencv_display=False,  # Disable OpenCV (fallback only)
            window_width=1200,
            window_height=800, 
            window_title="ChAruco Live Preview",
            enable_file_saving=False,  # Disable file saving for maximum performance
        )
        
        desktop_preview_node = DesktopPreviewNode(
            config=desktop_preview_config,
            name="DesktopPreview"
        )
        pipeline.add_node(desktop_preview_node)
    
    # Output formatting for WebRTC
    output_formatter = VideoOutputFormatter(show_live_preview=enable_live_preview, name="VideoOutput")
    pipeline.add_node(output_formatter)
    
    return pipeline


class CharucoWebRTCHandler:
    """Handles WebRTC-specific logic for ChAruco calibration."""
    
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.camera_assignments: Dict[str, int] = {}  # peer_id -> camera_id
        self.next_camera_id = 0
        
    def assign_camera_id(self, peer_id: str) -> int:
        """Assign a camera ID to a WebRTC peer."""
        if peer_id not in self.camera_assignments:
            self.camera_assignments[peer_id] = self.next_camera_id
            self.next_camera_id += 1
            logger.info(f"Assigned camera ID {self.camera_assignments[peer_id]} to peer {peer_id}")
        return self.camera_assignments[peer_id]
    
    def release_camera_id(self, peer_id: str):
        """Release camera ID when peer disconnects."""
        if peer_id in self.camera_assignments:
            camera_id = self.camera_assignments.pop(peer_id)
            logger.info(f"Released camera ID {camera_id} from peer {peer_id}")
            # Note: We don't reuse camera IDs to maintain consistency


async def create_charuco_webrtc_server(
    host: str = "0.0.0.0",
    port: int = 8081,
    num_cameras: int = 1,
    calibration_file: str = "charuco_calibration.json",
    output_width: int = 1920,
    output_height: int = 1080,
    use_quad_splitter: bool = False,
    charuco_config_file: Optional[str] = None,
    enable_live_preview: bool = True,
    enable_desktop_preview: bool = False,
    min_corners_for_homography: int = 8,
    force_fresh_calibration: bool = False
) -> WebRTCServer:
    """Create and configure the ChAruco WebRTC server."""
    
    # Server configuration
    examples_dir = Path(__file__).parent
    config = WebRTCConfig(
        host=host,
        port=port,
        enable_cors=True,
        stun_servers=["stun:stun.l.google.com:19302"],
        static_files_path=str(examples_dir)
    )
    
    # Pipeline factory function
    def create_pipeline() -> Pipeline:
        return create_charuco_pipeline(
            num_cameras=num_cameras,
            calibration_file=calibration_file,
            output_width=output_width,
            output_height=output_height,
            use_quad_splitter=use_quad_splitter,
            charuco_config_file=charuco_config_file,
            enable_live_preview=enable_live_preview,
            enable_desktop_preview=enable_desktop_preview,
            min_corners_for_homography=min_corners_for_homography,
            force_fresh_calibration=force_fresh_calibration
        )
    
    # Create server with ChAruco handler
    server = WebRTCServer(config=config, pipeline_factory=create_pipeline)
    
    return server


def parse_arguments():
    """Parse command line arguments with environment variable fallbacks."""
    parser = argparse.ArgumentParser(
        description="WebRTC ChAruco Calibration Server - Real-time multi-camera ChAruco calibration and perspective warping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables (used as fallbacks if CLI args not provided):
  SERVER_HOST=hostname        Server host (default: 0.0.0.0)
  SERVER_PORT=port           Server port (default: 8081)
  NUM_CAMERAS=count          Number of expected cameras (default: 4)
  CALIBRATION_FILE=path      Calibration storage file (default: charuco_calibration.json)
  OUTPUT_WIDTH=pixels        Warped output width (default: 1920)
  OUTPUT_HEIGHT=pixels       Warped output height (default: 1080)

Examples:
  python webrtc_charuco_pipeline_server.py --host 0.0.0.0 --port 8081 --cameras 4
  python webrtc_charuco_pipeline_server.py --calibration-file my_calibration.json
  SERVER_HOST=192.168.1.100 python webrtc_charuco_pipeline_server.py
        """
    )
    
    parser.add_argument(
        "--host", 
        type=str,
        default=os.environ.get("SERVER_HOST", "0.0.0.0"),
        help="Server host address (default: 0.0.0.0, env: SERVER_HOST)"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=int(os.environ.get("SERVER_PORT", "8081")),
        help="Server port (default: 8081, env: SERVER_PORT)"
    )
    
    parser.add_argument(
        "--cameras", "-c",
        type=int,
        default=int(os.environ.get("NUM_CAMERAS", "1")),
        help="Number of expected cameras (default: 1, env: NUM_CAMERAS)"
    )
    
    parser.add_argument(
        "--calibration-file", "-f",
        type=str,
        default=os.environ.get("CALIBRATION_FILE", "charuco_calibration.json"),
        help="Calibration storage file path (default: charuco_calibration.json, env: CALIBRATION_FILE)"
    )
    
    parser.add_argument(
        "--output-width", "-w",
        type=int,
        default=int(os.environ.get("OUTPUT_WIDTH", "1920")),
        help="Warped output width in pixels (default: 1920, env: OUTPUT_WIDTH)"
    )
    
    parser.add_argument(
        "--output-height", "-H",
        type=int,
        default=int(os.environ.get("OUTPUT_HEIGHT", "1080")),
        help="Warped output height in pixels (default: 1080, env: OUTPUT_HEIGHT)"
    )
    
    parser.add_argument(
        "--use-quad-splitter", "-q",
        action="store_true",
        default=os.environ.get("USE_QUAD_SPLITTER", "false").lower() == "true",
        help="Enable quad splitter for composite input frames (default: false, env: USE_QUAD_SPLITTER)"
    )
    
    parser.add_argument(
        "--charuco-config", "-cc",
        type=str,
        default=os.environ.get("CHARUCO_CONFIG_FILE", None),
        help="ChAruco board configuration JSON file (default: use built-in 5x4 config, env: CHARUCO_CONFIG_FILE)"
    )
    
    parser.add_argument(
        "--live-preview", "-lp",
        action="store_true",
        default=os.environ.get("ENABLE_LIVE_PREVIEW", "true").lower() == "true",
        help="Enable live preview with detection overlays (default: true, env: ENABLE_LIVE_PREVIEW)"
    )
    
    parser.add_argument(
        "--desktop-preview", "-dp",
        action="store_true",
        default=os.environ.get("ENABLE_DESKTOP_PREVIEW", "false").lower() == "true",
        help="Enable desktop preview by saving images to disk (default: false, env: ENABLE_DESKTOP_PREVIEW)"
    )
    
    parser.add_argument(
        "--min-corners", "-mc",
        type=int,
        default=int(os.environ.get("MIN_CORNERS_FOR_HOMOGRAPHY", "8")),
        help="Minimum corners required for homography computation (default: 8, env: MIN_CORNERS_FOR_HOMOGRAPHY)"
    )
    
    parser.add_argument(
        "--force-fresh-calibration", "-fc",
        action="store_true",
        default=os.environ.get("FORCE_FRESH_CALIBRATION", "false").lower() == "true",
        help="Force fresh camera calibration, ignore existing calibration files (default: false, env: FORCE_FRESH_CALIBRATION)"
    )
    
    return parser.parse_args()


async def main():
    """Main server function."""
    
    # Parse command line arguments and environment variables
    args = parse_arguments()
    
    # Use parsed arguments
    SERVER_HOST = args.host
    SERVER_PORT = args.port
    NUM_CAMERAS = args.cameras
    CALIBRATION_FILE = args.calibration_file
    OUTPUT_WIDTH = args.output_width
    OUTPUT_HEIGHT = args.output_height
    USE_QUAD_SPLITTER = args.use_quad_splitter
    ENABLE_LIVE_PREVIEW = args.live_preview
    ENABLE_DESKTOP_PREVIEW = args.desktop_preview
    MIN_CORNERS_FOR_HOMOGRAPHY = args.min_corners
    FORCE_FRESH_CALIBRATION = args.force_fresh_calibration
    
    # Temporarily set main logger to INFO for startup messages
    main_logger = logging.getLogger(__name__)
    main_logger.setLevel(logging.INFO)
    
    logger.info("=== ChAruco WebRTC Calibration Server ===")
    logger.info(f"Server: {SERVER_HOST}:{SERVER_PORT}")
    logger.info(f"Expected Cameras: {NUM_CAMERAS}")
    logger.info(f"Calibration File: {CALIBRATION_FILE}")
    logger.info(f"Output Resolution: {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}")
    logger.info(f"Quad Splitter: {'Enabled' if USE_QUAD_SPLITTER else 'Disabled'}")
    logger.info(f"Live Preview: {'Enabled' if ENABLE_LIVE_PREVIEW else 'Disabled'}")
    logger.info(f"Desktop Preview: {'Enabled' if ENABLE_DESKTOP_PREVIEW else 'Disabled'}")
    logger.info(f"Min Corners for Homography: {MIN_CORNERS_FOR_HOMOGRAPHY}")
    logger.info(f"Fresh Calibration: {'Enabled - will ignore existing calibration' if FORCE_FRESH_CALIBRATION else 'Disabled - will use existing calibration if available'}")
    # Show ChAruco board configuration
    if args.charuco_config and os.path.exists(args.charuco_config):
        config_display = CharucoConfig.from_json_file(args.charuco_config)
        config_source = f"from {args.charuco_config}"
    else:
        config_display = CharucoConfig()  # Use defaults
        config_source = "built-in defaults"
    
    logger.info("")
    logger.info(f"ChAruco Board Configuration ({config_source}):")
    logger.info(f"  • Size: {config_display.squares_x}×{config_display.squares_y} squares")
    logger.info(f"  • Square Length: {config_display.square_length*1000:.1f}mm")
    logger.info(f"  • Marker Length: {config_display.marker_length*1000:.1f}mm")
    logger.info(f"  • Dictionary: {config_display.dictionary}")
    logger.info(f"  • Margins: {config_display.margins*1000:.1f}mm")
    logger.info(f"  • Print DPI: {config_display.dpi}")
    
    # Create server
    server = await create_charuco_webrtc_server(
        host=SERVER_HOST,
        port=SERVER_PORT,
        num_cameras=NUM_CAMERAS,
        calibration_file=CALIBRATION_FILE,
        output_width=OUTPUT_WIDTH,
        output_height=OUTPUT_HEIGHT,
        use_quad_splitter=USE_QUAD_SPLITTER,
        charuco_config_file=args.charuco_config,
        enable_live_preview=ENABLE_LIVE_PREVIEW,
        enable_desktop_preview=ENABLE_DESKTOP_PREVIEW,
        min_corners_for_homography=MIN_CORNERS_FOR_HOMOGRAPHY,
        force_fresh_calibration=FORCE_FRESH_CALIBRATION
    )
    
    try:
        await server.start()
        
        logger.info("")
        logger.info("✅ ChAruco WebRTC server is running!")
        logger.info("📡 Connection endpoints:")
        logger.info(f"   • Web Client: http://localhost:{SERVER_PORT}/webrtc_client.html")
        logger.info(f"   • WebSocket: ws://{SERVER_HOST}:{SERVER_PORT}/ws")
        logger.info(f"   • Health Check: http://{SERVER_HOST}:{SERVER_PORT}/health")
        logger.info("")
        logger.info("🎯 Pipeline Features:")
        logger.info("   • Real-time ChAruco detection")
        logger.info("   • Multi-camera synchronization")
        logger.info("   • Automatic pose diversity selection")
        logger.info("   • Camera calibration")
        logger.info("   • Perspective warping and alignment")
        logger.info("   • Live composite view streaming")
        if ENABLE_DESKTOP_PREVIEW:
            logger.info("   • Real-time desktop preview (OpenCV + image files)")
            logger.info("   • Preview images saved to: ./live_preview/")
        logger.info("")
        logger.info("📋 Usage Instructions:")
        logger.info("   1. Connect multiple cameras/devices to the WebRTC server")
        logger.info("   2. Show ChAruco board to all cameras simultaneously")
        logger.info("   3. Move board through different poses for calibration diversity")
        logger.info("   4. Server automatically calibrates when sufficient frames collected")
        logger.info("   5. View real-time perspective-corrected composite output")
        logger.info("")
        logger.info("💡 Tips:")
        logger.info("   • Ensure good lighting for ChAruco detection")
        logger.info("   • Keep cameras stable during calibration")
        logger.info("   • Move board to cover different positions and orientations")
        logger.info("   • Full board visibility improves calibration accuracy")
        logger.info("")
        logger.info("Press Ctrl+C to stop the server")
        
        # Now set main logger back to WARNING to reduce noise during operation
        main_logger.setLevel(logging.WARNING)
        
        # Keep server running
        while True:
            await asyncio.sleep(10)
            
            # Log active connections (temporarily re-enable INFO level)
            connections_count = len(server.connections)
            if connections_count > 0:
                main_logger.setLevel(logging.INFO)
                logger.info(f"📊 Active connections: {connections_count}")
                main_logger.setLevel(logging.WARNING)
        
    except KeyboardInterrupt:
        main_logger.setLevel(logging.INFO)
        logger.info("🛑 Shutting down ChAruco WebRTC server...")
    except Exception as e:
        main_logger.setLevel(logging.INFO)
        logger.error(f"❌ Server error: {e}", exc_info=True)
    finally:
        await server.stop()
        main_logger.setLevel(logging.INFO)
        logger.info("✅ Server stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}", exc_info=True)