#!/usr/bin/env python3
"""
Video Quad Splitter Node

Splits a horizontally laid out quad image (4 images side by side) into 4 separate frames.
Useful for processing multi-camera composite streams where multiple views are combined
into a single frame.
"""

import logging
import numpy as np
import cv2
from typing import Dict, Any, Optional, List
from pathlib import Path
import sys

# Add project paths  
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "remote_media_processing"))

from remotemedia.core.node import Node

logger = logging.getLogger(__name__)


class VideoQuadSplitterNode(Node):
    """
    Splits a horizontally laid out quad image into 4 separate frames.
    
    Expects input frames where 4 camera views are concatenated horizontally
    in a single image. Outputs the 4 individual frames as a list.
    """
    
    def __init__(self, name: Optional[str] = None, num_splits: int = 4):
        """
        Initialize the VideoQuadSplitterNode.
        
        Args:
            name: Optional name for the node
            num_splits: Number of horizontal splits (default: 4)
        """
        super().__init__(name=name or "VideoQuadSplitter")
        self.num_splits = num_splits
        self.frame_counter = 0
        self.error_count = 0
        
        logger.info(f"🎬 VideoQuadSplitter initialized for {num_splits} splits")
        
    async def process(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process incoming frame and split it into separate camera views.
        
        Args:
            data: Input data containing 'frame' key with the composite image
            
        Returns:
            Dictionary with 'frames' containing list of split frames,
            or None if processing fails
        """
        try:
            # Get the composite frame
            frame = data.get('frame')
            if frame is None:
                logger.warning("⚠️ VideoQuadSplitter received data without 'frame' key")
                return None
                
            # Validate frame dimensions
            if len(frame.shape) != 3:
                logger.error(f"❌ Invalid frame shape: {frame.shape}. Expected 3D array (H, W, C)")
                self.error_count += 1
                return None
                
            height, width, channels = frame.shape
            self.frame_counter += 1
            
            # Calculate split width
            split_width = width // self.num_splits
            
            if split_width * self.num_splits != width:
                logger.warning(f"⚠️ Frame width {width} not evenly divisible by {self.num_splits}. "
                             f"Some pixels will be cropped.")
            
            # Split the frame horizontally
            frames = []
            for i in range(self.num_splits):
                # Calculate start and end positions for this split
                start_x = i * split_width
                end_x = start_x + split_width
                
                # Extract the sub-frame
                sub_frame = frame[:, start_x:end_x, :]
                frames.append(sub_frame)
                
            # Log processing info periodically
            if self.frame_counter <= 5 or self.frame_counter % 30 == 0:
                logger.info(f"✂️ Split frame #{self.frame_counter}: "
                          f"Input={width}x{height}, "
                          f"Output={self.num_splits} frames of {split_width}x{height}")
            
            # Prepare output data
            output_data = {
                'frames': frames,
                'timestamp': data.get('timestamp', self.frame_counter),
                'original_size': (width, height),
                'split_size': (split_width, height),
                'num_splits': self.num_splits
            }
            
            # Pass through any additional metadata
            for key in ['camera_id', 'fps', 'metadata']:
                if key in data:
                    output_data[key] = data[key]
            
            return output_data
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Error in VideoQuadSplitter: {e}", exc_info=True)
            
            # Log error statistics periodically
            if self.error_count % 10 == 0:
                logger.warning(f"⚠️ VideoQuadSplitter has encountered {self.error_count} errors")
            
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get node processing statistics.
        
        Returns:
            Dictionary containing processing statistics
        """
        return {
            'frames_processed': self.frame_counter,
            'errors_encountered': self.error_count,
            'num_splits': self.num_splits,
            'error_rate': self.error_count / max(1, self.frame_counter)
        }


class VideoQuadMergerNode(Node):
    """
    Merges 4 separate frames into a single horizontally laid out quad image.
    
    The inverse operation of VideoQuadSplitterNode. Takes a list of frames
    and combines them horizontally into a single composite image.
    """
    
    def __init__(self, name: Optional[str] = None, num_frames: int = 4):
        """
        Initialize the VideoQuadMergerNode.
        
        Args:
            name: Optional name for the node
            num_frames: Number of frames to merge horizontally (default: 4)
        """
        super().__init__(name=name or "VideoQuadMerger")
        self.num_frames = num_frames
        self.frame_counter = 0
        self.error_count = 0
        
        logger.info(f"🎬 VideoQuadMerger initialized for {num_frames} frames")
        
    async def process(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process incoming frames and merge them into a single composite view.
        
        Args:
            data: Input data containing 'frames' key with list of frames
            
        Returns:
            Dictionary with 'frame' containing the merged composite image,
            or None if processing fails
        """
        try:
            # Get the list of frames
            frames = data.get('frames')
            if frames is None:
                logger.warning("⚠️ VideoQuadMerger received data without 'frames' key")
                return None
            
            if not isinstance(frames, list):
                logger.error(f"❌ Expected 'frames' to be a list, got {type(frames)}")
                self.error_count += 1
                return None
            
            # Handle insufficient frames
            if len(frames) < self.num_frames:
                logger.warning(f"⚠️ Expected {self.num_frames} frames, got {len(frames)}. "
                             f"Padding with black frames.")
                
                # Get dimensions from first frame
                if len(frames) > 0:
                    h, w = frames[0].shape[:2]
                    channels = frames[0].shape[2] if len(frames[0].shape) > 2 else 1
                else:
                    h, w, channels = 480, 640, 3  # Default dimensions
                
                # Pad with black frames
                while len(frames) < self.num_frames:
                    black_frame = np.zeros((h, w, channels), dtype=np.uint8)
                    frames.append(black_frame)
            
            # Trim excess frames if necessary
            if len(frames) > self.num_frames:
                logger.warning(f"⚠️ Got {len(frames)} frames, using first {self.num_frames}")
                frames = frames[:self.num_frames]
            
            # Ensure all frames have the same height
            heights = [f.shape[0] for f in frames]
            max_height = max(heights)
            
            # Resize frames if heights don't match
            resized_frames = []
            for i, frame in enumerate(frames):
                if frame.shape[0] != max_height:
                    # Calculate new width to maintain aspect ratio
                    aspect_ratio = frame.shape[1] / frame.shape[0]
                    new_width = int(max_height * aspect_ratio)
                    resized = cv2.resize(frame, (new_width, max_height))
                    resized_frames.append(resized)
                else:
                    resized_frames.append(frame)
            
            # Concatenate frames horizontally
            merged_frame = np.hstack(resized_frames)
            self.frame_counter += 1
            
            # Log processing info periodically
            if self.frame_counter <= 5 or self.frame_counter % 30 == 0:
                height, width = merged_frame.shape[:2]
                logger.info(f"🔀 Merged frame #{self.frame_counter}: "
                          f"{self.num_frames} frames -> {width}x{height}")
            
            # Prepare output data
            output_data = {
                'frame': merged_frame,
                'timestamp': data.get('timestamp', self.frame_counter),
                'num_merged': len(frames),
                'merged_size': merged_frame.shape[:2]
            }
            
            # Pass through any additional metadata
            for key in ['camera_id', 'fps', 'metadata']:
                if key in data:
                    output_data[key] = data[key]
            
            return output_data
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Error in VideoQuadMerger: {e}", exc_info=True)
            
            # Log error statistics periodically
            if self.error_count % 10 == 0:
                logger.warning(f"⚠️ VideoQuadMerger has encountered {self.error_count} errors")
            
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get node processing statistics.
        
        Returns:
            Dictionary containing processing statistics
        """
        return {
            'frames_processed': self.frame_counter,
            'errors_encountered': self.error_count,
            'num_frames': self.num_frames,
            'error_rate': self.error_count / max(1, self.frame_counter)
        }