#!/usr/bin/env python3
"""
Video Quad Splitter to Multi-Stream Node

Splits a horizontally laid out quad image and outputs each split as a separate
frame with appropriate camera_id for the FrameBuffer to process.
This node is specifically designed to work with the VideoFrameBuffer node.
"""

import logging
import numpy as np
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
import sys

# Add project paths  
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "remote_media_processing"))

from remotemedia.core.node import Node

logger = logging.getLogger(__name__)


class VideoQuadSplitterToMultiNode(Node):
    """
    Splits a horizontally laid out quad image and outputs multiple frames
    with individual camera IDs for the FrameBuffer to process.
    
    This node takes a single composite frame and outputs multiple individual
    frames, each tagged with the appropriate camera_id.
    """
    
    def __init__(self, name: Optional[str] = None, num_splits: int = 4):
        """
        Initialize the VideoQuadSplitterToMultiNode.
        
        Args:
            name: Optional name for the node
            num_splits: Number of horizontal splits (default: 4)
        """
        super().__init__(name=name or "VideoQuadSplitterToMulti")
        self.num_splits = num_splits
        self.frame_counter = 0
        self.error_count = 0
        self.pending_outputs = []  # Queue for frames to output
        
        logger.info(f"🎬 VideoQuadSplitterToMulti initialized for {num_splits} splits")
        
    async def process(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process incoming frame and split it into separate camera views.
        Each split frame is output individually with its camera_id.
        
        Args:
            data: Input data containing 'frame' key with the composite image
            
        Returns:
            Dictionary with individual frame and camera_id for FrameBuffer,
            or None if no more frames to output
        """
        try:
            # Check if we have pending outputs from a previous split
            if self.pending_outputs:
                return self.pending_outputs.pop(0)
            
            # Get the composite frame
            frame = data.get('frame')
            if frame is None:
                logger.warning("⚠️ VideoQuadSplitterToMulti received data without 'frame' key")
                return None
                
            # Validate frame dimensions
            if len(frame.shape) != 3:
                logger.error(f"❌ Invalid frame shape: {frame.shape}. Expected 3D array (H, W, C)")
                self.error_count += 1
                return None
                
            height, width, channels = frame.shape
            self.frame_counter += 1
            timestamp = data.get('timestamp', self.frame_counter)
            
            # Calculate split width
            split_width = width // self.num_splits
            
            if split_width * self.num_splits != width:
                logger.warning(f"⚠️ Frame width {width} not evenly divisible by {self.num_splits}. "
                             f"Some pixels will be cropped.")
            
            # Split the frame horizontally and prepare outputs
            outputs = []
            for i in range(self.num_splits):
                # Calculate start and end positions for this split
                start_x = i * split_width
                end_x = start_x + split_width
                
                # Extract the sub-frame
                sub_frame = frame[:, start_x:end_x, :]
                
                # Create output data for this camera
                output_data = {
                    'frame': sub_frame,
                    'camera_id': i,
                    'timestamp': timestamp,
                    'original_size': (width, height),
                    'split_size': (split_width, height)
                }
                
                # Pass through any additional metadata
                for key in ['fps', 'metadata', 'pts', 'time_base', 'session_id', 'source']:
                    if key in data:
                        output_data[key] = data[key]
                
                outputs.append(output_data)
            
            # Log processing info periodically
            if self.frame_counter <= 5 or self.frame_counter % 30 == 0:
                logger.info(f"✂️ Split frame #{self.frame_counter}: "
                          f"Input={width}x{height}, "
                          f"Output={self.num_splits} frames of {split_width}x{height}")
            
            # Store outputs for subsequent calls
            if len(outputs) > 1:
                self.pending_outputs = outputs[1:]  # Save remaining outputs
                
            # Return the first output immediately
            return outputs[0] if outputs else None
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Error in VideoQuadSplitterToMulti: {e}", exc_info=True)
            
            # Log error statistics periodically
            if self.error_count % 10 == 0:
                logger.warning(f"⚠️ VideoQuadSplitterToMulti has encountered {self.error_count} errors")
            
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
            'error_rate': self.error_count / max(1, self.frame_counter),
            'pending_outputs': len(self.pending_outputs)
        }