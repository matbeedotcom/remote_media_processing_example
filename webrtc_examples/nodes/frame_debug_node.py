#!/usr/bin/env python3
"""
Frame Debug Node

Diagnostic node to save frames at various pipeline stages to understand
why ChAruco detection is failing.
"""

import logging
import numpy as np
import cv2
from typing import Dict, Any, Optional, List
from pathlib import Path
import sys
import time

# Add project paths  
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "remote_media_processing"))

from remotemedia.core.node import Node

logger = logging.getLogger(__name__)


class FrameDebugNode(Node):
    """
    Debug node that saves frames to disk for inspection.
    Can be inserted at any point in the pipeline.
    """
    
    def __init__(
        self, 
        name: Optional[str] = None, 
        save_frames: bool = True,
        save_interval: int = 30,
        max_saves: int = 20
    ):
        """
        Initialize the FrameDebugNode.
        
        Args:
            name: Optional name for the node
            save_frames: Whether to save frames to disk
            save_interval: Save every Nth frame
            max_saves: Maximum number of frames to save
        """
        super().__init__(name=name or "FrameDebug")
        self.save_frames = save_frames
        self.save_interval = save_interval
        self.max_saves = max_saves
        
        self.frame_counter = 0
        self.saved_count = 0
        self.save_dir = Path("debug_pipeline_frames")
        
        if self.save_frames:
            self.save_dir.mkdir(exist_ok=True)
            logger.info(f"🐛 FrameDebugNode will save frames to {self.save_dir}")
        
    async def process(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process data and optionally save debug frames.
        
        Args:
            data: Input data (passed through unchanged)
            
        Returns:
            Same data (pass-through node)
        """
        try:
            self.frame_counter += 1
            
            # Save debug info if enabled
            if (self.save_frames and 
                self.saved_count < self.max_saves and
                (self.frame_counter % self.save_interval == 0 or self.frame_counter <= 5)):
                
                await self._save_debug_info(data)
                self.saved_count += 1
            
            # Log frame info periodically
            if self.frame_counter <= 3 or self.frame_counter % 50 == 0:
                self._log_frame_info(data)
            
            # Pass through unchanged
            return data
            
        except Exception as e:
            logger.error(f"🐛 Error in FrameDebugNode: {e}")
            return data
    
    async def _save_debug_info(self, data: Dict[str, Any]):
        """Save debug information about the current data."""
        timestamp = int(time.time() * 1000)
        debug_info = []
        
        # Analyze the data structure
        debug_info.append(f"Frame #{self.frame_counter} at {timestamp}")
        debug_info.append(f"Data keys: {list(data.keys())}")
        
        # Check for various frame formats
        frames_to_save = []
        
        # Single frame
        if 'frame' in data:
            frame = data['frame']
            debug_info.append(f"Single frame: shape={frame.shape}, dtype={frame.dtype}")
            debug_info.append(f"  min={frame.min()}, max={frame.max()}, mean={frame.mean():.2f}")
            frames_to_save.append(('single_frame', frame))
        
        # Multiple frames
        if 'frames' in data and isinstance(data['frames'], list):
            frames_list = data['frames']
            debug_info.append(f"Frames list: {len(frames_list)} frames")
            
            for i, frame in enumerate(frames_list[:4]):  # Max 4 frames
                if frame is not None:
                    debug_info.append(f"  Frame {i}: shape={frame.shape}, dtype={frame.dtype}")
                    debug_info.append(f"    min={frame.min()}, max={frame.max()}, mean={frame.mean():.2f}")
                    frames_to_save.append((f'frame_{i}', frame))
                else:
                    debug_info.append(f"  Frame {i}: None")
        
        # Save frames
        for frame_name, frame in frames_to_save:
            if frame is not None and len(frame.shape) >= 2:
                # Convert to uint8 if needed
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:  # Normalized 0-1
                        save_frame = (frame * 255).astype(np.uint8)
                    else:
                        # Scale to 0-255 range
                        frame_min, frame_max = frame.min(), frame.max()
                        if frame_max > frame_min:
                            save_frame = ((frame - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)
                        else:
                            save_frame = frame.astype(np.uint8)
                else:
                    save_frame = frame
                
                # Ensure 3-channel for saving
                if len(save_frame.shape) == 2:
                    save_frame = cv2.cvtColor(save_frame, cv2.COLOR_GRAY2BGR)
                elif len(save_frame.shape) == 3 and save_frame.shape[2] == 1:
                    save_frame = cv2.cvtColor(save_frame, cv2.COLOR_GRAY2BGR)
                elif len(save_frame.shape) == 3 and save_frame.shape[2] == 3:
                    # RGB to BGR for OpenCV
                    save_frame = cv2.cvtColor(save_frame, cv2.COLOR_RGB2BGR)
                
                # Add debug overlay
                if save_frame.shape[0] > 50 and save_frame.shape[1] > 200:
                    cv2.putText(save_frame, f"Frame {self.frame_counter}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(save_frame, f"{frame_name}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(save_frame, f"Size: {frame.shape}", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # Save frame
                filename = self.save_dir / f"debug_{self.frame_counter:06d}_{frame_name}.png"
                cv2.imwrite(str(filename), save_frame)
                debug_info.append(f"Saved: {filename}")
        
        # Save debug text
        text_filename = self.save_dir / f"debug_{self.frame_counter:06d}_info.txt"
        with open(text_filename, 'w') as f:
            f.write('\n'.join(debug_info))
        
        logger.info(f"🐛 Saved debug frame {self.frame_counter}: {len(frames_to_save)} images")
    
    def _log_frame_info(self, data: Dict[str, Any]):
        """Log information about the current frame data."""
        info_parts = []
        
        # Analyze data structure
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                info_parts.append(f"{key}: {value.shape}")
            elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                info_parts.append(f"{key}: {len(value)} frames")
            elif key in ['timestamp', 'frame_counter']:
                info_parts.append(f"{key}: {value}")
        
        if info_parts:
            logger.info(f"🐛 Frame #{self.frame_counter}: {', '.join(info_parts)}")
        else:
            logger.warning(f"🐛 Frame #{self.frame_counter}: No recognizable frame data")