#!/usr/bin/env python3
"""
Video Stream Analyzer Node

Detailed logging and analysis of incoming video frames for debugging WebRTC streams.
Helps identify frame format, timing, and quality issues in the pipeline.
"""

import time
import logging
import numpy as np
import cv2
from typing import Dict, Any, Optional
from pathlib import Path
import sys

# Add project paths  
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "remote_media_processing"))

from remotemedia.core.node import Node

logger = logging.getLogger(__name__)


class VideoStreamAnalyzer(Node):
    """
    Analyzes incoming video streams with detailed logging and statistics.
    Useful for debugging WebRTC pipeline issues and frame processing problems.
    """
    
    def __init__(self, name: Optional[str] = None, log_interval: int = 30):
        super().__init__(name=name or "VideoStreamAnalyzer")
        self.log_interval = log_interval  # Log detailed stats every N frames
        
        # Statistics tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.last_frame_time = None
        
        # Frame analysis
        self.frame_sizes = []
        self.frame_types = []
        self.fps_history = []
        self.processing_times = []
        
        # Frame quality metrics
        self.brightness_history = []
        self.contrast_history = []
        
        logger.info(f"🔍 {self.name} initialized - will log every {log_interval} frames")
    
    async def process(self, data: Any) -> Any:
        """Analyze incoming video data and log detailed information."""
        process_start = time.time()
        logger.info(f"🔍 Processing data: {data}")
        try:
            # Handle different data formats
            if isinstance(data, dict):
                frame_data = self._analyze_dict_data(data)
            elif isinstance(data, tuple):
                frame_data = self._analyze_tuple_data(data)  
            elif isinstance(data, np.ndarray):
                frame_data = self._analyze_numpy_data(data)
            else:
                frame_data = self._analyze_unknown_data(data)
            
            # Update statistics
            self._update_statistics(frame_data, process_start)
            
            # Log periodically
            if self.frame_count % self.log_interval == 0 or self.frame_count <= 5:
                await self._log_detailed_stats(frame_data)
            
            # Save sample frames occasionally
            if self.frame_count in [1, 10, 30, 100] or self.frame_count % 300 == 0:
                await self._save_sample_frame(frame_data)
            
            return data  # Pass through unchanged
            
        except Exception as e:
            logger.error(f"❌ Error in {self.name}: {e}", exc_info=True)
            return data
    
    def _analyze_dict_data(self, data: dict) -> dict:
        """Analyze dictionary-format video data."""
        frame_data = {
            'type': 'dict',
            'keys': list(data.keys()),
            'has_frame': 'frame' in data,
            'frame': None,
            'metadata': {}
        }
        
        # Extract frame if present
        if 'frame' in data:
            frame = data['frame']
            frame_data['frame'] = frame
            frame_data['frame_info'] = self._analyze_frame_array(frame)
        
        # Extract metadata
        for key in ['width', 'height', 'format', 'pts', 'time_base', 'session_id', 'source', 'timestamp']:
            if key in data:
                frame_data['metadata'][key] = data[key]
        
        return frame_data
    
    def _analyze_tuple_data(self, data: tuple) -> dict:
        """Analyze tuple-format video data."""
        frame_data = {
            'type': 'tuple',
            'length': len(data),
            'elements': [type(item).__name__ for item in data],
            'frame': None
        }
        
        # Look for numpy array in tuple
        for i, item in enumerate(data):
            if isinstance(item, np.ndarray) and len(item.shape) >= 2:
                frame_data['frame'] = item
                frame_data['frame_info'] = self._analyze_frame_array(item)
                frame_data['frame_position'] = i
                break
        
        return frame_data
    
    def _analyze_numpy_data(self, data: np.ndarray) -> dict:
        """Analyze raw numpy array video data."""
        frame_data = {
            'type': 'numpy',
            'frame': data,
            'frame_info': self._analyze_frame_array(data)
        }
        return frame_data
    
    def _analyze_unknown_data(self, data: Any) -> dict:
        """Analyze unknown data format."""
        frame_data = {
            'type': 'unknown',
            'python_type': type(data).__name__,
            'has_frame': False,
            'frame': None,
            'data_str': str(data)[:100] + '...' if len(str(data)) > 100 else str(data)
        }
        return frame_data
    
    def _analyze_frame_array(self, frame: np.ndarray) -> dict:
        """Analyze a numpy array representing a video frame."""
        if frame is None or not isinstance(frame, np.ndarray):
            return {'valid': False, 'error': 'Not a valid numpy array'}
        
        try:
            info = {
                'valid': True,
                'shape': frame.shape,
                'dtype': str(frame.dtype),
                'size_bytes': frame.nbytes,
                'dimensions': len(frame.shape),
            }
            
            if len(frame.shape) >= 2:
                info['height'] = frame.shape[0]
                info['width'] = frame.shape[1]
                info['resolution'] = f"{frame.shape[1]}x{frame.shape[0]}"
                
                if len(frame.shape) == 3:
                    info['channels'] = frame.shape[2]
                    info['color_format'] = 'RGB/BGR' if frame.shape[2] == 3 else f"{frame.shape[2]}-channel"
                else:
                    info['channels'] = 1
                    info['color_format'] = 'Grayscale'
                
                # Calculate basic image statistics
                info['min_value'] = int(np.min(frame))
                info['max_value'] = int(np.max(frame))
                info['mean_value'] = float(np.mean(frame))
                info['std_value'] = float(np.std(frame))
                
                # Check if frame is blank/black
                info['is_blank'] = np.max(frame) < 10
                info['is_uniform'] = np.std(frame) < 1.0
                
                # Brightness and contrast estimates
                if len(frame.shape) == 3:
                    # Convert to grayscale for analysis
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                else:
                    gray = frame
                
                info['brightness'] = float(np.mean(gray))
                info['contrast'] = float(np.std(gray))
                
            return info
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def _update_statistics(self, frame_data: dict, process_start: float):
        """Update running statistics."""
        self.frame_count += 1
        current_time = time.time()
        processing_time = current_time - process_start
        
        self.processing_times.append(processing_time)
        
        # Calculate FPS
        if self.last_frame_time is not None:
            frame_interval = current_time - self.last_frame_time
            if frame_interval > 0:
                fps = 1.0 / frame_interval
                self.fps_history.append(fps)
        
        self.last_frame_time = current_time
        
        # Track frame info
        if frame_data.get('frame_info', {}).get('valid'):
            info = frame_data['frame_info']
            if 'size_bytes' in info:
                self.frame_sizes.append(info['size_bytes'])
            if 'brightness' in info:
                self.brightness_history.append(info['brightness'])
            if 'contrast' in info:
                self.contrast_history.append(info['contrast'])
        
        self.frame_types.append(frame_data['type'])
        
        # Keep history limited
        max_history = 100
        for history_list in [self.fps_history, self.frame_sizes, self.processing_times, 
                           self.brightness_history, self.contrast_history]:
            if len(history_list) > max_history:
                history_list[:] = history_list[-max_history:]
    
    async def _log_detailed_stats(self, frame_data: dict):
        """Log detailed statistics about the video stream."""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Basic stats
        logger.info(f"📊 {self.name} - Frame #{self.frame_count} - {elapsed:.1f}s elapsed")
        logger.info(f"   📦 Data type: {frame_data['type']}")
        
        # Frame analysis
        if 'frame_info' in frame_data and frame_data['frame_info'].get('valid'):
            info = frame_data['frame_info']
            logger.info(f"   🖼️  Frame: {info.get('resolution', 'unknown')} {info.get('color_format', '')} ({info.get('dtype', 'unknown')})")
            logger.info(f"   📏 Size: {info.get('size_bytes', 0)/1024:.1f}KB, Shape: {info.get('shape', 'unknown')}")
            logger.info(f"   🌟 Values: min={info.get('min_value', 0)}, max={info.get('max_value', 0)}, mean={info.get('mean_value', 0):.1f}")
            logger.info(f"   💡 Quality: brightness={info.get('brightness', 0):.1f}, contrast={info.get('contrast', 0):.1f}")
            
            if info.get('is_blank'):
                logger.warning(f"   ⚠️  Frame appears to be blank/black!")
            if info.get('is_uniform'):
                logger.warning(f"   ⚠️  Frame appears to be uniform (low detail)!")
        
        # Performance stats
        if self.fps_history:
            avg_fps = np.mean(self.fps_history[-10:])  # Last 10 frames
            logger.info(f"   ⏱️  FPS: {avg_fps:.1f} (recent average)")
        
        if self.processing_times:
            avg_processing = np.mean(self.processing_times[-10:]) * 1000  # ms
            logger.info(f"   🔄 Processing: {avg_processing:.2f}ms per frame")
        
        # Stream health
        total_fps = self.frame_count / elapsed if elapsed > 0 else 0
        logger.info(f"   📈 Overall: {total_fps:.1f} FPS average, {len(set(self.frame_types))} data type(s)")
        
        # Metadata
        if 'metadata' in frame_data and frame_data['metadata']:
            metadata = frame_data['metadata']
            logger.info(f"   📋 Metadata: {metadata}")
    
    async def _save_sample_frame(self, frame_data: dict):
        """Save sample frames for visual inspection."""
        try:
            if not frame_data.get('frame_info', {}).get('valid'):
                return
            
            frame = frame_data['frame']
            if frame is None:
                return
            
            # Create samples directory
            samples_dir = Path("video_samples")
            samples_dir.mkdir(exist_ok=True)
            
            # Generate filename
            timestamp = int(time.time())
            filename = f"frame_{self.frame_count:06d}_{timestamp}.jpg"
            filepath = samples_dir / filename
            
            # Convert and save frame
            if len(frame.shape) == 3:
                # Assume RGB, convert to BGR for OpenCV
                save_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                save_frame = frame
            
            cv2.imwrite(str(filepath), save_frame)
            logger.info(f"💾 Saved sample frame: {filepath}")
            
        except Exception as e:
            logger.debug(f"Could not save sample frame: {e}")
    
    def get_statistics(self) -> dict:
        """Get current statistics summary."""
        elapsed = time.time() - self.start_time
        
        return {
            'frame_count': self.frame_count,
            'elapsed_seconds': elapsed,
            'average_fps': self.frame_count / elapsed if elapsed > 0 else 0,
            'recent_fps': np.mean(self.fps_history[-10:]) if self.fps_history else 0,
            'data_types': list(set(self.frame_types)),
            'average_processing_ms': np.mean(self.processing_times) * 1000 if self.processing_times else 0,
            'average_frame_size_kb': np.mean(self.frame_sizes) / 1024 if self.frame_sizes else 0,
            'average_brightness': np.mean(self.brightness_history) if self.brightness_history else 0,
            'average_contrast': np.mean(self.contrast_history) if self.contrast_history else 0,
        }


if __name__ == "__main__":
    # Test the analyzer with dummy data
    import asyncio
    
    async def test_analyzer():
        analyzer = VideoStreamAnalyzer(log_interval=1)  # Log every frame for testing
        
        # Test different data formats
        test_data = [
            # Dict format (WebRTC style)
            {
                'frame': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                'width': 640,
                'height': 480,
                'format': 'rgb24',
                'source': 'webrtc'
            },
            # Numpy array
            np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
            # Tuple format
            (np.random.randint(0, 255, (120, 160), dtype=np.uint8), 24000, {'test': True}),
            # Unknown format
            "not a video frame"
        ]
        
        for i, data in enumerate(test_data):
            print(f"\n--- Testing data format {i+1} ---")
            result = await analyzer.process(data)
            assert result == data, "Data should pass through unchanged"
        
        # Print final statistics
        stats = analyzer.get_statistics()
        print(f"\n📊 Final Statistics: {stats}")
    
    asyncio.run(test_analyzer())