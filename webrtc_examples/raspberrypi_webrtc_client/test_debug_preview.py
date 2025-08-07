#!/usr/bin/env python3
"""
Test script for debugging WebRTC video frame preview functionality.

This script tests the frame preview and debugging features of the Raspberry Pi WebRTC client.
It can be used to verify that frames are being captured correctly before sending to the server.
"""

import asyncio
import sys
import os
import logging
import time
from pathlib import Path
import argparse

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera_manager import CameraManager
from video_track import CameraVideoTrack

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


async def test_camera_preview(
    camera_index: int = 0,
    duration: int = 10,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    save_frames: bool = True,
    preview_dir: str = "/tmp/webrtc_debug"
):
    """Test camera preview functionality."""
    
    logger.info("=== WebRTC Frame Preview Test ===")
    logger.info(f"Camera: {camera_index}")
    logger.info(f"Resolution: {width}x{height} @ {fps}fps")
    logger.info(f"Duration: {duration} seconds")
    logger.info(f"Save frames: {save_frames}")
    if save_frames:
        logger.info(f"Preview directory: {preview_dir}")
    logger.info("")
    
    # Initialize camera manager
    camera_manager = CameraManager()
    cameras = camera_manager.get_camera_list()
    
    if not cameras:
        logger.error("No cameras detected!")
        return False
    
    logger.info("📹 Available cameras:")
    for cam in cameras:
        logger.info(f"  [{cam.index}] {cam.name} ({cam.type})")
    
    # Get selected camera
    camera_info = camera_manager.get_camera(camera_index)
    if not camera_info:
        logger.error(f"Camera {camera_index} not found!")
        return False
    
    logger.info(f"\n🎯 Testing camera {camera_index}: {camera_info.name}")
    
    # Create video track with debug features enabled
    track = CameraVideoTrack(
        camera_info,
        width=width,
        height=height,
        fps=fps,
        camera_id=camera_index,
        debug_preview=True,  # Enable frame analysis
        save_preview_frames=save_frames,
        preview_dir=preview_dir
    )
    
    logger.info("📺 Starting frame capture...")
    logger.info("   Frame analysis will show:")
    logger.info("   • Frame brightness statistics")
    logger.info("   • Detection of blank/stuck frames")
    logger.info("   • Frame rate monitoring")
    if save_frames:
        logger.info(f"   • Preview frames saved to: {preview_dir}/camera_{camera_index}/")
    logger.info("")
    
    # Capture frames for the specified duration
    start_time = time.time()
    frame_count = 0
    error_count = 0
    
    try:
        while time.time() - start_time < duration:
            try:
                # Get a frame from the track
                frame = await track.recv()
                frame_count += 1
                
                # Log progress every second
                if frame_count % fps == 0:
                    elapsed = time.time() - start_time
                    actual_fps = frame_count / elapsed if elapsed > 0 else 0
                    logger.info(f"⏱️  Progress: {elapsed:.1f}s, {frame_count} frames, {actual_fps:.1f} FPS")
                
            except Exception as e:
                error_count += 1
                logger.error(f"Error getting frame: {e}")
                if error_count > 10:
                    logger.error("Too many errors, stopping test")
                    break
            
            # Small delay to match target FPS
            await asyncio.sleep(1.0 / fps)
    
    except KeyboardInterrupt:
        logger.info("\n⚠️  Test interrupted by user")
    
    finally:
        # Stop the track
        track.stop()
        
        # Get final statistics
        stats = track.get_stats()
        
        logger.info("\n=== Test Results ===")
        logger.info(f"Duration: {time.time() - start_time:.1f} seconds")
        logger.info(f"Frames captured: {stats['frame_count']}")
        logger.info(f"Average FPS: {stats['fps']:.1f}")
        logger.info(f"Errors: {stats['error_count']}")
        
        if stats.get('frame_stats'):
            frame_stats = stats['frame_stats']
            if stats['frame_count'] > 0:
                avg_brightness = frame_stats['total_brightness'] / stats['frame_count']
                logger.info("\n📊 Frame Analysis:")
                logger.info(f"  • Brightness range: {frame_stats['min_brightness']:.1f} - {frame_stats['max_brightness']:.1f}")
                logger.info(f"  • Average brightness: {avg_brightness:.1f}")
                logger.info(f"  • Blank frames: {frame_stats['blank_frames']}")
                logger.info(f"  • Dark frames: {frame_stats['dark_frames']}")
                logger.info(f"  • Bright frames: {frame_stats['bright_frames']}")
                
                # Warnings
                if frame_stats['blank_frames'] > 0:
                    logger.warning(f"⚠️  Detected {frame_stats['blank_frames']} blank/black frames")
                if frame_stats['dark_frames'] > stats['frame_count'] * 0.8:
                    logger.warning(f"⚠️  Most frames are dark - check lighting or camera exposure")
        
        if save_frames:
            preview_path = Path(preview_dir) / f"camera_{camera_index}"
            if preview_path.exists():
                saved_files = list(preview_path.glob("*.jpg"))
                logger.info(f"\n💾 Saved {len(saved_files)} preview frames to:")
                logger.info(f"   {preview_path}")
                if saved_files:
                    logger.info(f"   First frame: {saved_files[0].name}")
                    logger.info(f"   Last frame: {saved_files[-1].name}")
    
    return frame_count > 0


async def main():
    """Main test function."""
    parser = argparse.ArgumentParser(
        description="Test WebRTC frame preview and debugging features"
    )
    
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index to test (default: 0)"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Test duration in seconds (default: 10)"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Frame width (default: 640)"
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Frame height (default: 480)"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target FPS (default: 30)"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save preview frames to disk"
    )
    
    parser.add_argument(
        "--preview-dir",
        type=str,
        default="/tmp/webrtc_debug",
        help="Directory for preview frames (default: /tmp/webrtc_debug)"
    )
    
    args = parser.parse_args()
    
    # Run test
    success = await test_camera_preview(
        camera_index=args.camera,
        duration=args.duration,
        width=args.width,
        height=args.height,
        fps=args.fps,
        save_frames=not args.no_save,
        preview_dir=args.preview_dir
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nTest interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)