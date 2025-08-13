#!/usr/bin/env python3
"""
Test script to verify warped view generation using saved calibration data.
"""

import asyncio
import numpy as np
import cv2
import sys
import os
from pathlib import Path

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'charuco'))

from charuco.desktop_preview_node import DesktopPreviewNode, DesktopPreviewConfig

async def test_warped_preview():
    """Test the warped preview functionality."""
    
    # Configuration for warped preview
    config = DesktopPreviewConfig(
        enable_warping=True,
        calibration_file="charuco/camera_calibration.json",
        homography_dir="calibration_results_20250807_235659",  # Use the good homography file
        canvas_width=1920,
        canvas_height=1080,
        blend_mode="overlay",
        undistort_images=True,
        enable_pygame_display=True,
        enable_opencv_display=True,  # Fallback
        window_title="Warped ChAruco Preview Test"
    )
    
    # Create the desktop preview node
    preview_node = DesktopPreviewNode(config)
    
    # Load some test images from the calibration results
    test_images = {}
    calibration_dir = Path("calibration_results_20250807_235659")
    
    for i in range(4):
        img_path = calibration_dir / f"camera_{i}_original.jpg"
        if img_path.exists():
            image = cv2.imread(str(img_path))
            if image is not None:
                test_images[str(i)] = image
                print(f"📷 Loaded test image for camera {i}: {image.shape}")
    
    if not test_images:
        print("❌ No test images found in calibration results")
        return
    
    print(f"🔄 Testing warped view with {len(test_images)} cameras")
    print(f"📊 Homographies loaded: {len(preview_node.homographies)}")
    print(f"🎯 Reference camera: {preview_node.reference_camera}")
    
    # Test the warping functionality
    test_data = {
        'camera_frames': test_images,
        'images': list(test_images.values())
    }
    
    # Process the data
    result = await preview_node.process(test_data)
    
    if 'warped_preview' in result:
        warped_frame = result['warped_preview']
        print(f"✅ Generated warped view: {warped_frame.shape}")
        
        # Save the result
        output_path = "test_warped_output.jpg"
        cv2.imwrite(output_path, warped_frame)
        print(f"💾 Saved warped test result: {output_path}")
        
        # Keep the display window open for a few seconds
        import time
        print("🖥️  Displaying warped view for 10 seconds...")
        time.sleep(10)
    else:
        print("❌ No warped preview generated")
    
    # Cleanup
    preview_node.cleanup()

if __name__ == "__main__":
    asyncio.run(test_warped_preview())