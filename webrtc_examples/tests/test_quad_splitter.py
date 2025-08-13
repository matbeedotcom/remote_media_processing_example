#!/usr/bin/env python3
"""
Test script for VideoQuadSplitterNode
Tests the quad splitting functionality with the sample image.
"""

import asyncio
import cv2
import numpy as np
from pathlib import Path
import sys

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "remote_media_processing"))

from video_quad_splitter_node import VideoQuadSplitterNode, VideoQuadMergerNode


async def test_quad_splitter():
    """Test the VideoQuadSplitterNode with a sample image."""
    
    print("Testing VideoQuadSplitterNode...")
    
    # Load the sample quad image
    sample_image_path = "/home/acidhax/dev/originals/remote_media/remote_media_processing_example/webrtc_examples/video_samples/frame_000300_1754599428.jpg"
    
    if not Path(sample_image_path).exists():
        print(f"❌ Sample image not found: {sample_image_path}")
        return
    
    # Read the image
    quad_image = cv2.imread(sample_image_path)
    if quad_image is None:
        print(f"❌ Failed to load image: {sample_image_path}")
        return
    
    print(f"✅ Loaded image: {quad_image.shape[1]}x{quad_image.shape[0]} pixels")
    
    # Create the splitter node
    splitter = VideoQuadSplitterNode(name="TestSplitter", num_splits=4)
    
    # Process the image
    input_data = {
        'frame': quad_image,
        'timestamp': 1,
        'camera_id': 0
    }
    
    result = await splitter.process(input_data)
    
    if result is None:
        print("❌ Splitter returned None")
        return
    
    # Check the results
    frames = result.get('frames', [])
    print(f"✅ Split into {len(frames)} frames")
    
    # Display information about each split frame
    for i, frame in enumerate(frames):
        print(f"  Frame {i}: {frame.shape[1]}x{frame.shape[0]} pixels")
        
        # Save each split frame for verification
        output_path = f"/tmp/split_frame_{i}.jpg"
        cv2.imwrite(output_path, frame)
        print(f"  Saved to: {output_path}")
    
    # Test the merger node
    print("\nTesting VideoQuadMergerNode...")
    merger = VideoQuadMergerNode(name="TestMerger", num_frames=4)
    
    merge_input = {
        'frames': frames,
        'timestamp': 1
    }
    
    merge_result = await merger.process(merge_input)
    
    if merge_result is None:
        print("❌ Merger returned None")
        return
    
    merged_frame = merge_result.get('frame')
    if merged_frame is not None:
        print(f"✅ Merged frame: {merged_frame.shape[1]}x{merged_frame.shape[0]} pixels")
        
        # Save the merged frame
        output_path = "/tmp/merged_frame.jpg"
        cv2.imwrite(output_path, merged_frame)
        print(f"  Saved to: {output_path}")
        
        # Compare dimensions
        if merged_frame.shape == quad_image.shape:
            print("✅ Merged frame has same dimensions as original")
        else:
            print(f"⚠️  Dimension mismatch: Original {quad_image.shape} vs Merged {merged_frame.shape}")
    
    # Display statistics
    print("\nStatistics:")
    print(f"Splitter: {splitter.get_statistics()}")
    print(f"Merger: {merger.get_statistics()}")
    
    print("\n✅ Test completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_quad_splitter())