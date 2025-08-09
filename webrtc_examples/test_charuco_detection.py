#!/usr/bin/env python3
"""
Manual ChAruco detection test on saved calibration images.
Tests different board configurations to identify detection issues.
"""

import cv2
import numpy as np
import os
import sys

def test_charuco_detection(image_path, squares_x, squares_y, square_length, marker_length, dict_name="DICT_4X4_50"):
    """Test ChAruco detection with specified configuration."""
    
    print(f"\n{'='*60}")
    print(f"Testing: {os.path.basename(image_path)}")
    print(f"Board Config: {squares_x}x{squares_y}, square={square_length*1000:.1f}mm, marker={marker_length*1000:.1f}mm")
    print(f"{'='*60}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Initialize ArUco dictionary
    dict_id = getattr(cv2.aruco, dict_name)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    
    # Create ChAruco board
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length,
        marker_length,
        aruco_dict
    )
    
    # Initialize detector
    detector_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    
    # Detect ArUco markers
    corners, ids, rejected = aruco_detector.detectMarkers(gray)
    
    aruco_count = len(ids) if ids is not None else 0
    print(f"\n📊 ArUco Detection:")
    print(f"   ✅ ArUco markers found: {aruco_count}")
    if ids is not None and len(ids) > 0:
        print(f"   📍 Marker IDs: {ids.flatten().tolist()}")
    print(f"   ❌ Rejected candidates: {len(rejected) if rejected is not None else 0}")
    
    # Try ChAruco detection with new API
    if ids is not None and len(ids) > 0:
        print(f"\n🎯 ChAruco Detection (New API):")
        try:
            charuco_params = cv2.aruco.CharucoParameters()
            charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params)
            charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
            
            if charuco_corners is not None and len(charuco_corners) > 0:
                print(f"   ✅ ChAruco corners detected: {len(charuco_corners)}")
                expected_corners = (squares_x - 1) * (squares_y - 1)
                print(f"   📊 Coverage: {len(charuco_corners)}/{expected_corners} ({len(charuco_corners)/expected_corners*100:.1f}%)")
                print(f"   📍 Corner IDs: {charuco_ids.flatten().tolist()}")
            else:
                print(f"   ❌ No ChAruco corners detected!")
        except Exception as e:
            print(f"   ❌ New API failed: {e}")
        
        # Try old API method for comparison
        print(f"\n🎯 ChAruco Detection (Old API):")
        try:
            # Old API: interpolate ChAruco corners
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board
            )
            
            if retval > 0:
                print(f"   ✅ ChAruco corners detected: {retval}")
                expected_corners = (squares_x - 1) * (squares_y - 1)
                print(f"   📊 Coverage: {retval}/{expected_corners} ({retval/expected_corners*100:.1f}%)")
                if charuco_ids is not None:
                    print(f"   📍 Corner IDs: {charuco_ids.flatten().tolist()}")
            else:
                print(f"   ❌ No ChAruco corners detected!")
        except Exception as e:
            print(f"   ❌ Old API failed: {e}")
    
    # Create visualization
    vis_image = image.copy()
    
    # Draw ArUco markers
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(vis_image, corners, ids)
    
    # Try to draw ChAruco corners if detected
    try:
        if 'charuco_corners' in locals() and charuco_corners is not None and len(charuco_corners) > 0:
            cv2.aruco.drawDetectedCornersCharuco(vis_image, charuco_corners, charuco_ids, (0, 255, 0))
    except:
        pass
    
    # Save visualization
    output_path = image_path.replace('_original.jpg', '_detection_test.jpg')
    cv2.imwrite(output_path, vis_image)
    print(f"\n💾 Saved visualization: {output_path}")
    
    return aruco_count, len(charuco_corners) if 'charuco_corners' in locals() and charuco_corners is not None else 0


def main():
    # Test images
    test_images = [
        '/home/acidhax/dev/originals/remote_media/remote_media_processing_example/webrtc_examples/calibration_results_20250807_184400/camera_0_original.jpg',
        '/home/acidhax/dev/originals/remote_media/remote_media_processing_example/webrtc_examples/calibration_results_20250807_184400/camera_1_original.jpg',
        '/home/acidhax/dev/originals/remote_media/remote_media_processing_example/webrtc_examples/calibration_results_20250807_184400/camera_2_original.jpg',
        '/home/acidhax/dev/originals/remote_media/remote_media_processing_example/webrtc_examples/calibration_results_20250807_184400/camera_3_original.jpg'
    ]
    
    # Test configurations
    configs = [
        # Config 1: Original hardcoded values
        {"squares_x": 5, "squares_y": 4, "square_length": 0.03, "marker_length": 0.015, "name": "30mm/15mm"},
        # Config 2: Your specified values
        {"squares_x": 5, "squares_y": 4, "square_length": 0.04, "marker_length": 0.02, "name": "40mm/20mm"},
        # Config 3: Try different aspect ratio
        {"squares_x": 4, "squares_y": 3, "square_length": 0.04, "marker_length": 0.02, "name": "4x3 40mm/20mm"},
    ]
    
    print("=" * 80)
    print("ChAruco Detection Test - Multiple Configurations")
    print("=" * 80)
    
    results = {}
    
    for config in configs:
        print(f"\n{'#'*80}")
        print(f"# Testing Configuration: {config['name']}")
        print(f"# Board: {config['squares_x']}x{config['squares_y']}")
        print(f"# Square: {config['square_length']*1000:.1f}mm, Marker: {config['marker_length']*1000:.1f}mm")
        print(f"{'#'*80}")
        
        config_results = []
        
        for image_path in test_images:
            if os.path.exists(image_path):
                aruco_count, charuco_count = test_charuco_detection(
                    image_path,
                    config['squares_x'],
                    config['squares_y'],
                    config['square_length'],
                    config['marker_length']
                )
                config_results.append((os.path.basename(image_path), aruco_count, charuco_count))
            else:
                print(f"⚠️ Image not found: {image_path}")
        
        results[config['name']] = config_results
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS")
    print("=" * 80)
    
    for config_name, config_results in results.items():
        print(f"\n📋 Configuration: {config_name}")
        for image_name, aruco_count, charuco_count in config_results:
            status = "✅" if charuco_count > 0 else "❌"
            print(f"   {status} {image_name}: {aruco_count} ArUco → {charuco_count} ChAruco")
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    # Check which configuration worked best
    best_config = None
    best_total = 0
    
    for config_name, config_results in results.items():
        total_charuco = sum(r[2] for r in config_results)
        print(f"Config '{config_name}': Total ChAruco corners = {total_charuco}")
        if total_charuco > best_total:
            best_total = total_charuco
            best_config = config_name
    
    if best_config and best_total > 0:
        print(f"\n🏆 Best configuration: {best_config} with {best_total} total ChAruco corners")
    else:
        print(f"\n❌ No configuration successfully detected ChAruco corners!")
        print("\nPossible issues:")
        print("1. Board size/marker size mismatch with physical board")
        print("2. Dictionary mismatch (board might not be DICT_4X4_50)")
        print("3. Image quality issues (blur, lighting, perspective)")
        print("4. Board printing issues (markers not printed correctly)")


if __name__ == "__main__":
    main()