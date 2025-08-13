#!/usr/bin/env python3
"""
Debug script to capture and analyze ArUco detection issues in real-time.
Saves frames when ArUco markers are detected for analysis.
"""

import cv2
import numpy as np
import os
from datetime import datetime
import json

def analyze_aruco_detection(frame, frame_number):
    """Analyze ArUco detection in a frame and save debug info."""
    
    # Create debug directory if it doesn't exist
    debug_dir = f"aruco_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Initialize ArUco dictionary - try multiple dictionaries
    dictionaries = [
        ("DICT_4X4_50", cv2.aruco.DICT_4X4_50),
        ("DICT_4X4_100", cv2.aruco.DICT_4X4_100),
        ("DICT_4X4_250", cv2.aruco.DICT_4X4_250),
        ("DICT_4X4_1000", cv2.aruco.DICT_4X4_1000),
    ]
    
    results = {}
    
    for dict_name, dict_id in dictionaries:
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        detector_params = cv2.aruco.DetectorParameters()
        aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
        
        # Detect markers
        corners, ids, rejected = aruco_detector.detectMarkers(gray)
        
        if ids is not None and len(ids) > 0:
            results[dict_name] = {
                "count": len(ids),
                "ids": ids.flatten().tolist(),
                "rejected": len(rejected) if rejected else 0
            }
            
            # Draw detection on frame
            vis_frame = frame.copy()
            cv2.aruco.drawDetectedMarkers(vis_frame, corners, ids)
            
            # Add text overlay
            cv2.putText(vis_frame, f"Dict: {dict_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Detected: {len(ids)} markers", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"IDs: {ids.flatten().tolist()}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Save visualization
            filename = f"{debug_dir}/frame_{frame_number:04d}_{dict_name}.jpg"
            cv2.imwrite(filename, vis_frame)
    
    # Save analysis results
    if results:
        with open(f"{debug_dir}/frame_{frame_number:04d}_analysis.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nFrame {frame_number} Analysis:")
        print("-" * 40)
        for dict_name, data in results.items():
            print(f"{dict_name}: {data['count']} markers")
            print(f"  IDs: {data['ids']}")
            
            # Check for issues
            unique_ids = set(data['ids'])
            if len(unique_ids) != len(data['ids']):
                print(f"  ⚠️ DUPLICATE IDs DETECTED!")
            
            # Check expected range for 5x4 ChAruco
            if dict_name == "DICT_4X4_50":
                expected_ids = set(range(10))  # 0-9 for 5x4 board
                unexpected = unique_ids - expected_ids
                if unexpected:
                    print(f"  ⚠️ UNEXPECTED IDs for 5x4 ChAruco: {unexpected}")
                    print(f"  Expected: 0-9, Got: {unique_ids}")
    
    return results


def capture_from_webcam():
    """Capture frames from webcam and analyze ArUco detection."""
    
    cap = cv2.VideoCapture(0)  # Use default camera
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Press 'SPACE' to capture and analyze frame")
    print("Press 'Q' to quit")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break
        
        # Show live preview
        preview = frame.copy()
        
        # Quick ArUco detection for preview
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detector_params = cv2.aruco.DetectorParameters()
        aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
        corners, ids, _ = aruco_detector.detectMarkers(gray)
        
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(preview, corners, ids)
            cv2.putText(preview, f"Detected: {len(ids)} markers", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('ArUco Detection Debug', preview)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space to capture
            frame_count += 1
            print(f"\nCapturing frame {frame_count}...")
            analyze_aruco_detection(frame, frame_count)
            
        elif key == ord('q'):  # Q to quit
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Analyze existing image
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            print(f"Analyzing: {image_path}")
            frame = cv2.imread(image_path)
            analyze_aruco_detection(frame, 0)
        else:
            print(f"File not found: {image_path}")
    else:
        # Capture from webcam
        capture_from_webcam()