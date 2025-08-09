#!/usr/bin/env python3
"""
Debug script to analyze homography computation and saving issues.
"""

import json
import numpy as np
from pathlib import Path

def analyze_homography_file(homo_path):
    """Analyze a homography JSON file."""
    print(f"\n🔍 Analyzing: {homo_path}")
    
    try:
        with open(homo_path, 'r') as f:
            homo_data = json.load(f)
        
        print(f"📊 Found {len(homo_data)} homography matrices:")
        
        for camera_key, matrix_data in homo_data.items():
            matrix = np.array(matrix_data)
            is_identity = np.allclose(matrix, np.eye(3), atol=1e-6)
            
            print(f"  {camera_key}:")
            print(f"    Shape: {matrix.shape}")
            print(f"    Is Identity: {is_identity}")
            if not is_identity:
                print(f"    Matrix:\n{matrix}")
            else:
                print(f"    Matrix: Identity")
            print()
            
    except Exception as e:
        print(f"❌ Error reading {homo_path}: {e}")

def main():
    """Main function to analyze recent calibration results."""
    results_dir = Path(".")
    
    # Find all calibration result directories
    calibration_dirs = [d for d in results_dir.iterdir() 
                       if d.is_dir() and d.name.startswith("calibration_results_")]
    
    # Sort by name (which includes timestamp) and get the most recent ones
    calibration_dirs.sort(key=lambda x: x.name, reverse=True)
    recent_dirs = calibration_dirs[:3]  # Check last 3 runs
    
    print("🔍 Analyzing recent calibration homography files...")
    
    for cal_dir in recent_dirs:
        homo_file = cal_dir / "homographies.json"
        if homo_file.exists():
            analyze_homography_file(homo_file)
        else:
            print(f"❌ No homographies.json found in {cal_dir}")
    
    # Also check for the most recent one mentioned in the user's output
    specific_file = Path("calibration_results_20250808_002011/homographies.json")
    if specific_file.exists():
        print("=" * 60)
        print("🎯 Analyzing the specific file mentioned by user:")
        analyze_homography_file(specific_file)

if __name__ == "__main__":
    main()