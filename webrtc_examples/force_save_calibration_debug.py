#!/usr/bin/env python3
"""
Force save calibration debug images to analyze ArUco detection issues.
This modifies the multi_camera_calibration_node to save debug images on every Nth frame.
"""

import os
import sys

# Path to the multi_camera_calibration_node.py file
node_path = "/home/acidhax/dev/originals/remote_media/remote_media_processing_example/webrtc_examples/charuco/multi_camera_calibration_node.py"

def add_debug_saving():
    """Add debug image saving to the calibration node."""
    
    with open(node_path, 'r') as f:
        lines = f.readlines()
    
    # Find the process method
    for i, line in enumerate(lines):
        if "self.frames_processed += 1" in line:
            # Add debug saving after frame counter increment
            insert_point = i + 1
            
            debug_code = """            # DEBUG: Save frames periodically for analysis
            if self.frames_processed % 100 == 0:  # Save every 100 frames
                import cv2
                from datetime import datetime
                debug_dir = f"debug_frames_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.makedirs(debug_dir, exist_ok=True)
                
                for idx, frame in enumerate(frames):
                    if frame is not None:
                        # Draw ArUco detection info
                        debug_frame = frame.copy()
                        if idx < len(poses):
                            pose = poses[idx]
                            info_text = f"Cam {idx}: "
                            if pose.aruco_ids is not None:
                                aruco_count = len(pose.aruco_ids)
                                info_text += f"{aruco_count} ArUco, IDs: {pose.aruco_ids.flatten().tolist()[:10]}"
                            else:
                                info_text += "No ArUco"
                            
                            cv2.putText(debug_frame, info_text, (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                            if pose.charuco_corners is not None:
                                charuco_text = f"ChAruco: {len(pose.charuco_corners)} corners"
                            else:
                                charuco_text = "ChAruco: 0 corners"
                            cv2.putText(debug_frame, charuco_text, (10, 60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        filename = f"{debug_dir}/cam_{idx}_frame_{self.frames_processed:05d}.jpg"
                        cv2.imwrite(filename, debug_frame)
                
                logger.info(f"📸 DEBUG: Saved frames to {debug_dir}/")
"""
            
            # Check if debug code already exists
            if "DEBUG: Save frames periodically" not in ''.join(lines):
                lines.insert(insert_point, debug_code)
                
                with open(node_path, 'w') as f:
                    f.writelines(lines)
                
                print("✅ Added debug saving to multi_camera_calibration_node.py")
                print("📸 Frames will be saved every 100 frames to debug_frames_* directories")
                return True
            else:
                print("⚠️ Debug saving already added")
                return False
    
    print("❌ Could not find insertion point")
    return False


def remove_debug_saving():
    """Remove debug saving from the calibration node."""
    
    with open(node_path, 'r') as f:
        content = f.read()
    
    if "DEBUG: Save frames periodically" in content:
        lines = content.split('\n')
        new_lines = []
        skip_until_next_dedent = False
        
        for line in lines:
            if "DEBUG: Save frames periodically" in line:
                skip_until_next_dedent = True
                continue
            
            if skip_until_next_dedent:
                # Skip until we reach the original indentation level
                if line and not line.startswith('            #') and not line.startswith('                '):
                    if line.strip() and not line.startswith('            '):
                        skip_until_next_dedent = False
                        new_lines.append(line)
            else:
                new_lines.append(line)
        
        with open(node_path, 'w') as f:
            f.write('\n'.join(new_lines))
        
        print("✅ Removed debug saving from multi_camera_calibration_node.py")
        return True
    else:
        print("⚠️ Debug saving not found")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "remove":
        remove_debug_saving()
    else:
        if add_debug_saving():
            print("\n📝 Next steps:")
            print("1. Restart the server")
            print("2. Show the ChAruco board to the camera")
            print("3. Wait for 100 frames (~3 seconds)")
            print("4. Check the debug_frames_* directory for saved images")
            print("\nTo remove debug saving, run: python force_save_calibration_debug.py remove")