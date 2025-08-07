# ChAruco Multi-Camera Calibration and Perspective Warping

This module provides a complete pipeline for simultaneous camera calibration and perspective warping using ChAruco boards in multi-camera systems. It's designed for real-time processing of synchronized camera arrays.

## Overview

The system consists of four main components:

1. **ChAruco Detection Node** - Detects ChAruco boards and estimates camera poses
2. **Pose Diversity Selector Node** - Maintains a diverse set of calibration frames
3. **Perspective Warp Node** - Computes homographies and warps images to align views
4. **Multi-Camera Calibration Node** - Orchestrates the entire calibration pipeline

## ChAruco Board Configuration

The system is configured for the following ChAruco board specifications:
- **Board Size**: 27×17 squares
- **Square Length**: 9.2mm
- **Marker Length**: 6mm
- **Dictionary**: DICT_6X6_250
- **Margins**: 5.8mm
- **DPI**: 227

## Key Features

### Real-Time Processing
- Processes synchronized frames from multiple cameras
- Parallel ChAruco detection across all cameras
- Low-latency perspective warping for live preview

### Intelligent Frame Selection
- Uses pose diversity metrics to select optimal calibration frames
- Farthest-point sampling for maximum pose variance
- Automatic frame replacement when better diversity is found

### Robust Calibration
- Handles partial board detections gracefully
- Requires full board visibility for calibration frames
- Supports both automatic and manual calibration triggers

### Perspective Alignment
- Computes homographies from camera poses
- Real-time image warping to common reference frame
- Multiple blending modes (overlay, average, grid view)

## Usage

### Basic Usage

```python
import asyncio
from charuco import MultiCameraCalibrationNode, MultiCameraConfig, CharucoConfig

# Configure the system
config = MultiCameraConfig(
    num_cameras=4,
    charuco_config=CharucoConfig(
        squares_x=27,
        squares_y=17,
        square_length=0.0092,
        marker_length=0.006,
        dictionary="DICT_6X6_250"
    ),
    auto_calibrate=True,
    calibration_file="calibration.json"
)

# Create calibration node
calibration_node = MultiCameraCalibrationNode(config=config)

# Process synchronized frames
async def process_frames(frames, timestamp):
    result = await calibration_node.process({
        'frames': frames,  # List of numpy arrays
        'timestamp': timestamp,
        'blend_mode': 'overlay',
        'debug_visualization': True
    })
    
    return result['combined_view']  # Aligned composite view
```

### Individual Node Usage

```python
from charuco import CharucoDetectionNode, PerspectiveWarpNode

# ChAruco detection
detector = CharucoDetectionNode()
pose_result = await detector.process({
    'image': camera_frame,
    'camera_matrix': K,
    'dist_coeffs': dist,
    'camera_id': 0
})

# Perspective warping
warp_node = PerspectiveWarpNode()
warp_result = await warp_node.process({
    'images': [frame1, frame2, frame3],
    'poses': [pose1, pose2, pose3],
    'camera_matrices': [K1, K2, K3]
})
```

## Architecture

### Pipeline Flow
```
WebRTC Frames → ChAruco Detection → Pose Diversity Selection → Camera Calibration
                                                                        ↓
Combined View ← Perspective Warping ← Homography Computation ← Calibrated System
```

### Node Architecture
- **Async Processing**: All nodes support async/await patterns
- **Error Handling**: Graceful degradation when detection fails
- **State Management**: Maintains calibration state across frames
- **Caching**: Efficient homography caching for real-time performance

## Configuration Options

### CharucoConfig
- `squares_x/y`: Board dimensions
- `square_length`: Physical size of squares
- `marker_length`: Physical size of ArUco markers
- `dictionary`: ArUco dictionary type
- `margins`: Board margin size
- `dpi`: Print resolution

### WarpConfig
- `output_width/height`: Output image dimensions
- `reference_camera`: Index of reference camera
- `interpolation`: OpenCV interpolation method
- `border_mode`: Border handling method

### MultiCameraConfig
- `num_cameras`: Number of cameras in array
- `max_calibration_frames`: Maximum diverse frames to keep
- `min_frames_for_calibration`: Minimum frames needed for calibration
- `auto_calibrate`: Enable automatic calibration
- `calibration_file`: JSON file for saving/loading calibration

## Output Data Structure

```python
{
    'warped_frames': List[np.ndarray],      # Individual warped images
    'combined_view': np.ndarray,            # Composite aligned view
    'poses': List[PoseResult],              # Camera poses
    'homographies': List[np.ndarray],       # Transformation matrices
    'valid_cameras': List[int],             # Cameras with valid poses
    'calibration_status': {
        'calibrated': bool,                 # System calibration status
        'frames_collected': int,            # Frames in diverse set
        'frames_needed': int,               # Minimum frames required
        'cameras_calibrated': int           # Number of calibrated cameras
    },
    'statistics': {
        'frames_processed': int,            # Total frames processed
        'detection_rate': float,            # Success rate
        'valid_poses_current': int          # Valid poses in current frame
    }
}
```

## Error Handling

- **Missing Frames**: Returns previous valid results
- **Detection Failures**: Uses cached homographies when available
- **Calibration Errors**: Logs warnings and continues with defaults
- **Invalid Poses**: Filters out and continues with valid cameras

## Performance Considerations

- **Parallel Detection**: All cameras processed simultaneously
- **Homography Caching**: Avoids recomputation when poses are stable
- **Memory Management**: Limits calibration frame storage
- **Real-time Ready**: Designed for 30+ FPS processing

## Example Applications

1. **Multi-Camera Streaming**: Align multiple camera feeds in real-time
2. **3D Reconstruction**: Provide calibrated camera parameters
3. **Augmented Reality**: Enable accurate pose tracking
4. **Quality Control**: Monitor calibration drift over time

## Running the Example

```bash
cd remote_media_processing_example/webrtc_examples/charuco
python charuco_calibration_example.py
```

The example demonstrates:
- Mock camera array simulation
- Calibration data collection
- Pose diversity selection
- Real-time perspective warping
- Calibration file I/O

## Dependencies

- OpenCV (cv2)
- NumPy
- RemoteMedia Processing SDK
- Python 3.8+

## Calibration Tips

1. **Board Visibility**: Ensure the full ChAruco board is visible in all cameras
2. **Pose Diversity**: Move the board through different positions and orientations
3. **Lighting**: Use consistent, even lighting across all cameras
4. **Synchronization**: Ensure all cameras capture the same timestamp
5. **Stability**: Keep cameras stable during calibration process

## Future Enhancements

- Stereo calibration for depth estimation
- Automatic board detection and tracking
- Dynamic camera addition/removal
- GPU acceleration for real-time processing
- Advanced blending algorithms for seamless composites