# ChAruco Camera Calibration System

A comprehensive multi-camera calibration system using ChAruco boards for precise camera intrinsic and extrinsic calibration, supporting stereo and multi-view camera arrays.

## Features

- **Individual Camera Calibration**: Full-screen calibration for each camera independently
- **Multi-Camera Calibration**: Stereo and multi-view calibration for camera arrays
- **Pose Diversity Selection**: Automatic selection of diverse calibration frames
- **Real-time Visualization**: Live preview during calibration process
- **Physical Units**: Calibration with real-world measurements (mm, degrees, arcseconds)
- **VLBI Support**: Integration with optical Very Long Baseline Interferometry systems

## Directory Structure

```
charuco/
├── calibration/          # Calibration pipeline modules
│   ├── individual_camera_calibration_node.py
│   ├── multi_camera_calibration_node.py
│   └── full_calibration_pipeline.py
├── nodes/               # Processing nodes
│   ├── charuco_detection_node.py
│   ├── pose_diversity_selector_node.py
│   ├── perspective_warp_node.py
│   └── ...
├── utils/               # Utility modules
│   ├── sensor_config.py
│   └── generate_calibration_visualization.py
├── config/              # Configuration files
│   ├── charuco_board_config.json
│   ├── sensor_database.json
│   └── camera_system_config.json
├── examples/            # Example scripts
│   ├── full_calibration_example.py
│   └── charuco_calibration_example.py
├── vlbi/               # VLBI-specific modules
└── docs/               # Documentation
```

## Quick Start

### 1. Generate ChAruco Board

First, generate a ChAruco board for printing:

```bash
python examples/full_calibration_example.py --generate-board
```

This creates a `charuco_board_7x5.png` file. Print it at actual size for calibration.

### 2. Run Calibration

#### With Real Cameras

```bash
# Calibrate cameras 0, 1, 2, 3
python examples/full_calibration_example.py --cameras 0 1 2 3
```

#### With Test Data

```bash
python examples/full_calibration_example.py --test
```

### 3. Calibration Process

The calibration follows a two-phase approach:

1. **Individual Calibration Phase**:
   - Show the ChAruco board to each camera individually
   - Move the board to cover the entire field of view
   - System guides you through each camera sequentially

2. **Multi-Camera Calibration Phase**:
   - Show the board visible to multiple cameras simultaneously
   - Move the board to different positions and orientations
   - Keep the board in view of at least 2 cameras

## API Usage

### Basic Calibration Pipeline

```python
from charuco.calibration import FullCalibrationPipeline, FullCalibrationConfig
from charuco.nodes import CharucoConfig

# Configure ChAruco board
charuco_config = CharucoConfig(
    squares_x=7,
    squares_y=5,
    square_length=0.035,  # 35mm
    marker_length=0.025,   # 25mm
    dictionary="DICT_4X4_50"
)

# Configure pipeline
config = FullCalibrationConfig(
    num_cameras=4,
    charuco_config=charuco_config,
    individual_frames_per_camera=25,
    individual_min_coverage=70.0,
    multi_camera_frames=20,
    enable_stereo_calibration=True,
    calibration_output_dir="calibration_results"
)

# Create and run pipeline
pipeline = FullCalibrationPipeline(config)

# Process frames
result = await pipeline.process({
    'frames': camera_frames,
    'timestamp': timestamp
})
```

### Individual Camera Calibration Only

```python
from charuco.calibration import IndividualCameraCalibrationNode, IndividualCalibrationConfig

config = IndividualCalibrationConfig(
    num_cameras=4,
    frames_per_camera=20,
    min_coverage_percentage=75.0
)

calibrator = IndividualCameraCalibrationNode(config)
result = await calibrator.process({'frames': frames})
```

### Multi-Camera Calibration Only

```python
from charuco.calibration import MultiCameraCalibrationNode, MultiCameraConfig

config = MultiCameraConfig(
    num_cameras=4,
    max_calibration_frames=15,
    enable_stereo_calibration=True
)

calibrator = MultiCameraCalibrationNode(config)
result = await calibrator.process({'frames': frames})
```

## Configuration

### ChAruco Board Configuration

Edit `config/charuco_board_config.json`:

```json
{
    "squares_x": 7,
    "squares_y": 5,
    "square_length": 0.035,
    "marker_length": 0.025,
    "dictionary": "DICT_4X4_50",
    "margins": 0.005,
    "dpi": 200
}
```

### Sensor Configuration

Edit `config/sensor_database.json` to add camera sensor specifications:

```json
{
    "OV9281": {
        "name": "OV9281",
        "sensor_width_mm": 3.896,
        "sensor_height_mm": 2.922,
        "resolution_width": 1280,
        "resolution_height": 960,
        "pixel_pitch_um": 3.0,
        "quantum_efficiency": 0.56
    }
}
```

### Camera System Configuration

Edit `config/camera_system_config.json`:

```json
{
    "num_cameras": 4,
    "mounting_pattern": "square",
    "camera_spacing_mm": 100.0,
    "reference_camera_id": 0
}
```

## Output Files

After calibration, the following files are generated:

```
calibration_results/
├── individual_calibration_camera_0.json    # Individual calibrations
├── individual_calibration_camera_1.json
├── multi_camera_calibration.json          # Multi-camera calibration
├── stereo_calibration.json                # Stereo pairs calibration
├── calibration_report.json                # Comprehensive report
├── calibration_summary.txt                # Human-readable summary
└── calibration_results_*/                 # Timestamped results
    ├── homographies.json                   # Perspective transforms
    ├── combined_view.jpg                   # Merged camera view
    └── camera_*_original.jpg              # Individual camera views
```

## Calibration Quality Metrics

The system provides several quality metrics:

- **Reprojection Error**: RMS error in pixels (target: < 0.5 pixels)
- **Screen Coverage**: Percentage of FOV covered during calibration (target: > 70%)
- **Full Board Detections**: Number of frames with complete board visible
- **Stereo Baseline**: Physical distance between cameras (mm)
- **Convergence Angle**: Angle between optical axes (degrees)

## Advanced Features

### VLBI Integration

For optical interferometry applications:

```python
from charuco.vlbi import VLBICalibration

vlbi_cal = VLBICalibration(baseline_meters=10.0)
vlbi_cal.compute_uv_coverage(calibration_data)
```

### Custom Processing Nodes

Create custom nodes by inheriting from the base Node class:

```python
from remotemedia.core.node import Node

class CustomProcessingNode(Node):
    async def process(self, data):
        # Your processing logic
        return processed_data
```

## Troubleshooting

### Common Issues

1. **No ChAruco Detection**
   - Ensure proper lighting (avoid shadows and glare)
   - Check board is flat and not curved
   - Verify correct board configuration matches printed board

2. **Low Coverage Percentage**
   - Move board to corners and edges of camera view
   - Use slower, smoother movements
   - Ensure board fills significant portion of frame

3. **High Calibration Error**
   - Collect more diverse poses
   - Check for motion blur
   - Verify board measurements are accurate

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Requirements

- Python 3.8+
- OpenCV 4.5+ with contrib modules
- NumPy
- RemoteMedia SDK

## Installation

```bash
# Install dependencies
pip install opencv-contrib-python numpy

# Install RemoteMedia SDK
pip install -e .
```

## License

See LICENSE file in the repository root.

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style conventions
- All tests pass
- Documentation is updated
- Imports use relative paths within the package

## Support

For issues and questions, please open an issue on the project repository.