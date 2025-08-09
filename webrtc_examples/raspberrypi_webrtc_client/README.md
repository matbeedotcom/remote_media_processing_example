# Raspberry Pi WebRTC Client

A Python-based WebRTC client for Raspberry Pi that streams camera feeds to the ChAruco calibration server. Supports multiple cameras including USB cameras and Raspberry Pi cameras.

## Features

### Standard Video Streaming
- **Multi-camera support** - Stream from USB cameras, Raspberry Pi cameras, or both
- **Automatic camera detection** - Discovers and lists available cameras
- **Auto-resolution detection** - Automatically uses maximum available resolution for Picamera2
- **Real-time streaming** - Low-latency WebRTC video streaming
- **Automatic reconnection** - Handles network interruptions gracefully
- **Performance monitoring** - Built-in FPS and error tracking
- **Debug preview** - Save and analyze frames being sent
- **Flexible configuration** - Command-line and programmatic configuration

### RAW10 Data Channel Transfer (New)
- **Uncompressed sensor data** - Transfer RAW10/RAW8 bayer data directly
- **Data channel transport** - Bypass video compression for maximum quality
- **Compression options** - None, zlib, or lz4 compression
- **Multi-camera synchronization** - Hardware-synchronized quad arrays
- **Dual mode operation** - Video preview + RAW data simultaneously
- **VLBI optimized** - Designed for interferometric applications

## Hardware Requirements

- Raspberry Pi (3B+, 4, or newer recommended)
- USB cameras and/or Raspberry Pi camera module
- Network connection to ChAruco calibration server
- Adequate power supply for multiple cameras

## Software Requirements

- Python 3.7+
- OpenCV
- aiortc and WebRTC dependencies
- Optional: picamera2 (for Raspberry Pi camera)

## Installation

### 1. Install System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required system packages
sudo apt install -y python3-dev python3-pip
sudo apt install -y libavformat-dev libavcodec-dev libavdevice-dev
sudo apt install -y libavutil-dev libswscale-dev libswresample-dev libavfilter-dev
sudo apt install -y pkg-config

# For Raspberry Pi camera (optional)
sudo apt install -y python3-libcamera python3-kms++
```

### 2. Install Python Dependencies

```bash
# Install main dependencies
pip3 install -r requirements.txt

# For Raspberry Pi camera support (optional)
pip3 install picamera2
```

### 3. Setup Permissions (if using USB cameras)

```bash
# Add user to video group
sudo usermod -a -G video $USER

# Log out and back in for group changes to take effect
```

## Usage

### Basic Video Streaming

```bash
# Stream from camera 0
python3 main.py --camera 0

# Stream from all detected cameras
python3 main.py --camera all

# Stream from specific cameras
python3 main.py --camera 0,2,3

# Connect to remote server
python3 main.py --server ws://192.168.1.100:8081/ws --camera all
```

### RAW10 Data Channel Mode

```bash
# Single camera RAW10 transfer
python3 raw10_webrtc_client.py --server http://localhost:8082 --camera 0 --bit-depth 10

# Quad camera array with RAW10
python3 raw10_webrtc_client.py --quad --cameras 0 1 2 3 --compression zlib

# RAW8 for higher framerate
python3 raw10_webrtc_client.py --bit-depth 8 --fps 50 --width 2560 --height 400

# Dual mode: Video preview + RAW data
python3 raw10_webrtc_client.py --video-preview --video-width 640 --video-height 480

# No compression for lowest latency
python3 raw10_webrtc_client.py --compression none --fps 15
```

### Camera Management

```bash
# List available cameras
python3 main.py --list-cameras

# Test selected cameras
python3 main.py --test-cameras --camera all

# Test specific camera
python3 main.py --test-cameras --camera 0
```

### Resolution Configuration

```bash
# Use maximum available resolution (default for Picamera2)
python3 main.py --camera 0

# Use maximum resolution explicitly
python3 main.py --max-resolution

# Use resolution presets
python3 main.py --resolution 1080p  # 1920x1080
python3 main.py --resolution 720p   # 1280x720
python3 main.py --resolution 4k     # 3840x2160
python3 main.py --resolution max    # Maximum available

# Custom resolution
python3 main.py --width 1280 --height 720

# Custom frame rate
python3 main.py --fps 30
```

### Advanced Configuration

```bash
# Disable auto-reconnection
python3 main.py --no-reconnect

# Custom reconnection settings
python3 main.py --max-reconnects 5 --reconnect-delay 10

# Verbose logging
python3 main.py --verbose

# Debug preview mode
python3 main.py --debug-preview

# Save preview frames
python3 main.py --save-preview-frames --preview-dir /tmp/frames
```

## Command Line Options

### Standard Video Client (main.py)

| Option | Description | Default |
|--------|-------------|---------|
| `--server` | WebRTC server WebSocket URL | `ws://localhost:8081/ws` |
| `--camera` | Camera specification (index, 'all', or comma-separated) | `0` |
| `--width` | Video width in pixels | Auto-detect maximum |
| `--height` | Video height in pixels | Auto-detect maximum |
| `--max-resolution` | Use maximum available resolution | Auto for Picamera2 |
| `--resolution` | Preset resolution (max, 4k, 1080p, 720p, 480p, vga) | - |
| `--fps` | Video frame rate | `30` |
| `--list-cameras` | List available cameras and exit | - |
| `--test-cameras` | Test selected cameras and exit | - |
| `--no-reconnect` | Disable automatic reconnection | - |
| `--max-reconnects` | Maximum reconnection attempts | `10` |
| `--reconnect-delay` | Delay between reconnection attempts (seconds) | `5.0` |
| `--verbose` | Enable verbose logging | - |
| `--debug-preview` | Enable frame analysis and debugging | - |
| `--save-preview-frames` | Save preview frames to disk | - |
| `--preview-dir` | Directory for preview frames | `/tmp/webrtc_preview` |
| `--preview-interval` | Save every N frames | `30` |

### RAW10 Data Channel Client (raw10_webrtc_client.py)

| Option | Description | Default |
|--------|-------------|---------|
| `--server` | WebRTC server URL | `http://localhost:8082` |
| `--camera` | Camera index | `0` |
| `--quad` | Use quad camera array | - |
| `--cameras` | Camera indices for quad mode | `[0, 1, 2, 3]` |
| `--width` | RAW capture width | `3280` |
| `--height` | RAW capture height | `2464` |
| `--bit-depth` | Bit depth (8 or 10) | `10` |
| `--fps` | Capture framerate | `15.0` |
| `--compression` | Compression type (none, zlib, lz4) | `zlib` |
| `--video-preview` | Enable video preview stream | - |
| `--video-width` | Video preview width | `640` |
| `--video-height` | Video preview height | `480` |
| `--video-fps` | Video preview framerate | `30.0` |
| `--verbose` | Verbose logging | - |

## Camera Types

### USB Cameras
- Automatically detected via OpenCV
- Support for UVC (USB Video Class) cameras
- Multiple cameras supported simultaneously

### Raspberry Pi Camera
- Requires picamera2 library
- Better integration with Pi hardware
- Lower CPU usage compared to USB cameras
- **Automatic maximum resolution detection**
- Supports high resolution cameras (HQ Camera, Camera Module 3)

#### Supported Raspberry Pi Cameras

| Camera Model | Max Resolution | Default in Client |
|--------------|---------------|-------------------|
| Camera Module v1 | 2592×1944 | Uses maximum |
| Camera Module v2 | 3280×2464 | Uses maximum |
| Camera Module 3 | 4608×2592 | Uses maximum |
| HQ Camera | 4056×3040 | Uses maximum |
| Arducam variants | Varies | Auto-detected |

## Troubleshooting

### Camera Not Detected

```bash
# Check USB cameras
lsusb | grep -i camera

# Check video devices
ls -la /dev/video*

# Test with v4l2
v4l2-ctl --list-devices

# List cameras with our tool
python3 main.py --list-cameras
```

### Permission Issues

```bash
# Check video group membership
groups $USER

# Add user to video group if missing
sudo usermod -a -G video $USER
# Log out and back in
```

### Performance Issues

```bash
# Monitor system resources
htop

# Check USB bandwidth (for USB cameras)
lsusb -t

# Reduce resolution/FPS
python3 main.py --width 320 --height 240 --fps 15
```

### Connection Issues

```bash
# Test network connectivity
ping [server-ip]

# Check firewall settings
sudo ufw status

# Test WebSocket connection
curl -i -N -H "Connection: Upgrade" \
     -H "Upgrade: websocket" \
     -H "Sec-WebSocket-Key: test" \
     -H "Sec-WebSocket-Version: 13" \
     http://[server-ip]:8081/ws
```

## Performance Tips

1. **USB Bandwidth**: Limit the number of high-resolution USB cameras
2. **Power Supply**: Ensure adequate power for multiple cameras
3. **Network**: Use wired connection for best reliability
4. **Resolution**: Start with lower resolutions and increase as needed
5. **Frame Rate**: 15-30 FPS is usually sufficient for calibration

## Integration with ChAruco Server

This client is designed to work with the ChAruco WebRTC calibration server. The server expects:

- Synchronized video streams from multiple cameras
- Camera identification in video overlays
- Proper WebRTC signaling protocol

## Example Systemd Service

Create `/etc/systemd/system/charuco-client.service`:

```ini
[Unit]
Description=ChAruco WebRTC Client
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/charuco-client
ExecStart=/usr/bin/python3 main.py --camera all --server ws://192.168.1.100:8081/ws
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable charuco-client
sudo systemctl start charuco-client
sudo systemctl status charuco-client
```

## API Usage

```python
from raspberrypi_webrtc_client import RaspberryPiWebRTCClient
import asyncio

async def main():
    client = RaspberryPiWebRTCClient(
        server_url="ws://localhost:8081/ws",
        width=640,
        height=480,
        fps=30
    )
    
    # Select cameras
    client.select_cameras("all")
    
    # Run client
    await client.run()

if __name__ == "__main__":
    asyncio.run(main())
```

## License

This software is part of the RemoteMedia Processing SDK and follows the same license terms.