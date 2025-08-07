#!/bin/bash
# Setup script for Raspberry Pi WebRTC Client

set -e

echo "=== Raspberry Pi WebRTC Client Setup ==="
echo ""

# Check if we're on a Raspberry Pi
if ! command -v raspi-config &> /dev/null; then
    echo "⚠️  This script is designed for Raspberry Pi, but can work on other Linux systems"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "📦 Installing system dependencies..."

# Update system
sudo apt update

# Install build dependencies
sudo apt install -y python3-dev python3-pip python3-venv
sudo apt install -y build-essential cmake pkg-config
sudo apt install -y libavformat-dev libavcodec-dev libavdevice-dev
sudo apt install -y libavutil-dev libswscale-dev libswresample-dev libavfilter-dev
sudo apt install -y libjpeg-dev libpng-dev libtiff-dev
sudo apt install -y libgtk-3-dev libcanberra-gtk3-dev
sudo apt install -y libatlas-base-dev gfortran
sudo apt install -y v4l-utils

# Raspberry Pi specific packages
if command -v raspi-config &> /dev/null; then
    echo "🍓 Installing Raspberry Pi specific packages..."
    sudo apt install -y python3-libcamera python3-kms++
    sudo apt install -y libcamera-dev libcamera-tools
fi

echo "📹 Setting up camera permissions..."
sudo usermod -a -G video $USER

echo "🐍 Setting up Python environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Created Python virtual environment"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

echo "📚 Installing Python dependencies..."

# Install main requirements
pip install -r requirements.txt

# Install Raspberry Pi camera support if available
if command -v raspi-config &> /dev/null; then
    echo "📷 Installing Raspberry Pi camera support..."
    pip install picamera2 || echo "⚠️  Could not install picamera2, Raspberry Pi camera won't be available"
fi

echo "🔧 Testing installation..."

# Test basic imports
python3 -c "
import cv2
import numpy as np
import aiortc
import websockets
print('✅ Core dependencies imported successfully')
"

# Test camera detection
echo "🔍 Detecting cameras..."
python3 main.py --list-cameras

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Test cameras: python3 main.py --test-cameras --camera all"
echo "3. Start streaming: python3 main.py --camera all"
echo ""
echo "💡 Tips:"
echo "• Log out and back in for camera permissions to take effect"
echo "• Use --server ws://YOUR_SERVER_IP:8081/ws for remote servers"
echo "• Check README.md for detailed usage instructions"
echo ""

# Check if reboot is needed for permissions
if ! groups $USER | grep -q video; then
    echo "⚠️  You may need to log out and back in for camera permissions to take effect"
fi