#!/usr/bin/env python3
"""
Main script for Raspberry Pi WebRTC Client

Command-line interface for the ChAruco calibration WebRTC client.
"""

import asyncio
import argparse
import logging
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client import RaspberryPiWebRTCClient

# Configure logging
def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Reduce noise from some loggers
    if not verbose:
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        logging.getLogger("websockets").setLevel(logging.WARNING)
        logging.getLogger("aiortc").setLevel(logging.WARNING)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Raspberry Pi WebRTC Client for ChAruco Calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --camera 0                          # Stream from camera 0
  %(prog)s --camera all                        # Stream from all cameras
  %(prog)s --camera 0,2,3                      # Stream from specific cameras
  %(prog)s --server ws://192.168.1.100:8081/ws # Custom server
  %(prog)s --width 1280 --height 720           # Custom resolution
  %(prog)s --list-cameras                      # List available cameras
  %(prog)s --test-cameras --camera all         # Test all cameras
        """
    )
    
    parser.add_argument(
        "--server", 
        default="ws://localhost:8081/ws",
        help="WebRTC server WebSocket URL (default: %(default)s)"
    )
    
    parser.add_argument(
        "--camera",
        default="0",
        help="Camera specification: camera index, 'all', or comma-separated indices (default: %(default)s)"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Video width in pixels (default: %(default)s)"
    )
    
    parser.add_argument(
        "--height", 
        type=int,
        default=480,
        help="Video height in pixels (default: %(default)s)"
    )
    
    parser.add_argument(
        "--fps",
        type=int, 
        default=30,
        help="Video FPS (default: %(default)s)"
    )
    
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available cameras and exit"
    )
    
    parser.add_argument(
        "--test-cameras",
        action="store_true",
        help="Test selected cameras and exit"
    )
    
    parser.add_argument(
        "--no-reconnect",
        action="store_true",
        help="Disable automatic reconnection"
    )
    
    parser.add_argument(
        "--max-reconnects",
        type=int,
        default=10,
        help="Maximum reconnection attempts (default: %(default)s)"
    )
    
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=5.0,
        help="Delay between reconnection attempts in seconds (default: %(default)s)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


async def main():
    """Main async function."""
    args = parse_arguments()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    # Create client
    client = RaspberryPiWebRTCClient(
        server_url=args.server,
        width=args.width,
        height=args.height,
        fps=args.fps,
        auto_reconnect=not args.no_reconnect,
        max_reconnect_attempts=args.max_reconnects,
        reconnect_delay=args.reconnect_delay
    )
    
    # Handle list cameras command
    if args.list_cameras:
        print("=== Available Cameras ===")
        client.list_cameras()
        return 0
    
    # Select cameras
    if not client.select_cameras(args.camera):
        logger.error("Failed to select cameras")
        print("\nTip: Use --list-cameras to see available cameras")
        return 1
    
    # Handle test cameras command
    if args.test_cameras:
        print("=== Testing Selected Cameras ===")
        results = await client.test_cameras()
        
        all_passed = True
        for camera_idx, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"Camera {camera_idx}: {status}")
            if not passed:
                all_passed = False
        
        print(f"\nOverall: {'✅ All cameras working' if all_passed else '❌ Some cameras failed'}")
        return 0 if all_passed else 1
    
    # Show configuration
    print("=== Raspberry Pi WebRTC Client ===")
    print(f"Server: {args.server}")
    print(f"Resolution: {args.width}x{args.height} @ {args.fps}fps")
    print(f"Selected cameras: {client.selected_cameras}")
    print(f"Auto-reconnect: {not args.no_reconnect}")
    print("")
    
    # List available cameras
    client.list_cameras()
    print("")
    print("🚀 Starting WebRTC client...")
    print("Press Ctrl+C to stop")
    print("")
    
    # Run client
    try:
        await client.run()
        return 0
    except KeyboardInterrupt:
        logger.info("Client stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Client failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def sync_main():
    """Synchronous wrapper for main."""
    try:
        return asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested")
        return 0
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(sync_main())