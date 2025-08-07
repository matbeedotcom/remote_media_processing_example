#!/usr/bin/env python3
"""
Basic WebRTC Server Test

Simple WebRTC server without ChAruco pipeline to test basic connectivity.
This helps isolate whether the issue is with WebRTC fundamentals or the pipeline.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "remote_media_processing"))

from remotemedia.core.pipeline import Pipeline
from remotemedia.core.node import Node
from remotemedia.webrtc import WebRTCServer, WebRTCConfig
from remotemedia.nodes import PassThroughNode
from video_stream_analyzer import VideoStreamAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleVideoProcessor(Node):
    """Simple video processing node that just passes frames through."""
    
    def __init__(self, name="SimpleVideoProcessor"):
        super().__init__(name=name)
        self.frame_count = 0
    
    async def process(self, data):
        """Process video frames - just log and pass through."""
        try:
            if isinstance(data, dict) and 'frame' in data:
                self.frame_count += 1
                if self.frame_count % 30 == 0:  # Log every 30 frames (~1 second at 30fps)
                    logger.info(f"📹 Processed {self.frame_count} video frames")
                return data
            else:
                logger.debug(f"📄 Received non-video data: {type(data)}")
                return data
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return data


def create_simple_pipeline():
    """Create a simple pass-through pipeline for testing."""
    pipeline = Pipeline()
    
    # Video stream analyzer for detailed frame logging
    analyzer = VideoStreamAnalyzer(name="FrameAnalyzer", log_interval=30)  # Log every 30 frames
    pipeline.add_node(analyzer)
    
    # Simple video processor
    video_processor = SimpleVideoProcessor(name="VideoProcessor")
    pipeline.add_node(video_processor)
    
    # Pass through node to keep pipeline flowing
    pass_through = PassThroughNode(name="PassThrough")
    pipeline.add_node(pass_through)
    
    logger.info("🔧 Created simple test pipeline with video stream analyzer")
    return pipeline


async def create_simple_webrtc_server(host="0.0.0.0", port=8081):
    """Create a basic WebRTC server for testing."""
    
    config = WebRTCConfig(
        host=host,
        port=port,
        enable_cors=True,
        stun_servers=["stun:stun.l.google.com:19302"],
        static_files_path=str(Path(__file__).parent)
    )
    
    # Create server with simple pipeline
    server = WebRTCServer(config=config, pipeline_factory=create_simple_pipeline)
    
    return server


async def main():
    """Main test function."""
    HOST = "0.0.0.0"
    PORT = 8081
    
    logger.info("=== Basic WebRTC Server Test ===")
    logger.info(f"Starting simple WebRTC server on {HOST}:{PORT}")
    logger.info("This server has NO ChAruco processing - just basic video pass-through")
    logger.info("")
    
    # Create and start server
    server = await create_simple_webrtc_server(host=HOST, port=PORT)
    
    try:
        await server.start()
        
        logger.info("✅ Basic WebRTC server is running!")
        logger.info("")
        logger.info("🔌 Test with Raspberry Pi client:")
        logger.info(f"   python main.py --server ws://{HOST}:{PORT}/ws --camera 0")
        logger.info("")
        logger.info("📊 This will help determine if the issue is:")
        logger.info("   • Basic WebRTC connectivity (if this fails)")
        logger.info("   • ChAruco pipeline specific (if this works)")
        logger.info("")
        logger.info("Press Ctrl+C to stop")
        
        # Keep server running
        while True:
            await asyncio.sleep(10)
            
            # Log connection stats
            connections_count = len(server.connections)
            if connections_count > 0:
                logger.info(f"📊 Active connections: {connections_count}")
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down test server...")
    except Exception as e:
        logger.error(f"❌ Server error: {e}", exc_info=True)
    finally:
        await server.stop()
        logger.info("✅ Test server stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"❌ Failed to start test server: {e}", exc_info=True)