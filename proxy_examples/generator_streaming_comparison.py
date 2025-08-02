"""
Comparison of generator behavior: Local vs Remote with RemoteProxyClient.
"""

import asyncio
import time
from typing import AsyncIterator
from remotemedia.core.node import RemoteExecutorConfig
from remotemedia.remote.proxy_client import RemoteProxyClient


class DataStreamer:
    """Example class with generator methods."""
    
    def __init__(self):
        self.chunk_size = 1024
    
    def read_chunks(self, filename: str, num_chunks: int = 5):
        """Simulate reading a file in chunks."""
        print(f"[Server] Starting to read {filename}")
        for i in range(num_chunks):
            time.sleep(0.1)  # Simulate I/O
            chunk = f"Chunk {i+1}/{num_chunks} from {filename}"
            print(f"[Server] Yielding: {chunk}")
            yield chunk
        print(f"[Server] Finished reading {filename}")
    
    async def stream_data(self, source: str, count: int = 5):
        """Simulate streaming real-time data."""
        print(f"[Server] Starting stream from {source}")
        for i in range(count):
            await asyncio.sleep(0.1)  # Simulate real-time delay
            data = {"seq": i, "value": i * 10, "source": source}
            print(f"[Server] Streaming: {data}")
            yield data
        print(f"[Server] Stream complete")


async def test_local_behavior():
    """Show how generators work locally."""
    print("=== LOCAL GENERATOR BEHAVIOR ===\n")
    
    streamer = DataStreamer()
    
    # 1. Sync generator - process as items arrive
    print("1. Local sync generator:")
    start = time.time()
    for chunk in streamer.read_chunks("local.dat", 3):
        print(f"   [Client] Processing: {chunk}")
        # Can process immediately
    print(f"   Total time: {time.time() - start:.2f}s\n")
    
    # 2. Async generator - process as items arrive
    print("2. Local async generator:")
    start = time.time()
    async for data in streamer.stream_data("sensor", 3):
        print(f"   [Client] Got data: {data}")
        # Can process immediately
    print(f"   Total time: {time.time() - start:.2f}s\n")
    
    # 3. Early termination
    print("3. Early termination:")
    count = 0
    async for data in streamer.stream_data("sensor", 10):
        print(f"   [Client] Got: {data}")
        count += 1
        if data["value"] >= 20:
            print("   [Client] Stopping early!")
            break  # Generator stops producing
    print(f"   Only processed {count} items (not all 10)\n")


async def test_remote_behavior():
    """Show how generators work with RemoteProxyClient."""
    print("\n=== REMOTE GENERATOR BEHAVIOR (Current) ===\n")
    
    config = RemoteExecutorConfig(host="localhost", port=50052, ssl_enabled=False)
    
    async with RemoteProxyClient(config) as client:
        streamer = DataStreamer()
        remote = await client.create_proxy(streamer)
        
        # 1. Sync generator - gets materialized list
        print("1. Remote sync generator:")
        start = time.time()
        chunks = await remote.read_chunks("remote.dat", 3)
        print(f"   [Client] Got type: {type(chunks)}")
        print(f"   [Client] All chunks received after: {time.time() - start:.2f}s")
        for chunk in chunks:
            print(f"   [Client] Processing: {chunk}")
        print()
        
        # 2. Async generator - gets materialized list
        print("2. Remote async generator:")
        start = time.time()
        data_list = await remote.stream_data("sensor", 3)
        print(f"   [Client] Got type: {type(data_list)}")
        print(f"   [Client] All data received after: {time.time() - start:.2f}s")
        for data in data_list:
            print(f"   [Client] Processing: {data}")
        print()
        
        # 3. No early termination possible
        print("3. Cannot terminate early:")
        all_data = await remote.stream_data("sensor", 10)
        print(f"   [Client] Got all {len(all_data)} items (can't stop early)")
        print(f"   [Client] Must process entire list even if we only want first few")


async def show_practical_workaround():
    """Show practical workaround for streaming."""
    print("\n\n=== PRACTICAL WORKAROUND ===\n")
    
    class ChunkedDataStreamer(DataStreamer):
        """Enhanced streamer with chunked access methods."""
        
        def __init__(self):
            super().__init__()
            self._streams = {}  # Store active streams
        
        def init_stream(self, stream_id: str, filename: str, num_chunks: int):
            """Initialize a stream and store generator."""
            self._streams[stream_id] = {
                "generator": self.read_chunks(filename, num_chunks),
                "done": False
            }
            return {"stream_id": stream_id, "ready": True}
        
        def get_next_chunk(self, stream_id: str):
            """Get next chunk from stream."""
            if stream_id not in self._streams:
                return {"error": "Stream not found"}
            
            stream = self._streams[stream_id]
            if stream["done"]:
                return {"done": True}
            
            try:
                chunk = next(stream["generator"])
                return {"chunk": chunk, "done": False}
            except StopIteration:
                stream["done"] = True
                return {"done": True}
        
        def close_stream(self, stream_id: str):
            """Close and cleanup stream."""
            if stream_id in self._streams:
                del self._streams[stream_id]
                return {"closed": True}
            return {"error": "Stream not found"}
    
    config = RemoteExecutorConfig(host="localhost", port=50052, ssl_enabled=False)
    
    async with RemoteProxyClient(config) as client:
        streamer = ChunkedDataStreamer()
        remote = await client.create_proxy(streamer)
        
        print("Streaming with chunked access:")
        
        # Initialize stream
        stream_info = await remote.init_stream("s1", "chunked.dat", 5)
        print(f"Stream initialized: {stream_info}")
        
        # Get chunks one by one
        chunk_count = 0
        while True:
            result = await remote.get_next_chunk("s1")
            
            if result.get("done"):
                break
            
            chunk = result["chunk"]
            chunk_count += 1
            print(f"   [Client] Got chunk {chunk_count}: {chunk}")
            
            # Can stop early if needed
            if chunk_count >= 3:
                print("   [Client] Stopping early!")
                break
        
        # Cleanup
        await remote.close_stream("s1")
        print(f"Stream closed. Processed {chunk_count} chunks (not all 5)")


async def main():
    """Run all examples."""
    await test_local_behavior()
    await test_remote_behavior()
    await show_practical_workaround()
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("- Current: Generators are materialized to lists on the server")
    print("- Limitation: No streaming, no early termination, high memory for large data")
    print("- Workaround: Use explicit chunking methods instead of generators")
    print("- Future: Could implement true streaming with server-side changes")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())