"""
Simple example showing how to specify pip packages when using RemoteProxyClient.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from remotemedia.core.node import RemoteExecutorConfig
from remotemedia.remote.proxy_client import RemoteProxyClient


class RequestsExample:
    """Example class that uses the requests library."""
    
    def fetch_data(self, url: str) -> dict:
        """Fetch data from a URL using requests."""
        import requests
        
        response = requests.get(url)
        return {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "content_length": len(response.content),
            "text_preview": response.text[:200] if response.text else None
        }


async def main():
    # Configure with pip packages
    config = RemoteExecutorConfig(
        host="localhost", 
        port=50052, 
        ssl_enabled=False,
        pip_packages=["requests"]  # Specify packages to install
    )
    
    async with RemoteProxyClient(config) as client:
        # Create remote proxy
        example = RequestsExample()
        remote_example = await client.create_proxy(example)
        
        # Use the remote object - requests will be installed on the server
        result = await remote_example.fetch_data("https://api.github.com")
        
        print("Remote execution result:")
        print(f"Status code: {result['status_code']}")
        print(f"Content length: {result['content_length']}")
        print(f"Preview: {result['text_preview'][:50]}...")


if __name__ == "__main__":
    asyncio.run(main())