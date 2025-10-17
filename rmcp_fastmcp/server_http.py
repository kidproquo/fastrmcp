#!/usr/bin/env python3
"""
FastMCP R Statistical Server - HTTP/SSE Version

Run with HTTP transport for web-based MCP clients.
"""

import sys
from pathlib import Path

# Import the main server
sys.path.insert(0, str(Path(__file__).parent))
from server_complete import mcp, logger

if __name__ == "__main__":
    import asyncio

    logger.info("Starting FastMCP R Statistical Analysis Server (HTTP/SSE)")
    logger.info("Server will be available at: http://localhost:8000/mcp/v1")

    # Run with HTTP/SSE transport
    asyncio.run(mcp.run_sse_async(
        host="0.0.0.0",
        port=8000,
        path="/mcp/v1"
    ))