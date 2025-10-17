#!/usr/bin/env python3
"""
FastMCP R Statistical Server - Multi-Transport Launcher

Usage:
    python3 run_server.py           # Default: STDIO
    python3 run_server.py stdio      # Explicit STDIO
    python3 run_server.py http       # HTTP/SSE on port 8000
    python3 run_server.py http 3000  # HTTP/SSE on custom port
"""

import sys
import asyncio
from pathlib import Path

# Import the main server
sys.path.insert(0, str(Path(__file__).parent))
from server_complete import mcp, logger

def main():
    # Parse command line arguments
    mode = "stdio"  # default
    port = 8000     # default port for HTTP

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    if len(sys.argv) > 2 and mode == "http":
        try:
            port = int(sys.argv[2])
        except ValueError:
            print(f"Invalid port: {sys.argv[2]}")
            sys.exit(1)

    # Run the appropriate transport
    if mode == "stdio":
        logger.info("Starting FastMCP R Statistical Server (STDIO mode)")
        logger.info("Ready for MCP communication via standard input/output")
        asyncio.run(mcp.run_stdio_async())

    elif mode == "http":
        logger.info(f"Starting FastMCP R Statistical Server (HTTP mode)")
        logger.info(f"Server will be available at: http://localhost:{port}/mcp")
        asyncio.run(mcp.run_http_async(
            host="0.0.0.0",
            port=port,
            path="/mcp"
        ))

    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python3 run_server.py [stdio|http] [port]")
        sys.exit(1)

if __name__ == "__main__":
    main()