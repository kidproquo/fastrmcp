#!/bin/bash

# Simple test for FastMCP R Statistical Tools via curl

SERVER="http://localhost:8085/mcp"

echo "Testing FastMCP Server with curl - Simple Tests"
echo "================================================"
echo ""

# 1. Initialize with proper clientInfo
echo "1. Initializing connection..."
INIT_RESPONSE=$(curl -s -X POST "$SERVER" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "method": "initialize",
    "params": {
      "protocolVersion": "2025-06-18",
      "clientInfo": {
        "name": "curl-test",
        "version": "1.0.0"
      },
      "capabilities": {}
    },
    "id": 1
  }')

echo "$INIT_RESPONSE" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if 'result' in data:
        result = data['result']
        print(f\"  ✓ Connected successfully\")
        print(f\"  Server: {result.get('serverInfo', {}).get('name', 'Unknown')}\")
        print(f\"  Protocol: {result.get('protocolVersion', 'Unknown')}\")
    else:
        print(f\"  ✗ Error: {data.get('error', {}).get('message', 'Unknown error')}\")
except Exception as e:
    print(f\"  ✗ Failed to parse response: {e}\")
"

echo ""

# Extract session ID from response headers (if available)
SESSION_ID="test-session-$(date +%s)"

# 2. List tools - simpler test
echo "2. Listing available tools..."
TOOLS_RESPONSE=$(curl -s -X POST "$SERVER" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "X-Session-Id: $SESSION_ID" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/list",
    "params": {},
    "id": 2
  }')

echo "$TOOLS_RESPONSE" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if 'result' in data and 'tools' in data['result']:
        tools = data['result']['tools']
        print(f\"  ✓ Found {len(tools)} tools\")
        for tool in tools[:3]:
            print(f\"    - {tool['name']}\")
        if len(tools) > 3:
            print(f\"    ... and {len(tools) - 3} more\")
    else:
        print(f\"  ✗ Error: {data.get('error', {}).get('message', 'Unknown error')}\")
except Exception as e:
    print(f\"  ✗ Failed to parse response: {e}\")
    print(f\"  Raw response: {sys.stdin.read()}\")
"

echo ""

# 3. Call a simple tool - load_example
echo "3. Testing load_example tool..."
EXAMPLE_RESPONSE=$(curl -s -X POST "$SERVER" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "X-Session-Id: $SESSION_ID" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "load_example",
      "arguments": {
        "dataset_name": "mtcars"
      }
    },
    "id": 3
  }')

echo "$EXAMPLE_RESPONSE" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if 'result' in data:
        print(f\"  ✓ Tool executed successfully\")
        # Try to extract content
        if 'content' in data['result']:
            content = data['result']['content']
            if isinstance(content, list) and len(content) > 0:
                text = str(content[0].get('text', ''))[:200]
                print(f\"  Result preview: {text}...\")
    else:
        print(f\"  ✗ Error: {data.get('error', {}).get('message', 'Unknown error')}\")
except Exception as e:
    print(f\"  ✗ Failed to parse response: {e}\")
"

echo ""

# 4. Call summary_stats tool
echo "4. Testing summary_stats tool..."
STATS_RESPONSE=$(curl -s -X POST "$SERVER" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "X-Session-Id: $SESSION_ID" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "summary_stats",
      "arguments": {
        "data": {
          "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }
      }
    },
    "id": 4
  }')

echo "$STATS_RESPONSE" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if 'result' in data:
        print(f\"  ✓ Statistics calculated\")
        if 'content' in data['result']:
            content = data['result']['content']
            if isinstance(content, list) and len(content) > 0:
                text = str(content[0].get('text', ''))[:200]
                print(f\"  Result preview: {text}...\")
    else:
        error = data.get('error', {})
        print(f\"  ✗ Error: {error.get('message', 'Unknown error')}\")
        if 'data' in error:
            print(f\"    Details: {error['data']}\")
except Exception as e:
    print(f\"  ✗ Failed to parse response: {e}\")
"

echo ""
echo "Test complete!"