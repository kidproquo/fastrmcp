#!/bin/bash

# Test FastMCP R Statistical Tools via curl

SERVER="http://localhost:8085/mcp"
SESSION_ID="test-session-$(date +%s)"

echo "Testing FastMCP Server with curl"
echo "================================"
echo "Server: $SERVER"
echo "Session: $SESSION_ID"
echo ""

# 1. Initialize connection
echo "1. Initializing MCP connection..."
curl -X POST "$SERVER" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "X-Session-Id: $SESSION_ID" \
  -d '{
    "jsonrpc": "2.0",
    "method": "initialize",
    "params": {
      "protocolVersion": "2025-06-18",
      "capabilities": {
        "tools": {}
      }
    },
    "id": 1
  }' 2>/dev/null | python3 -m json.tool | head -20

echo -e "\n---\n"

# 2. List available tools
echo "2. Listing available tools (first 5)..."
curl -X POST "$SERVER" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "X-Session-Id: $SESSION_ID" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/list",
    "params": {},
    "id": 2
  }' 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
if 'result' in data and 'tools' in data['result']:
    tools = data['result']['tools'][:5]
    for tool in tools:
        print(f\"  - {tool['name']}: {tool.get('description', 'No description')[:60]}...\")
    print(f\"  ... and {len(data['result']['tools']) - 5} more tools\")
"

echo -e "\n---\n"

# 3. Test load_example tool
echo "3. Testing load_example tool (iris dataset)..."
curl -X POST "$SERVER" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "X-Session-Id: $SESSION_ID" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "load_example",
      "arguments": {
        "dataset_name": "iris"
      }
    },
    "id": 3
  }' 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
if 'result' in data:
    result = data['result']
    if 'content' in result:
        content = result['content']
        if isinstance(content, list) and len(content) > 0:
            text = content[0].get('text', '')
            # Parse the actual result
            import ast
            try:
                actual_data = ast.literal_eval(text)
                print(f\"  Dataset: {actual_data.get('description', 'No description')}\")
                print(f\"  Shape: {actual_data.get('shape', 'Unknown')}\")
                print(f\"  Columns: {', '.join(actual_data.get('columns', []))}\")
            except:
                print(text[:200])
else:
    print(json.dumps(data, indent=2)[:500])
"

echo -e "\n---\n"

# 4. Test summary_stats tool
echo "4. Testing summary_stats tool..."
curl -X POST "$SERVER" \
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
          "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
          "y": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        }
      }
    },
    "id": 4
  }' 2>/dev/null | python3 -c "
import json, sys, ast
data = json.load(sys.stdin)
if 'result' in data and 'content' in data['result']:
    text = data['result']['content'][0].get('text', '')
    try:
        stats = ast.literal_eval(text)
        if 'statistics' in stats:
            for var, values in stats['statistics'].items():
                print(f\"  Variable: {var}\")
                print(f\"    Mean: {values.get('mean', 'N/A')}\")
                print(f\"    SD: {values.get('sd', 'N/A')}\")
                print(f\"    Min: {values.get('min', 'N/A')}, Max: {values.get('max', 'N/A')}\")
    except:
        print(text[:300])
"

echo -e "\n---\n"

# 5. Test linear_model tool
echo "5. Testing linear_model tool (y ~ x)..."
curl -X POST "$SERVER" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "X-Session-Id: $SESSION_ID" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "linear_model",
      "arguments": {
        "data": {
          "x": [1, 2, 3, 4, 5],
          "y": [2.1, 4.2, 5.9, 8.1, 10.2]
        },
        "formula": "y ~ x"
      }
    },
    "id": 5
  }' 2>/dev/null | python3 -c "
import json, sys, ast
data = json.load(sys.stdin)
if 'result' in data and 'content' in data['result']:
    text = data['result']['content'][0].get('text', '')
    try:
        model = ast.literal_eval(text)
        print(f\"  R-squared: {model.get('r_squared', 'N/A')}\")
        print(f\"  Adj R-squared: {model.get('adj_r_squared', 'N/A')}\")
        if 'coefficients' in model:
            print(f\"  Coefficients: {model['coefficients']}\")
    except:
        print(text[:300])
"

echo -e "\n---\n"

# 6. Test t_test tool
echo "6. Testing t_test tool (one-sample)..."
curl -X POST "$SERVER" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "X-Session-Id: $SESSION_ID" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "t_test",
      "arguments": {
        "data": {
          "values": [4.5, 5.1, 4.8, 5.0, 4.9, 5.2, 4.7, 5.1, 4.9, 5.0]
        },
        "variable": "values",
        "mu": 5.0,
        "alternative": "two.sided"
      }
    },
    "id": 6
  }' 2>/dev/null | python3 -c "
import json, sys, ast
data = json.load(sys.stdin)
if 'result' in data and 'content' in data['result']:
    text = data['result']['content'][0].get('text', '')
    try:
        test = ast.literal_eval(text)
        print(f\"  Test statistic: {test.get('statistic', 'N/A')}\")
        print(f\"  P-value: {test.get('p_value', 'N/A')}\")
        print(f\"  Confidence interval: {test.get('confidence_interval', 'N/A')}\")
    except:
        print(text[:300])
"

echo -e "\n---\n"

# 7. Test correlation_analysis tool
echo "7. Testing correlation_analysis tool..."
curl -X POST "$SERVER" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "X-Session-Id: $SESSION_ID" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "correlation_analysis",
      "arguments": {
        "data": {
          "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
          "y": [2, 4, 5, 4, 5, 6, 8, 9, 10, 11],
          "z": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        },
        "method": "pearson"
      }
    },
    "id": 7
  }' 2>/dev/null | python3 -c "
import json, sys, ast
data = json.load(sys.stdin)
if 'result' in data and 'content' in data['result']:
    text = data['result']['content'][0].get('text', '')
    try:
        corr = ast.literal_eval(text)
        if 'correlation' in corr:
            print('  Correlation matrix:')
            matrix = corr['correlation']
            for var in ['x', 'y', 'z']:
                if var in matrix:
                    print(f\"    {var}: {matrix[var]}\")
    except:
        print(text[:300])
"

echo -e "\n---\n"
echo "Test complete!"
echo ""
echo "Server log (last 5 lines):"
tail -5 server_8085.log