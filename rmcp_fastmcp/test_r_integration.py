#!/usr/bin/env python3
"""
Test script for R integration to verify it works correctly
"""

import asyncio
import json
import sys
from pathlib import Path

# Add module to path
sys.path.insert(0, str(Path(__file__).parent))

from r_integration.r_executor import RExecutor

async def test_r_executor():
    """Test the R executor with simple operations."""
    print("Testing R Integration for FastMCP Server")
    print("=" * 50)

    # Initialize executor
    print("\n1. Initializing R executor...")
    try:
        executor = RExecutor()
        print("   ✓ R executor initialized successfully")
    except Exception as e:
        print(f"   ✗ Failed to initialize R executor: {e}")
        return False

    # Test 1: Simple calculation
    print("\n2. Testing simple calculation...")
    script = """
    result <- list(
        sum = 1 + 2 + 3,
        mean = mean(c(1, 2, 3, 4, 5))
    )
    """
    try:
        result = await executor.execute(script)
        print(f"   Result: {result}")
        if result.get("sum") == 6 and result.get("mean") == 3:
            print("   ✓ Simple calculation works")
        else:
            print("   ✗ Unexpected result")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test 2: Data exchange
    print("\n3. Testing data exchange...")
    script = """
    df <- as.data.frame(data)
    result <- list(
        n_rows = nrow(df),
        n_cols = ncol(df),
        column_names = names(df),
        x_mean = mean(df$x),
        y_sum = sum(df$y)
    )
    """
    test_data = {
        "x": [1, 2, 3, 4, 5],
        "y": [2, 4, 6, 8, 10]
    }
    try:
        result = await executor.execute(script, {"data": test_data})
        print(f"   Result: {result}")
        if result.get("n_rows") == 5 and result.get("x_mean") == 3:
            print("   ✓ Data exchange works")
        else:
            print("   ✗ Unexpected result")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test 3: Check if jsonlite is available
    print("\n4. Testing jsonlite package...")
    script = """
    if ("jsonlite" %in% installed.packages()[,"Package"]) {
        result <- list(status = "installed")
    } else {
        result <- list(status = "not installed")
    }
    """
    try:
        result = await executor.execute(script)
        print(f"   jsonlite status: {result.get('status')}")
        if result.get("status") == "installed":
            print("   ✓ jsonlite is available")
        else:
            print("   ✗ jsonlite is not installed")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test 4: Check other required packages
    print("\n5. Checking required R packages...")
    packages_to_check = ["ggplot2", "base64enc", "dplyr"]
    for package in packages_to_check:
        script = f"""
        if ("{package}" %in% installed.packages()[,"Package"]) {{
            result <- list(status = "installed")
        }} else {{
            result <- list(status = "not installed")
        }}
        """
        try:
            result = await executor.execute(script)
            status = result.get("status", "error")
            symbol = "✓" if status == "installed" else "✗"
            print(f"   {symbol} {package}: {status}")
        except Exception as e:
            print(f"   ✗ {package}: Error checking - {e}")

    print("\n" + "=" * 50)
    print("Test complete!")

    return True

if __name__ == "__main__":
    success = asyncio.run(test_r_executor())
    sys.exit(0 if success else 1)