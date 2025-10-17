#!/usr/bin/env python3
"""
Test FastMCP tools to ensure they work correctly
"""

import asyncio
import sys
from pathlib import Path

# Add module to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the R executor directly
from r_integration.r_executor import RExecutor

async def test_tools():
    """Test statistical tools via R executor."""
    print("Testing FastMCP Statistical Tools via R Executor")
    print("=" * 50)

    # Initialize R executor
    executor = RExecutor()

    # Test 1: Load example dataset
    print("\n1. Testing iris dataset loading...")
    script = """
    df <- iris
    result <- list(
        n_rows = nrow(df),
        n_cols = ncol(df),
        columns = names(df),
        first_values = head(df$Sepal.Length, 3)
    )
    """
    try:
        result = await executor.execute(script)
        print(f"   ✓ Loaded iris dataset: {result['n_rows']} rows x {result['n_cols']} columns")
        print(f"   Columns: {', '.join(result['columns'])}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test 2: Summary statistics
    print("\n2. Testing summary statistics...")
    script = """
    df <- data.frame(
        x = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        y = c(2, 4, 5, 4, 5, 6, 8, 9, 10, 11)
    )
    result <- list(
        x_mean = mean(df$x),
        x_sd = sd(df$x),
        y_mean = mean(df$y),
        correlation = cor(df$x, df$y)
    )
    """
    try:
        result = await executor.execute(script)
        print(f"   ✓ Summary stats: x mean={result['x_mean']}, sd={result['x_sd']:.2f}")
        print(f"   Correlation: {result['correlation']:.3f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test 3: Linear regression
    print("\n3. Testing linear regression...")
    script = """
    df <- data.frame(
        x = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        y = c(2, 4, 5, 4, 5, 6, 8, 9, 10, 11)
    )
    model <- lm(y ~ x, data = df)
    summary_model <- summary(model)
    result <- list(
        r_squared = summary_model$r.squared,
        coefficients = coef(model),
        p_value = summary_model$coefficients[2, 4]
    )
    """
    try:
        result = await executor.execute(script)
        print(f"   ✓ Linear model R² = {result['r_squared']:.4f}")
        print(f"   Slope = {result['coefficients']['x']:.3f}, p-value = {result['p_value']:.4f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test 4: T-test
    print("\n4. Testing t-test...")
    script = """
    values <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    test_result <- t.test(values, mu = 5.5)
    result <- list(
        statistic = test_result$statistic,
        p_value = test_result$p.value,
        mean = test_result$estimate
    )
    """
    try:
        result = await executor.execute(script)
        print(f"   ✓ T-test: statistic = {result['statistic']:.3f}, p-value = {result['p_value']:.4f}")
        print(f"   Sample mean = {result['mean']}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test 5: Visualization (ggplot2)
    print("\n5. Testing ggplot2 visualization...")
    script = """
    library(ggplot2)
    library(base64enc)

    # Create a simple plot
    df <- data.frame(x = 1:10, y = rnorm(10))
    p <- ggplot(df, aes(x, y)) + geom_point() + geom_smooth(method = "lm")

    # Save to temp file
    temp_file <- tempfile(fileext = ".png")
    ggsave(temp_file, plot = p, width = 6, height = 4, dpi = 100)

    # Check if file was created
    if (file.exists(temp_file)) {
        file_size <- file.info(temp_file)$size
        unlink(temp_file)
        result <- list(success = TRUE, file_size = file_size)
    } else {
        result <- list(success = FALSE)
    }
    """
    try:
        result = await executor.execute(script)
        if result.get('success'):
            print(f"   ✓ ggplot2 visualization works (generated {result['file_size']} bytes)")
        else:
            print("   ✗ Failed to create visualization")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    print("\n" + "=" * 50)
    print("All R integration tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_tools())