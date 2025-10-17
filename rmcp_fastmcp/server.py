#!/usr/bin/env python3
"""
FastMCP-based R Statistical Analysis Server

A streamlined MCP server using FastMCP framework for statistical analysis through R.
Inspired by the original RMCP implementation but with less boilerplate.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from fastmcp import FastMCP
from pydantic import BaseModel, Field

# Add R integration module to path
sys.path.insert(0, str(Path(__file__).parent))

from r_integration.r_executor import RExecutor

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP(
    "R Statistical Analysis Server (FastMCP)",
    version="0.1.0"
)

# Initialize R executor
r_executor = RExecutor()

# ============================================================================
# Data Models
# ============================================================================

class DataInput(BaseModel):
    """Input data for statistical analysis"""
    data: Dict[str, List[Any]] = Field(
        description="Data as column-oriented dictionary"
    )

class FormulaInput(BaseModel):
    """Input for formula-based analyses"""
    data: Dict[str, List[Any]] = Field(
        description="Data as column-oriented dictionary"
    )
    formula: str = Field(
        description="R formula (e.g., 'y ~ x1 + x2')"
    )

class TTestInput(BaseModel):
    """Input for t-test analysis"""
    data: Dict[str, List[Any]] = Field(
        description="Data as column-oriented dictionary"
    )
    variable: str = Field(
        description="Variable to test"
    )
    group: Optional[str] = Field(
        None,
        description="Group variable for two-sample test"
    )
    mu: float = Field(
        0,
        description="Hypothesized mean for one-sample test"
    )
    alternative: str = Field(
        "two.sided",
        description="Alternative hypothesis: two.sided, less, or greater"
    )
    paired: bool = Field(
        False,
        description="Whether to perform paired t-test"
    )

class CorrelationInput(BaseModel):
    """Input for correlation analysis"""
    data: Dict[str, List[Any]] = Field(
        description="Data as column-oriented dictionary"
    )
    variables: Optional[List[str]] = Field(
        None,
        description="Variables to include (all numeric if None)"
    )
    method: str = Field(
        "pearson",
        description="Correlation method: pearson, spearman, or kendall"
    )

class PlotInput(BaseModel):
    """Input for plotting functions"""
    data: Dict[str, List[Any]] = Field(
        description="Data as column-oriented dictionary"
    )
    x: str = Field(
        description="X-axis variable"
    )
    y: Optional[str] = Field(
        None,
        description="Y-axis variable (for scatter plots)"
    )
    title: Optional[str] = Field(
        None,
        description="Plot title"
    )
    return_image: bool = Field(
        True,
        description="Return base64-encoded image"
    )

# ============================================================================
# Statistical Tools
# ============================================================================

@mcp.tool()
async def summary_stats(input: DataInput) -> Dict[str, Any]:
    """
    Calculate comprehensive descriptive statistics for the dataset.

    Returns mean, median, std dev, min, max, and quartiles for each numeric variable.
    """
    script = """
    df <- as.data.frame(data)

    # Get numeric columns
    numeric_cols <- sapply(df, is.numeric)
    numeric_df <- df[, numeric_cols, drop = FALSE]

    if (ncol(numeric_df) == 0) {
        result <- list(error = "No numeric columns found")
    } else {
        # Calculate summary statistics
        summary_list <- list()
        for (col in names(numeric_df)) {
            values <- numeric_df[[col]]
            values <- values[!is.na(values)]

            if (length(values) > 0) {
                summary_list[[col]] <- list(
                    n = length(values),
                    mean = mean(values),
                    median = median(values),
                    std = sd(values),
                    min = min(values),
                    q1 = quantile(values, 0.25),
                    q3 = quantile(values, 0.75),
                    max = max(values),
                    missing = sum(is.na(numeric_df[[col]]))
                )
            }
        }
        result <- list(statistics = summary_list)
    }
    """

    return await r_executor.execute(script, {"data": input.data})

@mcp.tool()
async def linear_regression(input: FormulaInput) -> Dict[str, Any]:
    """
    Perform linear regression analysis.

    Returns coefficients, R-squared, p-values, and diagnostic statistics.
    """
    script = """
    df <- as.data.frame(data)

    # Fit linear model
    model <- lm(as.formula(formula), data = df)

    # Get summary
    summary_model <- summary(model)

    # Extract coefficients
    coef_df <- as.data.frame(summary_model$coefficients)
    names(coef_df) <- c("estimate", "std_error", "t_value", "p_value")

    result <- list(
        coefficients = as.list(coef_df),
        r_squared = summary_model$r.squared,
        adj_r_squared = summary_model$adj.r.squared,
        f_statistic = list(
            value = summary_model$fstatistic[1],
            df1 = summary_model$fstatistic[2],
            df2 = summary_model$fstatistic[3]
        ),
        residuals = list(
            min = min(model$residuals),
            q1 = quantile(model$residuals, 0.25),
            median = median(model$residuals),
            q3 = quantile(model$residuals, 0.75),
            max = max(model$residuals)
        ),
        n_observations = nrow(df)
    )
    """

    return await r_executor.execute(script, {
        "data": input.data,
        "formula": input.formula
    })

@mcp.tool()
async def t_test(input: TTestInput) -> Dict[str, Any]:
    """
    Perform t-test analysis (one-sample, two-sample, or paired).

    Returns t-statistic, p-value, confidence interval, and effect size.
    """
    script = """
    df <- as.data.frame(data)

    # Extract variable
    var_data <- df[[variable]]

    if (!is.null(group) && group != "") {
        # Two-sample t-test
        group_data <- df[[group]]
        unique_groups <- unique(group_data)

        if (length(unique_groups) != 2) {
            result <- list(error = "Group variable must have exactly 2 unique values")
        } else {
            group1 <- var_data[group_data == unique_groups[1]]
            group2 <- var_data[group_data == unique_groups[2]]

            test_result <- t.test(
                group1, group2,
                alternative = alternative,
                paired = paired
            )
        }
    } else {
        # One-sample t-test
        test_result <- t.test(
            var_data,
            mu = mu,
            alternative = alternative
        )
    }

    if (!exists("result")) {
        # Calculate effect size (Cohen's d)
        if (!is.null(group) && group != "") {
            mean_diff <- mean(group1, na.rm = TRUE) - mean(group2, na.rm = TRUE)
            pooled_sd <- sqrt(((length(group1) - 1) * sd(group1, na.rm = TRUE)^2 +
                              (length(group2) - 1) * sd(group2, na.rm = TRUE)^2) /
                             (length(group1) + length(group2) - 2))
            cohens_d <- mean_diff / pooled_sd
        } else {
            cohens_d <- (mean(var_data, na.rm = TRUE) - mu) / sd(var_data, na.rm = TRUE)
        }

        result <- list(
            statistic = test_result$statistic,
            p_value = test_result$p.value,
            confidence_interval = as.numeric(test_result$conf.int),
            estimate = as.numeric(test_result$estimate),
            alternative = test_result$alternative,
            method = test_result$method,
            effect_size = cohens_d
        )
    }
    """

    params = {
        "data": input.data,
        "variable": input.variable,
        "group": input.group,
        "mu": input.mu,
        "alternative": input.alternative,
        "paired": input.paired
    }

    return await r_executor.execute(script, params)

@mcp.tool()
async def correlation_analysis(input: CorrelationInput) -> Dict[str, Any]:
    """
    Calculate correlation matrix between variables.

    Returns correlation coefficients and p-values for all pairs.
    """
    script = """
    df <- as.data.frame(data)

    # Select variables
    if (!is.null(variables) && length(variables) > 0) {
        df_subset <- df[, variables, drop = FALSE]
    } else {
        # Use all numeric columns
        numeric_cols <- sapply(df, is.numeric)
        df_subset <- df[, numeric_cols, drop = FALSE]
    }

    if (ncol(df_subset) < 2) {
        result <- list(error = "Need at least 2 numeric variables for correlation")
    } else {
        # Calculate correlation matrix
        cor_matrix <- cor(df_subset, method = method, use = "pairwise.complete.obs")

        # Calculate p-values
        n <- nrow(df_subset)
        p_matrix <- matrix(NA, ncol(df_subset), ncol(df_subset))

        for (i in 1:(ncol(df_subset)-1)) {
            for (j in (i+1):ncol(df_subset)) {
                test_result <- cor.test(df_subset[,i], df_subset[,j], method = method)
                p_matrix[i,j] <- test_result$p.value
                p_matrix[j,i] <- test_result$p.value
            }
        }
        diag(p_matrix) <- 0

        result <- list(
            correlation_matrix = as.list(as.data.frame(cor_matrix)),
            p_values = as.list(as.data.frame(p_matrix)),
            variables = colnames(df_subset),
            method = method,
            n_observations = n
        )
    }
    """

    return await r_executor.execute(script, {
        "data": input.data,
        "variables": input.variables,
        "method": input.method
    })

@mcp.tool()
async def scatter_plot(input: PlotInput) -> Dict[str, Any]:
    """
    Create a scatter plot with optional trend line.

    Returns base64-encoded PNG image or plot data.
    """
    script = """
    library(ggplot2)

    df <- as.data.frame(data)

    # Create scatter plot
    p <- ggplot(df, aes_string(x = x, y = y)) +
        geom_point(alpha = 0.6, size = 2) +
        geom_smooth(method = "lm", se = TRUE, alpha = 0.2) +
        theme_minimal() +
        labs(x = x, y = y)

    if (!is.null(title) && title != "") {
        p <- p + ggtitle(title)
    }

    if (return_image) {
        # Save to temporary file
        temp_file <- tempfile(fileext = ".png")
        ggsave(temp_file, plot = p, width = 8, height = 6, dpi = 100)

        # Read and encode as base64
        img_data <- readBin(temp_file, "raw", file.info(temp_file)$size)
        img_base64 <- base64enc::base64encode(img_data)
        unlink(temp_file)

        result <- list(
            image = img_base64,
            format = "png",
            width = 800,
            height = 600
        )
    } else {
        result <- list(
            x_values = df[[x]],
            y_values = df[[y]],
            correlation = cor(df[[x]], df[[y]], use = "complete.obs")
        )
    }
    """

    return await r_executor.execute(script, {
        "data": input.data,
        "x": input.x,
        "y": input.y,
        "title": input.title,
        "return_image": input.return_image
    })

@mcp.tool()
async def load_example_dataset(name: str = Field(description="Dataset name: iris, mtcars, or economics")) -> Dict[str, Any]:
    """
    Load a built-in example dataset for testing.

    Available datasets: iris, mtcars, economics
    """
    script = """
    # Load requested dataset
    if (name == "iris") {
        df <- iris
        description <- "Fisher's Iris dataset with flower measurements"
    } else if (name == "mtcars") {
        df <- mtcars
        description <- "Motor Trend car performance dataset"
    } else if (name == "economics") {
        # Create synthetic economics dataset
        set.seed(42)
        n <- 100
        df <- data.frame(
            gdp = rnorm(n, mean = 50000, sd = 10000),
            unemployment = runif(n, min = 3, max = 10),
            inflation = rnorm(n, mean = 2, sd = 1),
            interest_rate = runif(n, min = 0, max = 5)
        )
        description <- "Synthetic economic indicators dataset"
    } else {
        result <- list(error = paste("Unknown dataset:", name))
    }

    if (!exists("result")) {
        result <- list(
            data = as.list(df),
            description = description,
            shape = c(nrow(df), ncol(df)),
            columns = names(df)
        )
    }
    """

    return await r_executor.execute(script, {"name": name})

# ============================================================================
# Resources
# ============================================================================

@mcp.resource("stats://help")
async def help_resource() -> str:
    """Get help documentation for the R Statistical Analysis Server."""
    return """# R Statistical Analysis Server (FastMCP)

## Available Tools

### Data Analysis
- **summary_stats**: Calculate descriptive statistics
- **correlation_analysis**: Compute correlation matrices

### Statistical Tests
- **t_test**: Perform t-tests (one-sample, two-sample, paired)
- **linear_regression**: Fit linear regression models

### Visualization
- **scatter_plot**: Create scatter plots with trend lines

### Data Loading
- **load_example_dataset**: Load example datasets (iris, mtcars, economics)

## Quick Start

1. Load an example dataset:
   ```
   load_example_dataset(name="iris")
   ```

2. Calculate summary statistics:
   ```
   summary_stats(data={...})
   ```

3. Run a linear regression:
   ```
   linear_regression(data={...}, formula="y ~ x")
   ```

## Formula Syntax

R formulas follow the pattern: `response ~ predictors`

Examples:
- Simple: `y ~ x`
- Multiple: `y ~ x1 + x2`
- Interaction: `y ~ x1 * x2`
- All variables: `y ~ .`
"""

@mcp.resource("stats://version")
async def version_resource() -> str:
    """Get version information."""
    import subprocess

    # Get R version
    r_version = subprocess.run(
        ["R", "--version"],
        capture_output=True,
        text=True
    ).stdout.split('\n')[0]

    return f"""FastMCP R Statistical Server
Version: 0.1.0
R Version: {r_version}
FastMCP Version: {mcp.version}
"""

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import asyncio

    logger.info("Starting FastMCP R Statistical Analysis Server")

    # Run the server
    asyncio.run(mcp.run())