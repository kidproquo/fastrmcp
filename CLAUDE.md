# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastMCP R Statistical Analysis Server - A comprehensive MCP (Model Context Protocol) server that provides 44 statistical analysis tools through R, implemented using the FastMCP framework. This is a streamlined reimplementation of the RMCP server with significantly less boilerplate (~1000 lines vs ~5000 lines).

## Key Commands

### Development Commands

```bash
# Run server in STDIO mode (default, for MCP clients)
cd rmcp_fastmcp
python3 server_complete.py

# Run server with the launcher (supports multiple transports)
python3 run_server.py              # STDIO mode
python3 run_server.py http         # HTTP mode on port 8000
python3 run_server.py http 3003    # HTTP mode on custom port

# Test the R integration
python3 test_r_integration.py

# Test specific tools
python3 test_tools.py
```

### R Package Management

```bash
# Verify R installation
R --version

# Install all required R packages (from README)
R -q -e "install.packages(c( \
  'jsonlite','plm','lmtest','sandwich','AER','dplyr', \
  'forecast','vars','urca','tseries','nortest','car', \
  'rpart','randomForest','ggplot2','gridExtra','tidyr', \
  'rlang','readxl','openxlsx','base64enc','reshape2','knitr','broom', \
  'MASS','boot','survival','nlme','mgcv','lme4','glmnet', \
  'e1071','caret','nnet','gbm','xgboost','kernlab','cluster', \
  'zoo','xts','TTR','quantmod','data.table','lattice', \
  'corrplot','viridis','RColorBrewer','lavaan' \
))"
```

### Docker Commands

```bash
# Build the Docker image
docker build -t fastrmcp .

# Run the container (HTTP mode on port 3003)
docker run -p 3003:3003 fastrmcp
```

## Architecture

### Core Components

1. **server_complete.py** (Main Server, ~1140 lines)
   - Entry point defining all 44 statistical tools using `@mcp.tool()` decorators
   - Tools organized into 8 categories: Regression & Econometrics, Time Series, Statistical Testing, Data Analysis & Transformation, Machine Learning, Visualization, File Operations, and Advanced Features
   - Each tool is an async function that constructs an R script and calls `r_executor.execute()`
   - Uses Pydantic models for input validation
   - Provides a catalog resource via `@mcp.resource("stats://catalog")`

2. **r_integration/r_executor.py** (R Executor, ~216 lines)
   - `RExecutor` class handles all Python-R communication
   - Uses subprocess to execute R scripts via `R --slave --no-save --no-restore`
   - Data exchange via JSON: Python dict → temp JSON file → R list → JSON output → Python dict
   - Automatic R package installation if missing packages detected
   - 30-second timeout for R script execution
   - Error handling for R errors, timeouts, and JSON parsing failures

3. **run_server.py** (Multi-Transport Launcher)
   - CLI wrapper for running server in STDIO or HTTP/SSE mode
   - Parses command-line arguments to select transport and port
   - Calls `mcp.run_stdio_async()` or `mcp.run_http_async()`

### Data Flow

```
MCP Client → FastMCP Framework → Tool Function (Python)
                                       ↓
                          Construct R script + data dict
                                       ↓
                          r_executor.execute(script, data)
                                       ↓
            subprocess: R --slave < full_script (with JSON I/O)
                                       ↓
                          Parse JSON output → return to client
```

### R Script Execution Pattern

All tools follow this pattern:
1. Define R script as a multi-line string (uses R packages like ggplot2, plm, forecast, etc.)
2. Script must set a `result` variable (list containing outputs)
3. Call `await r_executor.execute(script, data_dict)` where data_dict contains all inputs
4. The executor wraps the script with JSON I/O boilerplate and executes it
5. Variables from data_dict are automatically assigned in R's global environment

### Tool Categories (44 Total)

1. **Regression & Econometrics (8)**: linear_model, logistic_regression, panel_regression, instrumental_variables, var_model, correlation_analysis, regression_plot
2. **Time Series Analysis (6)**: arima_model, decompose_timeseries, stationarity_test, lag_lead, difference, time_series_plot
3. **Statistical Testing (5)**: t_test, anova, chi_square_test, normality_test
4. **Data Analysis & Transformation (9)**: summary_stats, outlier_detection, standardize, winsorize, frequency_table, filter_data, data_info, validate_data
5. **Machine Learning (4)**: kmeans_clustering, decision_tree, random_forest
6. **Visualization (6)**: scatter_plot, histogram, boxplot, correlation_heatmap, time_series_plot, regression_plot (all support base64 PNG output)
7. **File Operations (3)**: read_csv, write_csv, read_json
8. **Advanced Features (4)**: build_formula, suggest_fix, execute_r_analysis, load_example

## Important Implementation Details

### Adding New Tools

1. Define a Pydantic input model if needed (optional for simple inputs)
2. Create an async function decorated with `@mcp.tool()`
3. Write the R script as a multi-line string:
   - Load required R libraries
   - Perform analysis using variables from the data dict
   - Set `result` variable as a list with outputs
4. Call `await r_executor.execute(script, data_dict)` and return the result

Example:
```python
@mcp.tool()
async def my_analysis(data: Dict[str, List[Any]], param: str) -> Dict[str, Any]:
    """Description of the tool."""
    script = """
    library(jsonlite)
    df <- as.data.frame(data)
    # ... R analysis code ...
    result <- list(output = computed_value)
    """
    return await r_executor.execute(script, {"data": data, "param": param})
```

### Visualization Tools

- All visualization tools have a `return_image: bool` parameter
- When `True`, R generates a PNG to a temp file, encodes it as base64, and returns `{"image": base64_string, "format": "png"}`
- Uses `library(base64enc)` and `base64encode()` function
- PNG files are created with `png()` or `ggsave()`, then read with `readBin()` and encoded

### R Package Requirements

The server requires 46 R packages (listed in Dockerfile and README). The RExecutor automatically checks for missing packages on startup and attempts to install them from CRAN. Manual installation is recommended before first run to avoid startup delays.

### Safety Considerations

- `execute_r_analysis` tool blocks dangerous operations (`system`, `file.remove`, `unlink`)
- R scripts run with `--slave --no-save --no-restore` flags (non-interactive, no workspace save)
- 30-second timeout prevents runaway scripts
- All R scripts should set `result` variable; missing `result` returns an error

### Error Handling

- R errors are captured from stderr and returned in `{"error": "message"}` format
- JSON parsing errors include `{"error": "msg", "raw_output": "..."}` for debugging
- Timeouts return `{"error": "R script execution timed out"}`
- The `suggest_fix` tool provides error diagnosis for common R errors

## MCP Client Configuration

To use this server with an MCP client (e.g., Claude Desktop), add to the client's configuration:

```json
{
  "mcpServers": {
    "rmcp-fastmcp": {
      "command": "python3",
      "args": ["/path/to/rmcp_fastmcp/server_complete.py"]
    }
  }
}
```

For HTTP mode, the server runs on a specified port with the MCP endpoint at `/mcp`.

## Testing Strategy

- `test_r_integration.py`: Tests the RExecutor class directly
- `test_tools.py`: Tests individual MCP tools
- `test_curl.sh` and `test_simple_curl.sh`: Manual HTTP endpoint testing scripts

## Dependencies

- **Python**: fastmcp (>=2.12.4), mcp-python, pydantic, httpx, uvicorn, starlette
- **R**: 46 packages including jsonlite, ggplot2, base64enc, plm, forecast, vars, tseries, etc.
- See `requirements.txt` for complete Python dependencies
- See `Dockerfile` lines 17-26 or README for complete R package list
