# FastMCP R Statistical Analysis Server

A streamlined implementation of the RMCP statistical analysis server using the FastMCP framework.

## Overview

This FastMCP-based server provides 44 comprehensive statistical analysis tools through R, matching the functionality of the original RMCP implementation but with significantly less boilerplate code.

## Features

- **44 Statistical Tools** across 8 categories
- **Simplified Implementation** using FastMCP decorators
- **Full R Integration** with automatic package management
- **Base64 Image Support** for visualizations
- **MCP Protocol Compliant** with proper tool and resource definitions

## Installation

1. Ensure R is installed:
```bash
R --version
```
Ensure all the R packages are installed:

```bash
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

2. Install Python dependencies:
```bash
pip install fastmcp mcp-python
```

3. R packages will be automatically installed on first run.

## Usage

### Running the Server

```bash
cd rmcp_fastmcp
python3 server_complete.py
```

### Using with MCP Clients

The server runs in stdio mode by default, making it compatible with any MCP client:

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

## Tool Categories

### 1. Regression & Econometrics (8 tools)
- `linear_model` - OLS linear regression
- `logistic_regression` - GLM/logistic regression
- `panel_regression` - Fixed/random effects panel data
- `instrumental_variables` - 2SLS regression
- `var_model` - Vector autoregression
- `correlation_analysis` - Pairwise correlations
- `regression_plot` - Diagnostic plots

### 2. Time Series Analysis (6 tools)
- `arima_model` - ARIMA modeling and forecasting
- `decompose_timeseries` - Trend/seasonal decomposition
- `stationarity_test` - ADF/KPSS/PP tests
- `lag_lead` - Create lagged/lead variables
- `difference` - Differencing transformations
- `time_series_plot` - Time series visualization

### 3. Statistical Testing (5 tools)
- `t_test` - One/two-sample and paired t-tests
- `anova` - Analysis of variance
- `chi_square_test` - Chi-square tests
- `normality_test` - Shapiro-Wilk/Anderson-Darling tests

### 4. Data Analysis & Transformation (9 tools)
- `summary_stats` - Descriptive statistics
- `outlier_detection` - IQR/Z-score methods
- `standardize` - Z-score/min-max scaling
- `winsorize` - Outlier treatment
- `frequency_table` - Categorical frequencies
- `filter_data` - Conditional filtering
- `data_info` - Dataset information
- `validate_data` - Data quality checks

### 5. Machine Learning (4 tools)
- `kmeans_clustering` - K-means clustering
- `decision_tree` - Classification/regression trees
- `random_forest` - Random forest models

### 6. Visualization (6 tools)
- `scatter_plot` - Scatter plots with trends
- `histogram` - Histograms with density
- `boxplot` - Box plots
- `correlation_heatmap` - Correlation matrices

### 7. File Operations (3 tools)
- `read_csv` - Import CSV files
- `write_csv` - Export to CSV
- `read_json` - Import JSON files

### 8. Advanced Features (4 tools)
- `build_formula` - Natural language to R formula
- `suggest_fix` - Error diagnosis
- `execute_r_analysis` - Custom R code execution
- `load_example` - Load example datasets

## Example Usage

### Simple prompt
`I want to analyze if there's a correlation between study hours [2, 3, 1, 4, 5, 3, 6, 2, 4, 5] and test scores [65, 70, 60, 75, 85, 72, 90, 68, 78, 82]. Calculate the correlation and create a scatter plot.`

### Load Example Data
```python
result = await load_example(dataset_name="iris")
```

### Summary Statistics
```python
result = await summary_stats(
    data={"x": [1,2,3,4,5], "y": [2,4,6,8,10]}
)
```

### Linear Regression
```python
result = await linear_model(
    data={"x": [1,2,3,4,5], "y": [2,4,5,4,5]},
    formula="y ~ x"
)
```

### Create Visualization
```python
result = await scatter_plot(
    data={"x": [1,2,3,4,5], "y": [2,4,5,4,5]},
    x="x",
    y="y",
    title="My Plot",
    return_image=True
)
# result["image"] contains base64-encoded PNG
```

## Architecture

### Key Components

1. **FastMCP Framework** (`server_complete.py`)
   - Tool definitions using `@mcp.tool()` decorator
   - Resource definitions using `@mcp.resource()` decorator
   - Automatic JSON-RPC handling

2. **R Integration** (`r_integration/r_executor.py`)
   - Subprocess-based R execution
   - JSON data exchange
   - Automatic package installation

3. **Data Models** (Pydantic)
   - Type-safe input validation
   - Automatic documentation generation

## Differences from Original RMCP

### Advantages of FastMCP Implementation

- **Less Code**: ~1000 lines vs ~5000 lines in original
- **Simpler Structure**: Single file with clear tool definitions
- **Automatic Features**: Built-in help, validation, and error handling
- **Better Type Safety**: Pydantic models for all inputs

### Trade-offs

- **Less Customization**: FastMCP handles transport/protocol internally
- **Single Transport**: STDIO only (HTTP requires additional setup)
- **Simplified Error Handling**: Less granular error recovery

## Development

### Adding New Tools

1. Define input model:
```python
class MyInput(BaseModel):
    data: Dict[str, List[Any]]
    param: str = Field(description="Parameter description")
```

2. Add tool function:
```python
@mcp.tool()
async def my_tool(input: MyInput) -> Dict[str, Any]:
    \"\"\"Tool description.\"\"\"
    script = \"\"\"
    # R code here
    result <- list(output = "value")
    \"\"\"
    return await r_executor.execute(script, input.dict())
```

### Testing

Run the test script:
```bash
python3 test_server.py
```

## License

MIT License - Same as original RMCP

## Credits

Based on the original RMCP implementation, reimplemented using FastMCP for simplicity and maintainability.
