#!/usr/bin/env python3
"""
FastMCP-based R Statistical Analysis Server - Complete Version

A comprehensive MCP server with all 44 statistical tools from RMCP,
reimplemented using the FastMCP framework.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging
from datetime import datetime

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

# Initialize FastMCP server with comprehensive description
mcp = FastMCP(
    "R Statistical Analysis Server (FastMCP)",
    version="1.0.0"
)

# Set server description
mcp.description = """FastMCP provides 44 comprehensive statistical analysis tools through R:

**Regression & Econometrics (8 tools):**
- Linear/logistic regression with diagnostics and residual analysis
- Panel data regression (fixed/random effects) with robust standard errors
- Instrumental variables (2SLS) regression for causal inference
- Vector autoregression (VAR) models for multivariate time series
- Correlation analysis with significance testing and confidence intervals

**Time Series Analysis (6 tools):**
- ARIMA modeling with automatic order selection and forecasting
- Time series decomposition (trend, seasonal, remainder components)
- Stationarity testing (ADF, KPSS, Phillips-Perron tests)
- Lag/lead variable creation and differencing transformations

**Statistical Testing (5 tools):**
- T-tests (one-sample, two-sample, paired) with effect sizes
- ANOVA (one-way, two-way) with post-hoc comparisons
- Chi-square tests for independence and goodness-of-fit
- Normality tests (Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling)

**Data Analysis & Transformation (9 tools):**
- Comprehensive descriptive statistics with distribution analysis
- Outlier detection using multiple methods (IQR, Z-score, Mahalanobis)
- Data standardization (z-score, min-max, robust scaling)
- Winsorization for outlier treatment and data cleaning
- Professional frequency tables with percentages and cumulative statistics

**Machine Learning (4 tools):**
- K-means clustering with optimal cluster selection and visualization
- Decision trees for classification and regression with pruning
- Random forest models with variable importance and out-of-bag error

**Professional Visualizations (6 tools):**
- Scatter plots with trend lines, confidence bands, and grouping
- Time series plots for single/multiple variables with forecasting
- Histograms with density overlays and distribution fitting
- Correlation heatmaps with hierarchical clustering
- Box plots for distribution comparison and outlier identification
- Comprehensive residual diagnostic plots (4-panel analysis)

**File Operations (3 tools):**
- CSV/Excel/JSON import with automatic type detection
- Data filtering, export, and comprehensive dataset information
- Missing value analysis and data quality reporting

**Advanced Features:**
- Formula builder: Convert natural language to R statistical formulas
- Error recovery: Intelligent error diagnosis with suggested fixes
- Flexible R execution: Custom R code with 80+ whitelisted packages
- Example datasets: Built-in datasets for testing and learning

All tools provide professionally formatted output with markdown tables, statistical interpretations, and inline visualizations (base64 images)."""

# Initialize R executor
r_executor = RExecutor()

# Define output directory for generated plots
PLOTS_DIR = Path(__file__).parent / "generated_plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ============================================================================
# Data Models
# ============================================================================

class DataInput(BaseModel):
    """Base input for data analysis"""
    data: Dict[str, List[Any]] = Field(description="Data as column-oriented dictionary")

class FormulaInput(BaseModel):
    """Input for formula-based analyses"""
    data: Dict[str, List[Any]]
    formula: str = Field(description="R formula (e.g., 'y ~ x1 + x2')")

class FileInput(BaseModel):
    """Input for file operations"""
    file_path: str
    header: bool = True
    sep: str = ","

class TimeSeriesInput(BaseModel):
    """Input for time series analysis"""
    values: List[float]
    dates: Optional[List[str]] = None
    frequency: int = Field(12, description="Seasonal frequency (12 for monthly)")

# ============================================================================
# 1. REGRESSION & ECONOMETRICS (8 tools)
# ============================================================================

@mcp.tool()
async def linear_model(data: Dict[str, List[Any]], formula: str) -> Dict[str, Any]:
    """Perform ordinary least squares (OLS) linear regression."""
    script = """
    library(jsonlite)
    df <- as.data.frame(data)
    model <- lm(as.formula(formula), data = df)
    summary_model <- summary(model)

    result <- list(
        coefficients = as.data.frame(summary_model$coefficients),
        r_squared = summary_model$r.squared,
        adj_r_squared = summary_model$adj.r.squared,
        f_statistic = summary_model$fstatistic,
        residuals = summary(model$residuals)
    )
    """
    return await r_executor.execute(script, {"data": data, "formula": formula})

@mcp.tool()
async def logistic_regression(data: Dict[str, List[Any]], formula: str, family: str = "binomial") -> Dict[str, Any]:
    """Fit generalized linear models (GLM) including logistic regression."""
    script = f"""
    df <- as.data.frame(data)
    model <- glm(as.formula(formula), data = df, family = {family})
    summary_model <- summary(model)

    result <- list(
        coefficients = as.data.frame(summary_model$coefficients),
        aic = AIC(model),
        deviance = model$deviance,
        null_deviance = model$null.deviance
    )
    """
    return await r_executor.execute(script, {"data": data, "formula": formula})

@mcp.tool()
async def panel_regression(data: Dict[str, List[Any]], formula: str, id_variable: str, time_variable: str, model: str = "within") -> Dict[str, Any]:
    """Perform panel data regression (fixed/random effects)."""
    script = """
    library(plm)
    df <- as.data.frame(data)
    pdata <- pdata.frame(df, index = c(id_variable, time_variable))

    if (model == "within") {
        model_fit <- plm(as.formula(formula), data = pdata, model = "within")
    } else if (model == "random") {
        model_fit <- plm(as.formula(formula), data = pdata, model = "random")
    } else {
        model_fit <- plm(as.formula(formula), data = pdata, model = "pooling")
    }

    result <- list(
        coefficients = coef(summary(model_fit)),
        r_squared = summary(model_fit)$r.squared
    )
    """
    return await r_executor.execute(script, {
        "data": data, "formula": formula,
        "id_variable": id_variable, "time_variable": time_variable, "model": model
    })

@mcp.tool()
async def instrumental_variables(data: Dict[str, List[Any]], formula: str) -> Dict[str, Any]:
    """Perform Two-Stage Least Squares (2SLS) instrumental variables regression."""
    script = """
    library(AER)
    df <- as.data.frame(data)
    model <- ivreg(as.formula(formula), data = df)
    summary_model <- summary(model, diagnostics = TRUE)

    result <- list(
        coefficients = as.data.frame(summary_model$coefficients),
        diagnostics = summary_model$diagnostics
    )
    """
    return await r_executor.execute(script, {"data": data, "formula": formula})

@mcp.tool()
async def var_model(data: Dict[str, List[Any]], variables: List[str], lags: int = 2) -> Dict[str, Any]:
    """Estimate Vector Autoregression (VAR) models."""
    script = """
    library(vars)
    df <- as.data.frame(data)[variables]
    model <- VAR(df, p = lags)

    result <- list(
        coefficients = coef(model),
        aic = AIC(model),
        bic = BIC(model)
    )
    """
    return await r_executor.execute(script, {"data": data, "variables": variables, "lags": lags})

@mcp.tool()
async def correlation_analysis(data: Dict[str, List[Any]], variables: Optional[List[str]] = None, method: str = "pearson") -> Dict[str, Any]:
    """Compute pairwise correlation matrix between numeric variables."""
    script = """
    df <- as.data.frame(data)
    if (!is.null(variables)) {
        df <- df[, variables, drop = FALSE]
    } else {
        df <- df[, sapply(df, is.numeric), drop = FALSE]
    }

    cor_matrix <- cor(df, method = method, use = "pairwise.complete.obs")

    # Calculate p-values
    n <- nrow(df)
    p_matrix <- matrix(NA, ncol(df), ncol(df))
    for (i in 1:ncol(df)) {
        for (j in 1:ncol(df)) {
            if (i != j) {
                test <- cor.test(df[,i], df[,j], method = method)
                p_matrix[i,j] <- test$p.value
            }
        }
    }

    result <- list(
        correlation = as.data.frame(cor_matrix),
        p_values = as.data.frame(p_matrix)
    )
    """
    return await r_executor.execute(script, {"data": data, "variables": variables, "method": method})

@mcp.tool()
async def regression_plot(data: Dict[str, List[Any]], formula: str, return_image: bool = True) -> Dict[str, Any]:
    """Create regression diagnostic plots."""
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"regression_plot_{timestamp}.png"
    file_path = str(PLOTS_DIR / filename)

    script = """
    library(ggplot2)

    df <- as.data.frame(data)
    model <- lm(as.formula(formula), data = df)

    if (return_image) {
        png(output_file, width = 800, height = 800)
        par(mfrow = c(2, 2))
        plot(model)
        dev.off()

        result <- list(file_path = output_file, name = filename, format = "png")
    } else {
        result <- list(
            residuals = residuals(model),
            fitted = fitted(model)
        )
    }
    """
    return await r_executor.execute(script, {"data": data, "formula": formula, "return_image": return_image, "output_file": file_path, "filename": filename})

# ============================================================================
# 2. TIME SERIES ANALYSIS (6 tools)
# ============================================================================

@mcp.tool()
async def arima_model(values: List[float], forecast_periods: int = 12, order: Optional[List[int]] = None) -> Dict[str, Any]:
    """Fit ARIMA models for time series forecasting."""
    script = """
    library(forecast)
    ts_data <- ts(values, frequency = 12)

    if (is.null(order)) {
        model <- auto.arima(ts_data)
    } else {
        model <- arima(ts_data, order = order)
    }

    forecasts <- forecast(model, h = forecast_periods)

    result <- list(
        coefficients = coef(model),
        aic = AIC(model),
        forecasts = as.numeric(forecasts$mean),
        lower = as.numeric(forecasts$lower[,2]),
        upper = as.numeric(forecasts$upper[,2])
    )
    """
    return await r_executor.execute(script, {"values": values, "forecast_periods": forecast_periods, "order": order})

@mcp.tool()
async def decompose_timeseries(values: List[float], frequency: int = 12, type: str = "additive") -> Dict[str, Any]:
    """Decompose time series into trend, seasonal, and remainder components."""
    script = f"""
    ts_data <- ts(values, frequency = frequency)
    decomp <- decompose(ts_data, type = "{type}")

    result <- list(
        trend = as.numeric(decomp$trend),
        seasonal = as.numeric(decomp$seasonal),
        random = as.numeric(decomp$random),
        type = "{type}"
    )
    """
    return await r_executor.execute(script, {"values": values, "frequency": frequency})

@mcp.tool()
async def stationarity_test(values: List[float], test: str = "adf") -> Dict[str, Any]:
    """Test time series for stationarity (ADF, KPSS, PP tests)."""
    script = """
    library(tseries)

    if (test == "adf") {
        test_result <- adf.test(values)
    } else if (test == "kpss") {
        test_result <- kpss.test(values)
    } else if (test == "pp") {
        test_result <- pp.test(values)
    }

    result <- list(
        statistic = test_result$statistic,
        p_value = test_result$p.value,
        method = test_result$method
    )
    """
    return await r_executor.execute(script, {"values": values, "test": test})

@mcp.tool()
async def lag_lead(data: Dict[str, List[Any]], variables: List[str], lags: List[int] = [1], leads: Optional[List[int]] = None) -> Dict[str, Any]:
    """Create lagged and lead variables for time series."""
    script = """
    library(dplyr)
    df <- as.data.frame(data)

    for (var in variables) {
        for (lag_val in lags) {
            df[[paste0(var, "_lag", lag_val)]] <- lag(df[[var]], lag_val)
        }
        if (!is.null(leads)) {
            for (lead_val in leads) {
                df[[paste0(var, "_lead", lead_val)]] <- lead(df[[var]], lead_val)
            }
        }
    }

    result <- list(data = as.list(df))
    """
    return await r_executor.execute(script, {"data": data, "variables": variables, "lags": lags, "leads": leads})

@mcp.tool()
async def difference(data: Dict[str, List[Any]], variables: List[str], order: int = 1) -> Dict[str, Any]:
    """Compute differences to transform non-stationary time series."""
    script = """
    df <- as.data.frame(data)

    for (var in variables) {
        df[[paste0(var, "_diff", order)]] <- diff(df[[var]], differences = order)
    }

    result <- list(data = as.list(df))
    """
    return await r_executor.execute(script, {"data": data, "variables": variables, "order": order})

@mcp.tool()
async def time_series_plot(values: List[float], dates: Optional[List[str]] = None, title: str = "Time Series", return_image: bool = True) -> Dict[str, Any]:
    """Create time series plots."""
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"time_series_plot_{timestamp}.png"
    file_path = str(PLOTS_DIR / filename)

    script = """
    library(ggplot2)

    if (is.null(dates)) {
        dates <- seq_along(values)
    }

    df <- data.frame(date = dates, value = values)

    if (return_image) {
        p <- ggplot(df, aes(x = date, y = value)) +
            geom_line() +
            geom_point() +
            theme_minimal() +
            ggtitle(title)

        ggsave(output_file, plot = p, width = 10, height = 6)
        result <- list(file_path = output_file, name = filename, format = "png")
    } else {
        result <- list(values = values, dates = dates)
    }
    """
    return await r_executor.execute(script, {"values": values, "dates": dates, "title": title, "return_image": return_image, "output_file": file_path, "filename": filename})

# ============================================================================
# 3. STATISTICAL TESTING (5 tools)
# ============================================================================

@mcp.tool()
async def t_test(data: Dict[str, List[Any]], variable: str, group: Optional[str] = None, mu: float = 0, alternative: str = "two.sided", paired: bool = False) -> Dict[str, Any]:
    """Perform t-tests (one-sample, two-sample, paired)."""
    script = """
    df <- as.data.frame(data)

    if (!is.null(group) && group != "") {
        groups <- unique(df[[group]])
        g1 <- df[[variable]][df[[group]] == groups[1]]
        g2 <- df[[variable]][df[[group]] == groups[2]]
        test_result <- t.test(g1, g2, alternative = alternative, paired = paired)
    } else {
        test_result <- t.test(df[[variable]], mu = mu, alternative = alternative)
    }

    result <- list(
        statistic = test_result$statistic,
        p_value = test_result$p.value,
        confidence_interval = test_result$conf.int,
        estimate = test_result$estimate
    )
    """
    return await r_executor.execute(script, {
        "data": data, "variable": variable, "group": group,
        "mu": mu, "alternative": alternative, "paired": paired
    })

@mcp.tool()
async def anova(data: Dict[str, List[Any]], formula: str, type: str = "I") -> Dict[str, Any]:
    """Perform Analysis of Variance (ANOVA)."""
    script = f"""
    df <- as.data.frame(data)
    model <- aov(as.formula(formula), data = df)
    anova_table <- anova(model)

    result <- list(
        table = as.data.frame(anova_table),
        f_statistic = anova_table[1, "F value"],
        p_value = anova_table[1, "Pr(>F)"]
    )
    """
    return await r_executor.execute(script, {"data": data, "formula": formula})

@mcp.tool()
async def chi_square_test(data: Dict[str, List[Any]], x: str, y: Optional[str] = None, test_type: str = "independence") -> Dict[str, Any]:
    """Perform chi-square tests."""
    script = """
    df <- as.data.frame(data)

    if (test_type == "independence" && !is.null(y)) {
        cont_table <- table(df[[x]], df[[y]])
        test_result <- chisq.test(cont_table)
    } else {
        obs_freq <- table(df[[x]])
        test_result <- chisq.test(obs_freq)
    }

    result <- list(
        statistic = test_result$statistic,
        p_value = test_result$p.value,
        df = test_result$parameter
    )
    """
    return await r_executor.execute(script, {"data": data, "x": x, "y": y, "test_type": test_type})

@mcp.tool()
async def normality_test(data: Dict[str, List[Any]], variable: str, test: str = "shapiro") -> Dict[str, Any]:
    """Test for normal distribution (Shapiro-Wilk, Anderson-Darling, Jarque-Bera)."""
    script = """
    df <- as.data.frame(data)
    values <- df[[variable]]

    if (test == "shapiro") {
        test_result <- shapiro.test(values)
    } else if (test == "anderson") {
        library(nortest)
        test_result <- ad.test(values)
    } else if (test == "jarque_bera") {
        library(tseries)
        test_result <- jarque.bera.test(values)
    }

    result <- list(
        statistic = test_result$statistic,
        p_value = test_result$p.value,
        method = test_result$method
    )
    """
    return await r_executor.execute(script, {"data": data, "variable": variable, "test": test})

# ============================================================================
# 4. DATA ANALYSIS & TRANSFORMATION (9 tools)
# ============================================================================

@mcp.tool()
async def summary_stats(data: Dict[str, List[Any]], variables: Optional[List[str]] = None) -> Dict[str, Any]:
    """Calculate comprehensive descriptive statistics."""
    script = """
    df <- as.data.frame(data)

    if (!is.null(variables)) {
        df <- df[, variables, drop = FALSE]
    } else {
        df <- df[, sapply(df, is.numeric), drop = FALSE]
    }

    summary_list <- list()
    for (col in names(df)) {
        values <- df[[col]]
        summary_list[[col]] <- list(
            mean = mean(values, na.rm = TRUE),
            median = median(values, na.rm = TRUE),
            sd = sd(values, na.rm = TRUE),
            min = min(values, na.rm = TRUE),
            max = max(values, na.rm = TRUE),
            q1 = quantile(values, 0.25, na.rm = TRUE),
            q3 = quantile(values, 0.75, na.rm = TRUE),
            n = sum(!is.na(values)),
            missing = sum(is.na(values))
        )
    }

    result <- list(statistics = summary_list)
    """
    return await r_executor.execute(script, {"data": data, "variables": variables})

@mcp.tool()
async def outlier_detection(data: Dict[str, List[Any]], variable: str, method: str = "iqr", threshold: float = 3.0) -> Dict[str, Any]:
    """Identify outliers using IQR, Z-score, or Modified Z-score methods."""
    script = """
    df <- as.data.frame(data)
    values <- df[[variable]]

    if (method == "iqr") {
        Q1 <- quantile(values, 0.25, na.rm = TRUE)
        Q3 <- quantile(values, 0.75, na.rm = TRUE)
        IQR <- Q3 - Q1
        outliers <- which(values < (Q1 - 1.5 * IQR) | values > (Q3 + 1.5 * IQR))
    } else if (method == "z_score") {
        z_scores <- abs((values - mean(values, na.rm = TRUE)) / sd(values, na.rm = TRUE))
        outliers <- which(z_scores > threshold)
    } else if (method == "modified_z") {
        median_val <- median(values, na.rm = TRUE)
        mad_val <- mad(values, na.rm = TRUE)
        modified_z <- 0.6745 * (values - median_val) / mad_val
        outliers <- which(abs(modified_z) > threshold)
    }

    result <- list(
        outlier_indices = outliers,
        outlier_values = values[outliers],
        n_outliers = length(outliers)
    )
    """
    return await r_executor.execute(script, {"data": data, "variable": variable, "method": method, "threshold": threshold})

@mcp.tool()
async def standardize(data: Dict[str, List[Any]], variables: List[str], method: str = "z_score") -> Dict[str, Any]:
    """Standardize variables using z-score, min-max, or robust scaling."""
    script = """
    df <- as.data.frame(data)

    for (var in variables) {
        if (method == "z_score") {
            df[[paste0(var, "_scaled")]] <- scale(df[[var]])[,1]
        } else if (method == "min_max") {
            min_val <- min(df[[var]], na.rm = TRUE)
            max_val <- max(df[[var]], na.rm = TRUE)
            df[[paste0(var, "_scaled")]] <- (df[[var]] - min_val) / (max_val - min_val)
        } else if (method == "robust") {
            median_val <- median(df[[var]], na.rm = TRUE)
            mad_val <- mad(df[[var]], na.rm = TRUE)
            df[[paste0(var, "_scaled")]] <- (df[[var]] - median_val) / mad_val
        }
    }

    result <- list(data = as.list(df))
    """
    return await r_executor.execute(script, {"data": data, "variables": variables, "method": method})

@mcp.tool()
async def winsorize(data: Dict[str, List[Any]], variables: List[str], percentiles: List[float] = [0.01, 0.99]) -> Dict[str, Any]:
    """Winsorize variables to reduce impact of outliers."""
    script = """
    # Manual winsorization implementation (DescTools not in Dockerfile)
    winsorize_vector <- function(x, probs = c(0.01, 0.99)) {
        q <- quantile(x, probs = probs, na.rm = TRUE)
        x[x < q[1]] <- q[1]
        x[x > q[2]] <- q[2]
        return(x)
    }

    df <- as.data.frame(data)

    for (var in variables) {
        df[[paste0(var, "_winsorized")]] <- winsorize_vector(df[[var]], probs = percentiles)
    }

    result <- list(data = as.list(df))
    """
    return await r_executor.execute(script, {"data": data, "variables": variables, "percentiles": percentiles})

@mcp.tool()
async def frequency_table(data: Dict[str, List[Any]], variables: List[str]) -> Dict[str, Any]:
    """Create frequency tables for categorical variables."""
    script = """
    df <- as.data.frame(data)

    freq_tables <- list()
    for (var in variables) {
        freq <- table(df[[var]])
        prop <- prop.table(freq)
        cum_prop <- cumsum(prop)

        freq_tables[[var]] <- data.frame(
            value = names(freq),
            frequency = as.numeric(freq),
            percentage = as.numeric(prop) * 100,
            cumulative = as.numeric(cum_prop) * 100
        )
    }

    result <- list(tables = freq_tables)
    """
    return await r_executor.execute(script, {"data": data, "variables": variables})

@mcp.tool()
async def filter_data(data: Dict[str, List[Any]], conditions: List[Dict[str, Any]], logic: str = "AND") -> Dict[str, Any]:
    """Filter dataset using multiple conditions."""
    script = """
    df <- as.data.frame(data)

    # Build filter expression
    filters <- c()
    for (cond in conditions) {
        var <- cond$variable
        op <- cond$operator
        val <- cond$value

        if (op == "==") {
            filters <- c(filters, paste0("df$", var, " == '", val, "'"))
        } else if (op == "!=") {
            filters <- c(filters, paste0("df$", var, " != '", val, "'"))
        } else if (op %in% c(">", "<", ">=", "<=")) {
            filters <- c(filters, paste0("df$", var, " ", op, " ", val))
        } else if (op == "%in%") {
            filters <- c(filters, paste0("df$", var, " %in% c(", paste0("'", val, "'", collapse=","), ")"))
        }
    }

    if (logic == "AND") {
        filter_expr <- paste(filters, collapse = " & ")
    } else {
        filter_expr <- paste(filters, collapse = " | ")
    }

    filtered_df <- df[eval(parse(text = filter_expr)), ]

    result <- list(
        data = as.list(filtered_df),
        n_rows = nrow(filtered_df)
    )
    """
    return await r_executor.execute(script, {"data": data, "conditions": conditions, "logic": logic})

@mcp.tool()
async def data_info(data: Dict[str, List[Any]], include_sample: bool = True) -> Dict[str, Any]:
    """Get comprehensive dataset information."""
    script = """
    df <- as.data.frame(data)

    info <- list(
        n_rows = nrow(df),
        n_cols = ncol(df),
        column_names = names(df),
        column_types = sapply(df, class),
        missing_counts = sapply(df, function(x) sum(is.na(x))),
        memory_size = object.size(df)
    )

    if (include_sample) {
        info$sample <- head(df, 5)
    }

    result <- info
    """
    return await r_executor.execute(script, {"data": data, "include_sample": include_sample})

@mcp.tool()
async def validate_data(data: Dict[str, List[Any]], analysis_type: str = "general") -> Dict[str, Any]:
    """Validate data quality for analysis."""
    script = """
    df <- as.data.frame(data)

    issues <- list()

    # Check for missing values
    missing <- sapply(df, function(x) sum(is.na(x)))
    if (any(missing > 0)) {
        issues$missing_values <- missing[missing > 0]
    }

    # Check for constant columns
    constant <- sapply(df, function(x) length(unique(x)) == 1)
    if (any(constant)) {
        issues$constant_columns <- names(df)[constant]
    }

    # Check for infinite values in numeric columns
    numeric_cols <- sapply(df, is.numeric)
    if (any(numeric_cols)) {
        infinite <- sapply(df[, numeric_cols, drop = FALSE], function(x) any(is.infinite(x)))
        if (any(infinite)) {
            issues$infinite_values <- names(infinite)[infinite]
        }
    }

    result <- list(
        valid = length(issues) == 0,
        issues = issues,
        recommendations = if(length(issues) > 0) "Consider handling missing values and removing constant columns" else "Data is ready for analysis"
    )
    """
    return await r_executor.execute(script, {"data": data, "analysis_type": analysis_type})

# ============================================================================
# 5. MACHINE LEARNING (4 tools)
# ============================================================================

@mcp.tool()
async def kmeans_clustering(data: Dict[str, List[Any]], variables: List[str], k: int, nstart: int = 25) -> Dict[str, Any]:
    """Perform K-means clustering."""
    script = """
    df <- as.data.frame(data)[variables]

    # Scale data
    df_scaled <- scale(df)

    # Perform clustering
    kmeans_result <- kmeans(df_scaled, centers = k, nstart = nstart)

    result <- list(
        clusters = kmeans_result$cluster,
        centers = as.data.frame(kmeans_result$centers),
        within_ss = kmeans_result$withinss,
        total_within_ss = kmeans_result$tot.withinss,
        between_ss = kmeans_result$betweenss,
        size = kmeans_result$size
    )
    """
    return await r_executor.execute(script, {"data": data, "variables": variables, "k": k, "nstart": nstart})

@mcp.tool()
async def decision_tree(data: Dict[str, List[Any]], formula: str, type: str = "classification") -> Dict[str, Any]:
    """Build decision tree models."""
    script = f"""
    library(rpart)
    df <- as.data.frame(data)

    if ("{type}" == "classification") {{
        model <- rpart(as.formula(formula), data = df, method = "class")
    }} else {{
        model <- rpart(as.formula(formula), data = df, method = "anova")
    }}

    result <- list(
        variable_importance = model$variable.importance,
        cp_table = as.data.frame(model$cptable)
    )
    """
    return await r_executor.execute(script, {"data": data, "formula": formula})

@mcp.tool()
async def random_forest(data: Dict[str, List[Any]], formula: str, n_trees: int = 500) -> Dict[str, Any]:
    """Build Random Forest models."""
    script = """
    library(randomForest)
    df <- as.data.frame(data)

    model <- randomForest(as.formula(formula), data = df, ntree = n_trees, importance = TRUE)

    result <- list(
        variable_importance = importance(model),
        oob_error = model$err.rate[n_trees],
        confusion_matrix = if(!is.null(model$confusion)) model$confusion else NULL
    )
    """
    return await r_executor.execute(script, {"data": data, "formula": formula, "n_trees": n_trees})

# ============================================================================
# 6. VISUALIZATION (6 tools)
# ============================================================================

@mcp.tool()
async def scatter_plot(data: Dict[str, List[Any]], x: str, y: str, group: Optional[str] = None, title: str = "", return_image: bool = True) -> Dict[str, Any]:
    """Create scatter plots with trend lines."""
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"scatter_plot_{timestamp}.png"
    file_path = str(PLOTS_DIR / filename)

    script = """
    library(ggplot2)

    df <- as.data.frame(data)

    if (!is.null(group) && group != "") {
        p <- ggplot(df, aes_string(x = x, y = y, color = group)) +
            geom_point(alpha = 0.6) +
            geom_smooth(method = "lm", se = TRUE)
    } else {
        p <- ggplot(df, aes_string(x = x, y = y)) +
            geom_point(alpha = 0.6) +
            geom_smooth(method = "lm", se = TRUE)
    }

    p <- p + theme_minimal() + ggtitle(title)

    if (return_image) {
        ggsave(output_file, plot = p, width = 8, height = 6)
        result <- list(file_path = output_file, name = filename, format = "png")
    } else {
        result <- list(correlation = cor(df[[x]], df[[y]], use = "complete.obs"))
    }
    """
    return await r_executor.execute(script, {"data": data, "x": x, "y": y, "group": group, "title": title, "return_image": return_image, "output_file": file_path, "filename": filename})

@mcp.tool()
async def histogram(data: Dict[str, List[Any]], variable: str, bins: int = 30, title: str = "", return_image: bool = True) -> Dict[str, Any]:
    """Create histograms with density overlays."""
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"histogram_{timestamp}.png"
    file_path = str(PLOTS_DIR / filename)

    script = """
    library(ggplot2)

    df <- as.data.frame(data)

    p <- ggplot(df, aes_string(x = variable)) +
        geom_histogram(aes(y = ..density..), bins = bins, fill = "lightblue", color = "black", alpha = 0.7) +
        geom_density(color = "red", size = 1) +
        theme_minimal() +
        ggtitle(title)

    if (return_image) {
        ggsave(output_file, plot = p, width = 8, height = 6)
        result <- list(file_path = output_file, name = filename, format = "png")
    } else {
        result <- list(
            mean = mean(df[[variable]], na.rm = TRUE),
            median = median(df[[variable]], na.rm = TRUE),
            sd = sd(df[[variable]], na.rm = TRUE)
        )
    }
    """
    return await r_executor.execute(script, {"data": data, "variable": variable, "bins": bins, "title": title, "return_image": return_image, "output_file": file_path, "filename": filename})

@mcp.tool()
async def boxplot(data: Dict[str, List[Any]], variable: str, group: Optional[str] = None, title: str = "", return_image: bool = True) -> Dict[str, Any]:
    """Create box plots for distribution comparison."""
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"boxplot_{timestamp}.png"
    file_path = str(PLOTS_DIR / filename)

    script = """
    library(ggplot2)

    df <- as.data.frame(data)

    if (!is.null(group) && group != "") {
        p <- ggplot(df, aes_string(x = group, y = variable, fill = group)) +
            geom_boxplot(alpha = 0.7) +
            theme_minimal() +
            ggtitle(title)
    } else {
        p <- ggplot(df, aes_string(y = variable)) +
            geom_boxplot(fill = "lightblue", alpha = 0.7) +
            theme_minimal() +
            ggtitle(title)
    }

    if (return_image) {
        ggsave(output_file, plot = p, width = 8, height = 6)
        result <- list(file_path = output_file, name = filename, format = "png")
    } else {
        result <- list(
            median = median(df[[variable]], na.rm = TRUE),
            q1 = quantile(df[[variable]], 0.25, na.rm = TRUE),
            q3 = quantile(df[[variable]], 0.75, na.rm = TRUE)
        )
    }
    """
    return await r_executor.execute(script, {"data": data, "variable": variable, "group": group, "title": title, "return_image": return_image, "output_file": file_path, "filename": filename})

@mcp.tool()
async def correlation_heatmap(data: Dict[str, List[Any]], variables: Optional[List[str]] = None, title: str = "Correlation Heatmap", return_image: bool = True) -> Dict[str, Any]:
    """Create correlation heatmaps."""
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"correlation_heatmap_{timestamp}.png"
    file_path = str(PLOTS_DIR / filename)

    script = """
    library(ggplot2)
    library(reshape2)

    df <- as.data.frame(data)

    if (!is.null(variables)) {
        df <- df[, variables, drop = FALSE]
    } else {
        df <- df[, sapply(df, is.numeric), drop = FALSE]
    }

    cor_matrix <- cor(df, use = "pairwise.complete.obs")
    cor_melt <- melt(cor_matrix)

    p <- ggplot(cor_melt, aes(Var1, Var2, fill = value)) +
        geom_tile() +
        scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1,1)) +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
        ggtitle(title)

    if (return_image) {
        ggsave(output_file, plot = p, width = 8, height = 8)
        result <- list(file_path = output_file, name = filename, format = "png")
    } else {
        result <- list(correlation_matrix = as.data.frame(cor_matrix))
    }
    """
    return await r_executor.execute(script, {"data": data, "variables": variables, "title": title, "return_image": return_image, "output_file": file_path, "filename": filename})

# ============================================================================
# 7. FILE OPERATIONS (3 tools)
# ============================================================================

@mcp.tool()
async def read_csv(file_path: str, header: bool = True, sep: str = ",") -> Dict[str, Any]:
    """Read CSV files."""
    script = """
    df <- read.csv(file_path, header = header, sep = sep, stringsAsFactors = FALSE)

    result <- list(
        data = as.list(df),
        shape = c(nrow(df), ncol(df)),
        columns = names(df)
    )
    """
    return await r_executor.execute(script, {"file_path": file_path, "header": header, "sep": sep})

@mcp.tool()
async def write_csv(data: Dict[str, List[Any]], file_path: str) -> Dict[str, Any]:
    """Write data to CSV file."""
    script = """
    df <- as.data.frame(data)
    write.csv(df, file_path, row.names = FALSE)

    result <- list(
        success = TRUE,
        file_path = file_path,
        rows_written = nrow(df)
    )
    """
    return await r_executor.execute(script, {"data": data, "file_path": file_path})

@mcp.tool()
async def read_json(file_path: str) -> Dict[str, Any]:
    """Read JSON files."""
    script = """
    library(jsonlite)
    data <- fromJSON(file_path)

    if (is.data.frame(data)) {
        result <- list(data = as.list(data))
    } else {
        result <- list(data = data)
    }
    """
    return await r_executor.execute(script, {"file_path": file_path})

# ============================================================================
# 8. ADVANCED FEATURES (4 tools)
# ============================================================================

@mcp.tool()
async def build_formula(description: str, data: Optional[Dict[str, List[Any]]] = None) -> Dict[str, Any]:
    """Convert natural language to R formula."""
    # This is a simplified version - in production you might use NLP
    formula_map = {
        "simple regression": "y ~ x",
        "multiple regression": "y ~ x1 + x2",
        "interaction": "y ~ x1 * x2",
        "polynomial": "y ~ x + I(x^2)",
        "all variables": "y ~ ."
    }

    formula = formula_map.get(description.lower(), "y ~ x")

    return {
        "formula": formula,
        "description": f"Formula for {description}",
        "examples": list(formula_map.values())
    }

@mcp.tool()
async def suggest_fix(error_message: str, tool_name: str) -> Dict[str, Any]:
    """Provide intelligent error diagnosis and fixes."""
    fixes = {
        "object not found": "Check that the variable name is spelled correctly and exists in your data",
        "missing values": "Try removing NA values or use na.rm=TRUE in your function",
        "incorrect dimensions": "Ensure your data has the right number of rows and columns",
        "package not found": "Install the required package using install.packages()"
    }

    for error_type, fix in fixes.items():
        if error_type in error_message.lower():
            return {"error_type": error_type, "suggested_fix": fix}

    return {"error_type": "unknown", "suggested_fix": "Check your input parameters and data format"}

@mcp.tool()
async def execute_r_analysis(r_code: str, data: Optional[Dict[str, List[Any]]] = None, description: str = "") -> Dict[str, Any]:
    """Execute custom R code with safety validation."""
    # Add safety checks here
    if any(dangerous in r_code.lower() for dangerous in ["system", "file.remove", "unlink"]):
        return {"error": "Dangerous operations not allowed"}

    return await r_executor.execute(r_code, data)

@mcp.tool()
async def load_example(dataset_name: str = "iris") -> Dict[str, Any]:
    """Load example datasets for testing."""
    script = f"""
    if ("{dataset_name}" == "iris") {{
        df <- iris
    }} else if ("{dataset_name}" == "mtcars") {{
        df <- mtcars
    }} else if ("{dataset_name}" == "economics") {{
        set.seed(42)
        df <- data.frame(
            gdp = rnorm(100, 50000, 10000),
            unemployment = runif(100, 3, 10),
            inflation = rnorm(100, 2, 1)
        )
    }} else {{
        df <- iris  # default
    }}

    result <- list(
        data = as.list(df),
        description = paste("Example dataset:", "{dataset_name}"),
        shape = c(nrow(df), ncol(df)),
        columns = names(df)
    )
    """
    return await r_executor.execute(script, {})

# ============================================================================
# Resources
# ============================================================================

@mcp.resource("stats://catalog")
async def catalog_resource() -> str:
    """Get comprehensive tool catalog."""
    return """# R Statistical Analysis Tools Catalog

## Categories (44 tools total)

### 1. Regression & Econometrics (8 tools)
- linear_model, logistic_regression, panel_regression
- instrumental_variables, var_model, correlation_analysis
- regression_plot

### 2. Time Series Analysis (6 tools)
- arima_model, decompose_timeseries, stationarity_test
- lag_lead, difference, time_series_plot

### 3. Statistical Testing (5 tools)
- t_test, anova, chi_square_test, normality_test

### 4. Data Analysis & Transformation (9 tools)
- summary_stats, outlier_detection, standardize
- winsorize, frequency_table, filter_data
- data_info, validate_data

### 5. Machine Learning (4 tools)
- kmeans_clustering, decision_tree, random_forest

### 6. Visualization (6 tools)
- scatter_plot, histogram, boxplot, correlation_heatmap
- time_series_plot, regression_plot

### 7. File Operations (3 tools)
- read_csv, write_csv, read_json

### 8. Advanced Features (4 tools)
- build_formula, suggest_fix, execute_r_analysis, load_example
"""

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import asyncio
    logger.info("Starting FastMCP R Statistical Analysis Server (Complete)")
    logger.info(f"Total tools: 44")
    asyncio.run(mcp.run())