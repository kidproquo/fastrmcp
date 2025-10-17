#!/usr/bin/env Rscript

# Install script for all required R packages
# Run this script to ensure all necessary packages are installed

cat("Installing required R packages for FastMCP R Statistical Server...\n\n")

# List of required packages
required_packages <- c(
    "jsonlite",      # JSON data exchange (essential)
    "ggplot2",       # Visualization
    "base64enc",     # Image encoding
    "plm",           # Panel regression
    "AER",           # Instrumental variables
    "vars",          # VAR models
    "forecast",      # Time series forecasting
    "tseries",       # Time series tests
    "dplyr",         # Data manipulation
    "nortest",       # Normality tests
    "DescTools",     # Winsorization
    "rpart",         # Decision trees
    "randomForest",  # Random forests
    "reshape2"       # Data reshaping for heatmaps
)

# Check which packages are missing
installed_packages <- installed.packages()[,"Package"]
missing_packages <- required_packages[!required_packages %in% installed_packages]

if (length(missing_packages) == 0) {
    cat("All required packages are already installed!\n")
} else {
    cat(paste("Missing packages:", paste(missing_packages, collapse = ", "), "\n\n"))

    # Install missing packages
    for (package in missing_packages) {
        cat(paste("Installing", package, "...\n"))
        tryCatch({
            install.packages(package,
                           repos = "https://cran.r-project.org",
                           quiet = TRUE,
                           dependencies = TRUE)
            cat(paste("  ✓", package, "installed successfully\n"))
        }, error = function(e) {
            cat(paste("  ✗ Failed to install", package, ":", e$message, "\n"))
        })
    }
}

# Verify installation
cat("\n\nVerifying installation...\n")
still_missing <- required_packages[!required_packages %in% installed.packages()[,"Package"]]

if (length(still_missing) == 0) {
    cat("✓ All packages installed successfully!\n")
} else {
    cat(paste("✗ The following packages could not be installed:",
              paste(still_missing, collapse = ", "), "\n"))
    cat("You may need to install them manually or check for system dependencies.\n")
}

# Test loading each package
cat("\nTesting package loading...\n")
for (package in required_packages) {
    if (package %in% installed.packages()[,"Package"]) {
        tryCatch({
            suppressPackageStartupMessages(library(package, character.only = TRUE))
            cat(paste("  ✓", package, "loads successfully\n"))
        }, error = function(e) {
            cat(paste("  ✗", package, "installed but fails to load:", e$message, "\n"))
        })
    }
}

cat("\nInstallation complete!\n")