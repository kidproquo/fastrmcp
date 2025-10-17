"""
R Executor Module

Handles R script execution and data exchange between Python and R.
"""

import json
import subprocess
import tempfile
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class RExecutor:
    """Execute R scripts with data exchange through JSON."""

    def __init__(self):
        """Initialize R executor and verify R installation."""
        self.r_command = "R"
        self._verify_r_installation()
        self._ensure_packages()

    def _verify_r_installation(self):
        """Verify that R is installed and accessible."""
        try:
            result = subprocess.run(
                [self.r_command, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info("R installation verified")
                # Log R version
                version_line = result.stdout.split('\n')[0]
                logger.info(f"R version: {version_line}")
            else:
                raise RuntimeError("R is not properly installed")
        except FileNotFoundError:
            raise RuntimeError("R is not installed. Please install R to use this server.")
        except subprocess.TimeoutExpired:
            raise RuntimeError("R verification timed out")

    def _ensure_packages(self):
        """Ensure required R packages are installed."""
        check_script = """
        # List of required packages (matching Dockerfile)
        required_packages <- c(
            "jsonlite",      # JSON data exchange
            "ggplot2",       # Visualization
            "base64enc",     # Image encoding
            "plm",           # Panel regression
            "AER",           # Instrumental variables
            "vars",          # VAR models
            "forecast",      # Time series forecasting
            "tseries",       # Time series tests
            "dplyr",         # Data manipulation
            "nortest",       # Normality tests
            "rpart",         # Decision trees
            "randomForest",  # Random forests
            "reshape2"       # Data reshaping for heatmaps
        )

        missing_packages <- required_packages[!required_packages %in% installed.packages()[,"Package"]]

        if (length(missing_packages) > 0) {
            cat("MISSING:", paste(missing_packages, collapse = ","))
        } else {
            cat("OK")
        }
        """

        result = subprocess.run(
            [self.r_command, "--slave", "--no-save", "--no-restore"],
            input=check_script,
            capture_output=True,
            text=True,
            timeout=10
        )

        output = result.stdout.strip()
        if output.startswith("MISSING:"):
            missing = output.replace("MISSING:", "").split(",")
            logger.warning(f"Missing R packages: {missing}")
            logger.info("Installing missing packages...")
            self._install_packages(missing)
        elif output == "OK":
            logger.info("All required R packages are installed")

    def _install_packages(self, packages: list):
        """Install missing R packages."""
        for package in packages:
            install_script = f"""
            install.packages("{package}", repos = "https://cran.r-project.org", quiet = TRUE)
            """
            try:
                subprocess.run(
                    [self.r_command, "--slave", "--no-save", "--no-restore"],
                    input=install_script,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                logger.info(f"Installed R package: {package}")
            except subprocess.TimeoutExpired:
                logger.error(f"Timeout installing R package: {package}")

    async def execute(self, script: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute an R script with optional data input.

        Args:
            script: R script to execute
            data: Optional data to pass to R

        Returns:
            Result from R as a dictionary
        """
        # Prepare the full R script with JSON I/O
        full_script = """
        # Load jsonlite safely
        tryCatch({
            library(jsonlite)
        }, error = function(e) {
            cat(paste0('{"error": "Missing R package: jsonlite. Please install it with install.packages(\\"jsonlite\\")"}'))
            quit(status = 1)
        })

        # Function to safely convert R objects to JSON-serializable format
        safe_convert <- function(obj) {
            if (is.data.frame(obj) || is.matrix(obj)) {
                as.list(as.data.frame(obj))
            } else if (is.list(obj)) {
                lapply(obj, safe_convert)
            } else if (is.factor(obj)) {
                as.character(obj)
            } else {
                obj
            }
        }

        # Read input data if provided
        """

        if data:
            # Write data to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(data, f)
                data_file = f.name

            full_script += f"""
        input_data <- fromJSON("{data_file}")
        # Make data available in the environment
        for (name in names(input_data)) {{
            assign(name, input_data[[name]], envir = .GlobalEnv)
        }}
        """

        # Add user script
        full_script += f"""
        # User script
        {script}

        # Convert result to JSON
        if (exists("result")) {{
            result <- safe_convert(result)
            output <- toJSON(result, auto_unbox = TRUE, null = "null", na = "null")
        }} else {{
            output <- toJSON(list(error = "No result variable defined"), auto_unbox = TRUE)
        }}

        cat(output)
        """

        try:
            # Execute R script
            process = subprocess.run(
                [self.r_command, "--slave", "--no-save", "--no-restore"],
                input=full_script,
                capture_output=True,
                text=True,
                timeout=30
            )

            # Clean up temp file if created
            if data and 'data_file' in locals():
                try:
                    Path(data_file).unlink()
                except:
                    pass

            if process.returncode != 0:
                error_msg = process.stderr.strip() or "Unknown R error"
                logger.error(f"R execution error: {error_msg}")
                return {"error": error_msg}

            # Parse JSON output
            output = process.stdout.strip()
            if not output:
                return {"error": "No output from R script"}

            try:
                result = json.loads(output)
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse R output as JSON: {output}")
                return {"error": f"Invalid JSON from R: {str(e)}", "raw_output": output}

        except subprocess.TimeoutExpired:
            return {"error": "R script execution timed out"}
        except Exception as e:
            logger.error(f"Unexpected error executing R script: {e}")
            return {"error": f"Execution error: {str(e)}"}