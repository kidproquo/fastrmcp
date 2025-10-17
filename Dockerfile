FROM rocker/r2u:noble

# System deps (Python + build tools + common libs)
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev \
        build-essential git \
        libcurl4-openssl-dev libssl-dev libxml2-dev \
        ca-certificates wget; \
    rm -rf /var/lib/apt/lists/*

# Make r2u behavior explicit for scripts
RUN echo "options(bspm.enable=TRUE, bspm.quiet=TRUE)" >> /etc/R/Rprofile.site

# Preinstall R stack (binaries via r2u/bspm => fast)
RUN R -q -e "install.packages(c( \
  'jsonlite','plm','lmtest','sandwich','AER','dplyr', \
  'forecast','vars','urca','tseries','nortest','car', \
  'rpart','randomForest','ggplot2','gridExtra','tidyr', \
  'rlang','readxl','openxlsx','base64enc','reshape2','knitr','broom', \
  'MASS','boot','survival','nlme','mgcv','lme4','glmnet', \
  'e1071','caret','nnet','gbm','xgboost','kernlab','cluster', \
  'zoo','xts','TTR','quantmod','data.table','lattice', \
  'corrplot','viridis','RColorBrewer','lavaan' \
))"


RUN mkdir /app
COPY rmcp_fastmcp/requirements.txt /app/.

# ---- Python: create a venv to avoid PEP 668 issues ----
ENV VENV=/opt/venv
RUN set -eux; \
    python3 -m venv "$VENV"; \
    . "$VENV/bin/activate"; \
    pip install --upgrade pip; \
    pip install -r /app/requirements.txt

# Ensure venv tools are first on PATH for subsequent steps/CI
ENV PATH="$VENV/bin:$PATH"

EXPOSE 3003

COPY rmcp_fastmcp /app/rmcp_fastmcp
WORKDIR /app/rmcp_fastmcp
CMD ["python", "run_server.py", "http", "3003"]
