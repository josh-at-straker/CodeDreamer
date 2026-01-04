# CodeDreamer - Simple Docker with Pixi
#
# Build: docker build -t codedreamer .
# Run:   docker compose up -d

FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

WORKDIR /app

# Install Pixi
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL https://pixi.sh/install.sh | bash \
    && rm -rf /var/lib/apt/lists/*

# Add pixi to PATH and set CUDA paths for llama-cpp-python build
ENV PATH="/root/.pixi/bin:$PATH" \
    CUDACXX=/usr/local/cuda/bin/nvcc \
    CUDA_PATH=/usr/local/cuda

# Copy project files
COPY pyproject.toml pixi.lock README.md ./
COPY codedreamer/ codedreamer/

# Install all dependencies via Pixi
RUN pixi install && pixi run setup-cuda

# Create data directories
RUN mkdir -p /app/dreams /app/data

# Environment defaults
ENV DREAMER_DREAMS_DIR=/app/dreams \
    DREAMER_DB_PATH=/app/data/codedreamer.db \
    DREAMER_GRAPH_PATH=/app/data/graph.json \
    DREAMER_N_GPU_LAYERS=99 \
    DREAMER_LOG_LEVEL=INFO \
    DREAMER_REASONING_MAX_TOKENS=4000 \
    DREAMER_CODER_MAX_TOKENS=3000

VOLUME ["/app/dreams", "/app/data", "/models", "/codebase"]
EXPOSE 8080

HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["pixi", "run"]
CMD ["codedreamer", "serve", "--host", "0.0.0.0", "--port", "8080"]
