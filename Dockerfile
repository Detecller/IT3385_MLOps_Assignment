# Base image
FROM python:3.10-slim

# Install curl & build dependencies
RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy full project
COPY . .

# Install dependencies from Poetry
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Streamlit config
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ENABLECORS=false

# Expose port
EXPOSE 8501

# Run Streamlit app
CMD streamlit run Introduction.py --server.port $PORT --server.address 0.0.0.0 --server.headless true