FROM python:3.11-slim

WORKDIR /app

# System deps for spaCy, pypff, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# spaCy models
RUN python -m spacy download en_core_web_sm && \
    python -m spacy download nl_core_news_sm

# App code
COPY src/ src/
COPY config/ config/

# Gradio port
EXPOSE 7861

# Launch app — mode and paths from env vars
CMD ["python", "-m", "src.app", "--port", "7861"]
