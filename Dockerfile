FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for matplotlib
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY energy_forecast/ energy_forecast/
COPY main.py .

# Default: run initial phase + 3 live cycles for demo
CMD ["python", "-u", "main.py", "--days", "14", "--live-cycles", "3", "--fast-mode"]
