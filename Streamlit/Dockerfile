FROM python:3.12-slim

# ── System deps ───────────────────────────────────────────────────────────
# PySpark needs a JVM. default-jdk-headless keeps the image reasonably small.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        default-jdk-headless \
        procps \
        curl && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

WORKDIR /app

# ── Python deps (cached separately from source for faster rebuilds) ────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── App source, models, params.yaml, etc. ───────────────────────────────────
COPY . .

# Spark/PySpark defaults — avoids some slim-image quirks
ENV PYSPARK_PYTHON=python3
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "FastApi.app:app", "--host", "0.0.0.0", "--port", "8000"]