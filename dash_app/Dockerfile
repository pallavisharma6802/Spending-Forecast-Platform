FROM python:3.12-slim
# openjdk-21-jdk-headless: PySpark 4.x requires Java 17+; this base image's
# Debian release only carries 21/25, not 17, so 21 it is - running in local
# mode here, no separate Hadoop/YARN cluster needed. This is what lets the
# deployed app do real distributed compute inside a single container.
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-21-jdk-headless \
    && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PYSPARK_PYTHON=/usr/local/bin/python
ENV PYSPARK_DRIVER_PYTHON=/usr/local/bin/python

WORKDIR /app
COPY dash_app/requirements.txt dash_app/requirements.txt
RUN pip install --no-cache-dir -r dash_app/requirements.txt

COPY core/ core/
COPY agent/ agent/
COPY dash_app/ dash_app/
COPY frontend/data/ frontend/data/

EXPOSE 8501
CMD ["gunicorn", "--bind", "0.0.0.0:8501", "--workers", "1", "--timeout", "180", "dash_app.app:server"]
