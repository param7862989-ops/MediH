# MediHaven synthetic demo, served by Waitress. Research mode (DATA_MODE=mimic) is not deployable.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libzbar0: optional pyzbar QR decoding; libglib2.0-0: runtime library of headless OpenCV
RUN apt-get update \
    && apt-get install -y --no-install-recommends libzbar0 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Run unprivileged; only the runtime data folders are writable
RUN useradd --create-home --uid 10001 medihaven \
    && mkdir -p database data/processed models_store logs \
    && chown -R medihaven:medihaven database data models_store logs
USER medihaven

# Secrets (FLASK_SECRET_KEY, JWT_SECRET_KEY, optional GEMINI_API_KEY) come from the platform's
# environment settings, never from the image. Without strong secrets the app refuses to start.
ENV FLASK_ENV=production \
    DATA_MODE=synthetic \
    PORT=8000
EXPOSE 8000

# The first start seeds the demo database and trains the models, so allow it time
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import os, urllib.request as u; u.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT', '8000'), timeout=4)"

CMD ["sh", "-c", "exec waitress-serve --host=0.0.0.0 --port=${PORT} --threads=8 wsgi:app"]
