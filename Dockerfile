FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000 \
    HOST=0.0.0.0

RUN python -m pip install --no-cache-dir delta-exchange-mcp==0.7.0 google-auth==2.40.3 requests==2.32.5 \
    && command -v delta-exchange-mcp \
    && useradd --create-home --uid 10001 dashboard

WORKDIR /app
COPY serve_dashboard.py ./
COPY ethresearch/__init__.py ethresearch/delta_mcp.py ./ethresearch/
COPY config/production_strategy.json ./config/
COPY web/index.html web/app.js web/styles.css ./web/

USER dashboard
EXPOSE 10000
CMD ["python", "serve_dashboard.py"]
