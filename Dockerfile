FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=10000 HOST=0.0.0.0
RUN python -m pip install --no-cache-dir delta-exchange-mcp==0.7.0 google-auth==2.40.3 requests==2.32.5 \
    && useradd --create-home --uid 10001 dashboard
WORKDIR /app
COPY --chown=dashboard:dashboard serve_dashboard.py ./
COPY --chown=dashboard:dashboard ethresearch/__init__.py ethresearch/delta_mcp.py ethresearch/crt.py ethresearch/crt_live.py ./ethresearch/
COPY --chown=dashboard:dashboard config/production_strategy.json ./config/
COPY --chown=dashboard:dashboard web/index.html web/app.js web/styles.css ./web/
RUN mkdir -p /app/runtime/crt && chown -R dashboard:dashboard /app/runtime
USER dashboard
EXPOSE 10000
CMD ["python", "serve_dashboard.py"]
