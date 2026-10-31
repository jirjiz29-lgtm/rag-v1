#!/usr/bin/env bash
# Start the FastAPI backend and the Streamlit dashboard together. Ctrl+C stops both.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-.venv/bin/python}
"$PY" -m uvicorn main:app --host "${API_HOST:-127.0.0.1}" --port "${API_PORT:-8000}" &
API_PID=$!
trap 'kill $API_PID 2>/dev/null' EXIT
"$PY" -m streamlit run app.py --server.port "${UI_PORT:-8501}"
