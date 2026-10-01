#!/usr/bin/env bash
# ==============================================================================
# Smart Traffic Management System — Cloud Production Startup Script
# ==============================================================================
set -e

PORT="${PORT:-5000}"
HOST="${HOST:-0.0.0.0}"
THREADS="${WAITRESS_THREADS:-4}"

echo "======================================================="
echo " SMART TRAFFIC MANAGEMENT SYSTEM — PRODUCTION CLOUD"
echo " Host: ${HOST} | Port: ${PORT} | Threads: ${THREADS}"
echo "======================================================="

exec python -m waitress --host="${HOST}" --port="${PORT}" --threads="${THREADS}" app:app
