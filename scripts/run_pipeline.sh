#!/usr/bin/env bash
set -euo pipefail

PY=${1:-/usr/bin/python3}

"$PY" src/run_all.py --python "$PY" --with_asr
