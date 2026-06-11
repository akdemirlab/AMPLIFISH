#!/usr/bin/env bash
# Thin wrapper: run the AMPLIFISH demo with the active Python environment.
# Assumes `pip install -e .` has been run (see README).
set -euo pipefail
cd "$(dirname "$0")"
python run_demo.py "$@"
