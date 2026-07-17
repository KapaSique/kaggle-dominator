#!/usr/bin/env bash
set -euo pipefail

exec python3 scripts/evolution.py gate "$@"
