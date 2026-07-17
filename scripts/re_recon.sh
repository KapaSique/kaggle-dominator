#!/usr/bin/env bash
# Retired legacy entry point. The trusted Codex daily orchestration owns recon.
set -euo pipefail

printf '%s\n' \
  're_recon.sh is retired and performs no action.' \
  'Use the trusted Codex daily orchestration with scripts/evolution.py for local evidence.' >&2
exit 2
