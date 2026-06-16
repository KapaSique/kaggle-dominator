#!/usr/bin/env bash
# kaggle_eval_loop.sh — headless eval-driven loop for a Kaggle competition.
#
# The agent proposes ONE new experiment per round, runs it on a Kaggle kernel,
# reads the real metric, logs it, and updates BEST_KNOWN only on a confirmed gain.
# It NEVER submits — it prepares candidates for a human to confirm.
#
# Prereqs in the working dir:
#   - BEST_KNOWN.txt   : current best real score (one number), e.g. 0.97106
#   - results.csv      : header `slug,model,features,params,cv,lb` (loop appends rows)
#   - kaggle CLI authenticated (~/.kaggle/kaggle.json), claude CLI installed
#
# Usage: ./kaggle_eval_loop.sh <competition-slug> [target] [rounds]
# Kill switch: `touch STOP` in the working dir to halt after the current round.
set -euo pipefail

COMP="${1:?usage: kaggle_eval_loop.sh <competition-slug> [target] [rounds]}"
TARGET="${2:-1.0}"          # stop early if BEST reaches this
ROUNDS="${3:-8}"            # hard cap on iterations (budget guard)
BEST_FILE="BEST_KNOWN.txt"
[ -f "$BEST_FILE" ] || { echo "0" > "$BEST_FILE"; }
[ -f results.csv ] || echo "slug,model,features,params,cv,lb" > results.csv

for i in $(seq 1 "$ROUNDS"); do
  [ -f STOP ] && { echo "[loop] STOP flag found — halting"; rm -f STOP; break; }
  BEST=$(cat "$BEST_FILE")
  echo "===== round $i/$ROUNDS — BEST_KNOWN=$BEST — $(date) ====="

  # The constitution lives in the kaggle-dominator skill; the agent reads it on trigger.
  claude -p "Competition: $COMP. BEST_KNOWN=$BEST (real metric, do NOT regress).
Read results.csv for what's already been tried. Propose ONE experiment that adds DIVERSITY
not yet present (new feature set / model nature / fold scheme / pseudo-label / external OOF).
Build the kernel, push it under my Kaggle account, poll to completion, read the out-of-fold
score on the competition metric, and append exactly one row to results.csv.
Do NOT submit anything. End your reply with: SCORE=<number>." \
    --allowedTools "Bash Read Write Edit Grep Glob" || { echo "[loop] round $i failed, continuing"; continue; }

  NEW=$(tail -1 results.csv | awk -F, '{print $NF}')
  if awk "BEGIN{exit !($NEW > $BEST)}"; then
    echo "$NEW" > "$BEST_FILE"
    echo "[loop] NEW BEST_KNOWN=$NEW (round $i)"
  else
    echo "[loop] no gain (round $i: $NEW <= $BEST) — BEST_KNOWN held"
  fi
  if awk "BEGIN{exit !($(cat $BEST_FILE) >= $TARGET)}"; then
    echo "[loop] target $TARGET reached — stopping"; break
  fi
done

echo "===== loop done — final BEST_KNOWN=$(cat $BEST_FILE) ====="
echo "Review results.csv, pick the candidate, and submit manually:"
echo "  kaggle competitions submit -c $COMP -f <file> -m '<version + hypothesis>'"
