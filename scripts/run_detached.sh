#!/usr/bin/env bash
# Launches `lrmc run-queue` detached from the current terminal/AnyDesk
# session, so it survives a disconnect. Two mechanisms, in preference order:
#   1. tmux (if installed): a named session "lrmc-queue" you can reattach to
#      and watch live -- preferred.
#   2. nohup + setsid: fully detached, no live reattach, output goes to a
#      log file (tail it, or use `python -m lrmc status`).
#
# Usage: scripts/run_detached.sh [path/to/queue.yaml]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

QUEUE="${1:-experiments/queue.yaml}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_queue_$(date +%Y%m%d_%H%M%S).log"

PYTHON_BIN="python"
if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
fi

CMD="$PYTHON_BIN -m lrmc.cli.main run-queue \"$QUEUE\""

if command -v tmux >/dev/null 2>&1; then
    SESSION="lrmc-queue"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "tmux session '$SESSION' already exists -- reattach with: tmux attach -t $SESSION"
        exit 0
    fi
    echo "Starting in tmux session '$SESSION' (reattach any time: tmux attach -t $SESSION)"
    tmux new-session -d -s "$SESSION" "$CMD 2>&1 | tee -a \"$LOG_FILE\""
    echo "Log also written to $LOG_FILE"
else
    echo "tmux not found -- falling back to nohup (fully detached, no live reattach)."
    echo "Log: $LOG_FILE"
    nohup setsid bash -c "$CMD" >>"$LOG_FILE" 2>&1 </dev/null &
    echo "Started with PID $!. Check progress with: $PYTHON_BIN -m lrmc.cli.main status"
fi
