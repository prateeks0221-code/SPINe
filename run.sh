#!/bin/bash
# CV Spine — VibeCheck Dashboard launcher
# Handles MKL fix + graceful shutdown of all child processes

set -e

# ── MKL/Torch fix ──
export LD_PRELOAD=/home/lenovo/miniconda3/envs/spine/lib/python3.13/site-packages/torch/lib/libtorch_cpu.so
export MKL_SERVICE_FORCE_INTEL=1

# ── Change to project root ──
cd "$(dirname "$0")"

# ── Track PIDs for cleanup ──
PIDS=()

cleanup() {
    echo ""
    echo "[CV-SPINE] Shutting down..."

    # Kill main python process and all children
    if [ -n "$MAIN_PID" ]; then
        kill -- -$MAIN_PID 2>/dev/null || true
    fi

    # Kill any remaining affiliated processes
    for pid in "${PIDS[@]}"; do
        kill $pid 2>/dev/null || true
    done

    # Kill any lingering processes on port 8765
    fuser -k 8765/tcp 2>/dev/null || true

    # Kill any orphaned YOLO/torch processes from this session
    pkill -f "products.vibecheck.dashboard.app" 2>/dev/null || true

    echo "[CV-SPINE] All processes terminated."
    exit 0
}

# ── Trap all exit signals ──
trap cleanup SIGINT SIGTERM EXIT

echo "══════════════════════════════════════════════"
echo "  CV SPINE — VibeCheck Dashboard"
echo "  http://localhost:8765"
echo "  Press Ctrl+C to stop all processes"
echo "══════════════════════════════════════════════"

# ── Kill any previous instance on port 8765 ──
fuser -k 8765/tcp 2>/dev/null || true
sleep 0.5

# ── Launch dashboard (in own process group) ──
setsid python -m products.vibecheck.dashboard.app &
MAIN_PID=$!
PIDS+=($MAIN_PID)

echo "[CV-SPINE] Started (PID: $MAIN_PID)"

# ── Wait for main process ──
wait $MAIN_PID
