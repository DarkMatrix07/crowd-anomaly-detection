#!/usr/bin/env bash
set -euo pipefail

# Always run from the script's own directory
cd "$(dirname "$0")"

# ── colours ────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ── keep window open on any exit ────────────────────────────────────────────
trap 'echo; read -rp "Press Enter to close..." _' EXIT

echo "============================================================"
echo "  Abnormal Crowd Behaviour Detection System"
echo "  Demo Launcher"
echo "============================================================"
echo

# ── 1. Python ───────────────────────────────────────────────────────────────
if ! command -v python &>/dev/null; then
    error "Python not found in PATH."
    error "Install Python 3.11+ from https://python.org"
    exit 1
fi
info "Python: $(python --version)"

# ── 2. Dependencies ─────────────────────────────────────────────────────────
info "Checking dependencies..."
if ! python -c "import streamlit, cv2, joblib, sklearn" &>/dev/null; then
    warn "Some packages missing — installing from requirements.txt..."
    pip install -r requirements.txt || { error "pip install failed."; exit 1; }
    info "Dependencies installed."
fi
info "Dependencies OK."

# ── 3. Model artifact ───────────────────────────────────────────────────────
MODEL="artifacts/models/shanghaitech_windowed_rf.joblib"
if [[ ! -f "$MODEL" ]]; then
    error "Model file not found: $MODEL"
    error "Make sure the full project folder was copied correctly."
    exit 1
fi
info "Model found."

# ── 4. Demo data ─────────────────────────────────────────────────────────────
DEMO_CLIP="data/demo/frames/01_0130"
if [[ ! -d "$DEMO_CLIP" || -z "$(ls -A "$DEMO_CLIP" 2>/dev/null)" ]]; then
    warn "Demo clips not found — running setup..."
    python scripts/setup_demo_data.py
    if [[ $? -ne 0 ]]; then
        warn "Demo data setup failed (dataset may not be present)."
        warn "You can still use 'Upload Video' mode in the demo app."
    fi
else
    info "Demo clips found."
fi

# ── 5. Launch ───────────────────────────────────────────────────────────────
echo
info "Starting demo — browser will open automatically..."
echo "  To stop: press Ctrl+C in this window"
echo "============================================================"
echo

python -m streamlit run scripts/crowd_anomaly_demo.py

info "Demo stopped."
