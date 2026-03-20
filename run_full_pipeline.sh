#!/usr/bin/env bash
# PAE_katana - Full Experiment Pipeline
# Flow: V0 train -> Evolution -> V0 test -> V1 test

set -euo pipefail

# ============================================
# CONFIGURATION
# ============================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load API key from .env file
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  if [[ -f "$SCRIPT_DIR/.env" ]]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
    echo "Loaded OPENAI_API_KEY from .env"
  else
    echo "ERROR: OPENAI_API_KEY not set and .env file not found."
    exit 1
  fi
fi

# Activate environment (Katana uses module + venv)
module load python/3.10.8 2>/dev/null || true
source "$SCRIPT_DIR/venv/bin/activate"

# Configuration
TASK_ROOT="${TASK_ROOT:-$SCRIPT_DIR/data/problem_splits/classic_3000}"
DOMAINS="${DOMAINS:-amazon yelp goodreads}"
MAX_WORKERS="${MAX_WORKERS:-4}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "=============================================="
echo "PAE_katana Full Pipeline"
echo "TASK_ROOT=$TASK_ROOT"
echo "DOMAINS=$DOMAINS"
echo "MAX_WORKERS=$MAX_WORKERS"
echo "=============================================="

# ============================================
# PHASE 1: V0 Baseline on Train Set
# ============================================
echo ""
echo "=============================================="
echo "PHASE 1: V0 Baseline on Train Set"
echo "=============================================="

for domain in $DOMAINS; do
    echo ""
    echo "[V0-Train/$domain] Starting at $(date)..."

    ENABLE_EVOLUTION="false" python run_experiment.py \
        --task_dir "$TASK_ROOT/train/$domain/tasks" \
        --gt_dir "$TASK_ROOT/train/$domain/groundtruth" \
        --task_set "$domain" \
        --version "v0_train" \
        --method_name "proposal_v0" \
        --max_workers "$MAX_WORKERS" \
        --output_dir "results/proposal_v0"

    echo "[V0-Train/$domain] Done at $(date)"
done

# ============================================
# PHASE 2: Skill Evolution
# ============================================
echo ""
echo "=============================================="
echo "PHASE 2: Skill Evolution from V0 Failures"
echo "=============================================="

for domain in $DOMAINS; do
    echo "[Evolution/$domain] Starting..."
    python scripts/evolution_engine.py \
        --version "v0_train" \
        --domain "$domain" \
        --output_dir "src"
    echo "[Evolution/$domain] Done"
done

# ============================================
# PHASE 3: V0 on Test Set (Track2)
# ============================================
echo ""
echo "=============================================="
echo "PHASE 3: V0 on Test Set (Track2)"
echo "=============================================="

for domain in $DOMAINS; do
    echo ""
    echo "[V0-Test/$domain] Starting at $(date)..."

    ENABLE_EVOLUTION="false" python run_experiment.py \
        --task_dir "tasks/track2/$domain/tasks" \
        --gt_dir "tasks/track2/$domain/groundtruth" \
        --task_set "$domain" \
        --version "track2_v0" \
        --method_name "proposal_track2_v0" \
        --max_workers "$MAX_WORKERS" \
        --output_dir "results/proposal_track2_v0"

    echo "[V0-Test/$domain] Done at $(date)"
done

# ============================================
# PHASE 4: V1 on Test Set (Track2)
# ============================================
echo ""
echo "=============================================="
echo "PHASE 4: V1 on Test Set (Track2)"
echo "=============================================="

for domain in $DOMAINS; do
    echo ""
    echo "[V1-Test/$domain] Starting at $(date)..."

    ENABLE_EVOLUTION="true" python run_experiment.py \
        --task_dir "tasks/track2/$domain/tasks" \
        --gt_dir "tasks/track2/$domain/groundtruth" \
        --task_set "$domain" \
        --version "track2_v1" \
        --method_name "proposal_track2_v1" \
        --max_workers "$MAX_WORKERS" \
        --output_dir "results/proposal_track2_v1"

    echo "[V1-Test/$domain] Done at $(date)"
done

# ============================================
# Summary
# ============================================
echo ""
echo "=============================================="
echo "Pipeline Completed at $(date)"
echo "=============================================="
echo ""
echo "Results:"
echo "  V0 Test: results/proposal_track2_v0/"
echo "  V1 Test: results/proposal_track2_v1/"
ls -la results/ 2>/dev/null || true
