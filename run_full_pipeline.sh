#!/usr/bin/env bash
# PAE_katana - Full Experiment Pipeline
# Runs V0 baseline -> Evolution -> V1 evolved on all domains

set -euo pipefail

# ============================================
# CONFIGURATION
# ============================================

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load API key from .env file
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  if [[ -f "$SCRIPT_DIR/.env" ]]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
    echo "✓ Loaded OPENAI_API_KEY from .env"
  else
    echo "ERROR: OPENAI_API_KEY not set and .env file not found."
    echo "Create .env file with: OPENAI_API_KEY=your-key"
    exit 1
  fi
fi

# Task configuration
TASK_ROOT="${TASK_ROOT:-$SCRIPT_DIR/data/problem_splits/classic_3000}"
DOMAINS="${DOMAINS:-amazon yelp goodreads}"
MAX_WORKERS="${MAX_WORKERS:-8}"

# Which phases to run
RUN_DEV_TRAIN="${RUN_DEV_TRAIN:-1}"   # V0/V1 on dev train
RUN_TRACK2="${RUN_TRACK2:-1}"          # Final evaluation on track2

# Environment setup
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pae_katana

echo "=============================================="
echo "PAE_katana Full Pipeline"
echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "TASK_ROOT=$TASK_ROOT"
echo "DOMAINS=$DOMAINS"
echo "MAX_WORKERS=$MAX_WORKERS"
echo "RUN_DEV_TRAIN=$RUN_DEV_TRAIN RUN_TRACK2=$RUN_TRACK2"
echo "=============================================="

# ============================================
# PHASE 1: V0 Baseline (Seed Skills Only)
# ============================================
if [[ "$RUN_DEV_TRAIN" == "1" ]]; then
    echo ""
    echo "=============================================="
    echo "PHASE 1: V0 Baseline on Dev Train"
    echo "=============================================="

    for domain in $DOMAINS; do
        echo ""
        echo "[V0/$domain] Starting..."

        ENABLE_EVOLUTION="false" python run_experiment.py \
            --task_dir "$TASK_ROOT/train/$domain/tasks" \
            --groundtruth_dir "$TASK_ROOT/train/$domain/groundtruth" \
            --task_set "$domain" \
            --version "v0_devtrain" \
            --method_name "proposal_v0" \
            --max_workers "$MAX_WORKERS" \
            --output_dir "results/proposal_v0"

        echo "[V0/$domain] Done"
    done
fi

# ============================================
# PHASE 2: Skill Evolution
# ============================================
echo ""
echo "=============================================="
echo "PHASE 2: Skill Evolution from V0 Failures"
echo "=============================================="

for domain in $DOMAINS; do
    echo "[Evolution/$domain] Analyzing failures and evolving skills..."
    python scripts/evolution_engine.py \
        --version "v0_devtrain" \
        --domain "$domain" \
        --output_dir "src"
done

# ============================================
# PHASE 3: V1 Evolved (With Evolved Skills)
# ============================================
if [[ "$RUN_DEV_TRAIN" == "1" ]]; then
    echo ""
    echo "=============================================="
    echo "PHASE 3: V1 Evolved on Dev Train"
    echo "=============================================="

    for domain in $DOMAINS; do
        echo ""
        echo "[V1/$domain] Starting with evolved skills..."

        ENABLE_EVOLUTION="true" python run_experiment.py \
            --task_dir "$TASK_ROOT/train/$domain/tasks" \
            --groundtruth_dir "$TASK_ROOT/train/$domain/groundtruth" \
            --task_set "$domain" \
            --version "v1_devtrain" \
            --method_name "proposal_v1" \
            --max_workers "$MAX_WORKERS" \
            --output_dir "results/proposal_v1"

        echo "[V1/$domain] Done"
    done
fi

# ============================================
# PHASE 4: Track2 Final Evaluation
# ============================================
if [[ "$RUN_TRACK2" == "1" ]]; then
    echo ""
    echo "=============================================="
    echo "PHASE 4: Track2 Final Evaluation"
    echo "=============================================="

    for domain in $DOMAINS; do
        echo ""
        echo "[Track2/$domain] Final evaluation..."

        ENABLE_EVOLUTION="true" python run_experiment.py \
            --task_dir "tasks/track2/$domain/tasks" \
            --groundtruth_dir "tasks/track2/$domain/groundtruth" \
            --task_set "$domain" \
            --version "track2_final" \
            --method_name "proposal_final" \
            --max_workers "$MAX_WORKERS" \
            --output_dir "results/proposal_final"

        echo "[Track2/$domain] Done"
    done
fi

# ============================================
# SUMMARY
# ============================================
echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  - results/proposal_v0/    (V0 baseline)"
echo "  - results/proposal_v1/    (V1 evolved)"
echo "  - results/proposal_final/ (Track2 final)"
echo ""
echo "To analyze results:"
echo "  python scripts/summarize_v0_v1_results.py"
