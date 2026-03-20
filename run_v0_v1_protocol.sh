#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
ROOT_DIR="$PROJECT_DIR"  # PAE_katana structure: root = project dir
TASK_ROOT="${TASK_ROOT:-$PROJECT_DIR/data/problem_splits/classic_3000}"

ACTIVE_SKILLS="$PROJECT_DIR/src/skills_db.json"
SEED_SKILLS="${SEED_SKILLS:-$PROJECT_DIR/src/skills_db_v0_universal_seed.json}"
BACKUP_SKILLS="$PROJECT_DIR/src/skills_db.pre_protocol_backup.json"

TRACK="${TRACK:-cognitive}"               # gradual | cognitive | both
DOMAINS_STR="${DOMAINS:-amazon yelp goodreads}"
MAX_WORKERS="${MAX_WORKERS:-1}"
RUN_DEV_TRAIN="${RUN_DEV_TRAIN:-1}"
RUN_DEV_VAL="${RUN_DEV_VAL:-1}"
RUN_TRACK2="${RUN_TRACK2:-1}"
NUM_TASKS_DEV_TRAIN="${NUM_TASKS_DEV_TRAIN:-999999}"
NUM_TASKS_DEV_VAL="${NUM_TASKS_DEV_VAL:-999999}"

read -r -a DOMAINS <<< "$DOMAINS_STR"

if [[ ! -f "$SEED_SKILLS" ]]; then
  echo "❌ Seed skills file not found: $SEED_SKILLS" >&2
  exit 1
fi

if [[ ! -f "$ACTIVE_SKILLS" ]]; then
  echo "❌ Active skills file not found: $ACTIVE_SKILLS" >&2
  exit 1
fi

cp "$ACTIVE_SKILLS" "$BACKUP_SKILLS"
cleanup() {
  if [[ -f "$BACKUP_SKILLS" ]]; then
    cp "$BACKUP_SKILLS" "$ACTIVE_SKILLS"
  fi
}
trap cleanup EXIT

echo "=============================================="
echo "🚀 PersonalRecAgent V0→V1 Validation Protocol"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "TRACK=$TRACK"
echo "DOMAINS=${DOMAINS[*]}"
echo "TASK_ROOT=$TASK_ROOT"
echo "MAX_WORKERS=$MAX_WORKERS"
echo "RUN_DEV_TRAIN=$RUN_DEV_TRAIN RUN_DEV_VAL=$RUN_DEV_VAL RUN_TRACK2=$RUN_TRACK2"
echo "=============================================="

# Load API key from .env file if not already set
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  if [[ -f "$PROJECT_DIR/.env" ]]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
    echo "✓ Loaded OPENAI_API_KEY from .env"
  else
    echo "⚠️  OPENAI_API_KEY is not set and .env file not found."
    echo "   Create .env file with: OPENAI_API_KEY=your-key"
    exit 1
  fi
fi
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export KMP_INIT_AT_FORK="${KMP_INIT_AT_FORK:-FALSE}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

count_task_files() {
  local task_dir="$1"
  if [[ ! -d "$task_dir" ]]; then
    echo 0
    return
  fi
  find "$task_dir" -maxdepth 1 -type f -name 'task_*.json' | wc -l | tr -d ' '
}

resolve_task_count() {
  local task_dir="$1"
  local requested="$2"
  local available
  available="$(count_task_files "$task_dir")"

  if [[ "$available" -eq 0 ]]; then
    echo 0
  elif [[ -z "$requested" || "$requested" -le 0 || "$requested" -ge "$available" ]]; then
    echo "$available"
  else
    echo "$requested"
  fi
}

run_experiment_custom() {
  local domain="$1"
  local task_dir="$2"
  local gt_dir="$3"
  local version="$4"
  local method_name="$5"
  local enable_evolution="$6"
  local use_cognitive_guidance="$7"
  local num_tasks="$8"

  echo "   → Running ${method_name} on ${domain} (${version})"
  ENABLE_EVOLUTION="$enable_evolution" \
  USE_COGNITIVE_GUIDANCE="$use_cognitive_guidance" \
  python "$PROJECT_DIR/run_experiment.py" \
    --task_set "$domain" \
    --task_dir "$task_dir" \
    --gt_dir "$gt_dir" \
    --num_tasks "$num_tasks" \
    --version "$version" \
    --method_name "$method_name" \
    --max_workers "$MAX_WORKERS"
}

run_experiment_track2() {
  local domain="$1"
  local version="$2"
  local method_name="$3"
  local enable_evolution="$4"
  local use_cognitive_guidance="$5"

  local num_tasks
  case "$domain" in
    amazon) num_tasks=388 ;;
    yelp) num_tasks=384 ;;
    goodreads) num_tasks=400 ;;
    *)
      echo "❌ Unknown domain: $domain" >&2
      exit 1
      ;;
  esac

  echo "   → Running ${method_name} on official track2 ${domain} (${version})"
  ENABLE_EVOLUTION="$enable_evolution" \
  USE_COGNITIVE_GUIDANCE="$use_cognitive_guidance" \
  python "$PROJECT_DIR/run_experiment.py" \
    --task_set "$domain" \
    --num_tasks "$num_tasks" \
    --version "$version" \
    --method_name "$method_name" \
    --max_workers "$MAX_WORKERS"
}

evolve_domain() {
  local domain="$1"
  local failed_cases="$2"
  local output_skills="$3"
  local use_gradual_prompt="$4"

  if [[ ! -f "$failed_cases" ]]; then
    echo "❌ Failed cases file not found: $failed_cases" >&2
    exit 1
  fi

  echo "   → Evolving skills for ${domain}"
  USE_GRADUAL_PROMPT="$use_gradual_prompt" \
  python "$PROJECT_DIR/scripts/evolution_engine.py" \
    --domain "$domain" \
    --failed_cases "$failed_cases" \
    --output "$output_skills"

  if [[ ! -f "$output_skills" ]]; then
    echo "❌ Evolution did not produce: $output_skills" >&2
    exit 1
  fi
}

print_result_locations() {
  local domain="$1"
  local v0_method="$2"
  local v0_version="$3"
  local v1_method="$4"
  local v1_version="$5"

  echo "      V0: $ROOT_DIR/results/$v0_method/${domain}_${v0_version}_*_metrics.json"
  echo "      V1: $ROOT_DIR/results/$v1_method/${domain}_${v1_version}_*_metrics.json"
}

run_track() {
  local track_name="$1"

  local v0_prefix=""
  local v1_prefix=""
  local use_cognitive_guidance=""
  local use_gradual_prompt=""
  local skill_suffix=""

  case "$track_name" in
    gradual)
      v0_prefix="v0a"
      v1_prefix="v1a"
      use_cognitive_guidance="false"
      use_gradual_prompt="true"
      skill_suffix="gradual"
      ;;
    cognitive)
      v0_prefix="v0b"
      v1_prefix="v1b"
      use_cognitive_guidance="true"
      use_gradual_prompt="false"
      skill_suffix="cognitive"
      ;;
    *)
      echo "❌ Unknown track: $track_name" >&2
      exit 1
      ;;
  esac

  echo
  echo "=============================================="
  echo "🧪 Running track: $track_name"
  echo "=============================================="

  for domain in "${DOMAINS[@]}"; do
    echo
    echo "[$track_name/$domain]"

    local dev_train_tasks="$TASK_ROOT/train/$domain/tasks"
    local dev_train_gt="$TASK_ROOT/train/$domain/groundtruth"
    local dev_val_tasks="$TASK_ROOT/val/$domain/tasks"
    local dev_val_gt="$TASK_ROOT/val/$domain/groundtruth"
    local evolved_skills="$PROJECT_DIR/src/skills_db_${v1_prefix}_${skill_suffix}_${domain}.json"
    local failed_cases="$PROJECT_DIR/data/failed_cases_${v0_prefix}_devtrain_${domain}.json"

    if [[ "$RUN_DEV_TRAIN" == "1" ]]; then
      echo " Phase 1: V0 on dev train"
      local dev_train_count
      dev_train_count="$(resolve_task_count "$dev_train_tasks" "$NUM_TASKS_DEV_TRAIN")"
      if [[ "$dev_train_count" -eq 0 ]]; then
        echo "   ⚠️ No dev-train tasks found for $domain, skipping domain"
        continue
      fi
      cp "$SEED_SKILLS" "$ACTIVE_SKILLS"
      run_experiment_custom \
        "$domain" \
        "$dev_train_tasks" \
        "$dev_train_gt" \
        "${v0_prefix}_devtrain" \
        "${v0_prefix}_devtrain_baseline" \
        "false" \
        "$use_cognitive_guidance" \
        "$dev_train_count"
    fi

    echo " Phase 2: Evolve V1 from dev-train failures"
    evolve_domain "$domain" "$failed_cases" "$evolved_skills" "$use_gradual_prompt"

    if [[ "$RUN_DEV_VAL" == "1" ]]; then
      local dev_val_count
      dev_val_count="$(resolve_task_count "$dev_val_tasks" "$NUM_TASKS_DEV_VAL")"
      if [[ "$dev_val_count" -eq 0 ]]; then
        echo " Phase 3: dev val is empty for $domain, skipping"
      else
      echo " Phase 3: V0 vs V1 on dev val"

      cp "$SEED_SKILLS" "$ACTIVE_SKILLS"
      run_experiment_custom \
        "$domain" \
        "$dev_val_tasks" \
        "$dev_val_gt" \
        "${v0_prefix}_devval" \
        "${v0_prefix}_devval_baseline" \
        "false" \
        "$use_cognitive_guidance" \
        "$dev_val_count"

      cp "$evolved_skills" "$ACTIVE_SKILLS"
      run_experiment_custom \
        "$domain" \
        "$dev_val_tasks" \
        "$dev_val_gt" \
        "${v1_prefix}_devval" \
        "${v1_prefix}_devval_evolved" \
        "true" \
        "$use_cognitive_guidance" \
        "$dev_val_count"

      echo "   Dev-val comparison files:"
      print_result_locations \
        "$domain" \
        "${v0_prefix}_devval_baseline" \
        "${v0_prefix}_devval" \
        "${v1_prefix}_devval_evolved" \
        "${v1_prefix}_devval"
      fi
    fi

    if [[ "$RUN_TRACK2" == "1" ]]; then
      echo " Phase 4: V0 vs V1 on official track2"

      cp "$SEED_SKILLS" "$ACTIVE_SKILLS"
      run_experiment_track2 \
        "$domain" \
        "${v0_prefix}_track2" \
        "${v0_prefix}_track2_baseline" \
        "false" \
        "$use_cognitive_guidance"

      cp "$evolved_skills" "$ACTIVE_SKILLS"
      run_experiment_track2 \
        "$domain" \
        "${v1_prefix}_track2" \
        "${v1_prefix}_track2_evolved" \
        "true" \
        "$use_cognitive_guidance"

      echo "   Official track2 comparison files:"
      print_result_locations \
        "$domain" \
        "${v0_prefix}_track2_baseline" \
        "${v0_prefix}_track2" \
        "${v1_prefix}_track2_evolved" \
        "${v1_prefix}_track2"
    fi
  done
}

case "$TRACK" in
  gradual)
    run_track gradual
    ;;
  cognitive)
    run_track cognitive
    ;;
  both)
    run_track gradual
    run_track cognitive
    ;;
  *)
    echo "❌ TRACK must be one of: gradual | cognitive | both" >&2
    exit 1
    ;;
esac

echo
echo "✅ Protocol completed."
echo "   Read: $PROJECT_DIR/V0_V1_VALIDATION_PROTOCOL.md"
echo "   Results root: $ROOT_DIR/results"
