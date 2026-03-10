#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

CONFIG_PATH="config/train_full_story.yaml"
IMPL_NAME="baseline"
OUTPUT_ROOT="output/batch_size_sweep"
BATCH_SIZES_CSV="8,16,32,64"
DEVICES_CSV="cuda:0"
DRY_RUN=0
MAX_PARALLEL=""

usage() {
  cat <<'EOF'
Usage:
  bash script/batch_size_sweep.sh [options]

Options:
  --config <path>            YAML config path. Default: config/train_full_story.yaml
  --impl-name <name>         Implementation/experiment prefix. Default: baseline
  --output-root <path>       Root output directory. Default: output/batch_size_sweep
  --batch-sizes <csv>        Comma-separated batch sizes. Default: 8,16,32,64
  --devices <csv>            Comma-separated devices. Default: cuda:0
                             If only one device is given, it is reused for all batch sizes.
                             Otherwise, device count must equal batch-size count.
  --max-parallel <n>         Max concurrent runs. Default: device count (or 1 if single device).
  --dry-run                  Print commands only, do not execute.
  -h, --help                 Show this help.

Examples:
  bash script/batch_size_sweep.sh \
    --impl-name my_impl \
    --batch-sizes 8,16,32 \
    --devices cuda:0,cuda:1,cuda:0 \
    --max-parallel 2

  bash script/batch_size_sweep.sh \
    --config config/train_small_story.yaml \
    --impl-name exp_v2 \
    --batch-sizes 4,8 \
    --devices cuda:1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --impl-name)
      IMPL_NAME="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --batch-sizes)
      BATCH_SIZES_CSV="$2"
      shift 2
      ;;
    --devices)
      DEVICES_CSV="$2"
      shift 2
      ;;
    --max-parallel)
      MAX_PARALLEL="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

IFS=',' read -r -a BATCH_SIZES <<< "$BATCH_SIZES_CSV"
IFS=',' read -r -a DEVICES <<< "$DEVICES_CSV"

if [[ ${#BATCH_SIZES[@]} -eq 0 ]]; then
  echo "No batch sizes provided."
  exit 1
fi

for bs in "${BATCH_SIZES[@]}"; do
  if ! [[ "$bs" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid batch size: $bs. Batch sizes must be positive integers."
    exit 1
  fi
done

if [[ ${#DEVICES[@]} -ne 1 && ${#DEVICES[@]} -ne ${#BATCH_SIZES[@]} ]]; then
  echo "Device count must be 1 or equal to batch-size count."
  echo "batch-size count: ${#BATCH_SIZES[@]}, device count: ${#DEVICES[@]}"
  exit 1
fi

if [[ -z "$MAX_PARALLEL" ]]; then
  MAX_PARALLEL="${#DEVICES[@]}"
fi

if ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-parallel must be a positive integer."
  exit 1
fi

sanitize() {
  echo "$1" | tr ':/.' '___'
}

mkdir -p "$OUTPUT_ROOT/$IMPL_NAME"

active_jobs=0
FAIL=0

for i in "${!BATCH_SIZES[@]}"; do
  batch_size="${BATCH_SIZES[$i]}"
  if [[ ${#DEVICES[@]} -eq 1 ]]; then
    device="${DEVICES[0]}"
  else
    device="${DEVICES[$i]}"
  fi

  bs_tag="$(sanitize "$batch_size")"
  device_tag="$(sanitize "$device")"
  exp_name="${IMPL_NAME}_bs_${bs_tag}_dev_${device_tag}"
  out_dir="${OUTPUT_ROOT}/${IMPL_NAME}/${exp_name}"

  cmd=(
    python cs336_basics/train.py
    --config "$CONFIG_PATH"
    --batch_size "$batch_size"
    --device "$device"
    --exp_name "$exp_name"
    --output_path "$out_dir"
  )

  echo "[run $((i + 1))/${#BATCH_SIZES[@]}] batch_size=$batch_size device=$device exp=$exp_name"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '  %q ' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}" &
    active_jobs=$((active_jobs + 1))

    if [[ "$active_jobs" -ge "$MAX_PARALLEL" ]]; then
      wait -n || FAIL=1
      active_jobs=$((active_jobs - 1))
    fi
  fi
done

if [[ "$DRY_RUN" -eq 0 ]]; then
  wait || FAIL=1
fi

if [[ "$FAIL" -ne 0 ]]; then
  echo "Sweep finished with failures."
  exit 1
fi

echo "Sweep finished successfully."
