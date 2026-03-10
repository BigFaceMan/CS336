#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

CONFIG_PATH="config/train_full_story.yaml"
IMPL_NAME="baseline"
OUTPUT_ROOT="output/lr_sweep"
LRS_CSV="1e-4,3e-4,6e-4,1e-3"
DEVICES_CSV="cuda:0"
DRY_RUN=0
MAX_PARALLEL=""

usage() {
  cat <<'EOF'
Usage:
  bash script/lr_sweep.sh [options]

Options:
  --config <path>         YAML config path. Default: config/train_full_story.yaml
  --impl-name <name>      Implementation/experiment prefix. Default: baseline
  --output-root <path>    Root output directory. Default: output/lr_sweep
  --lrs <csv>             Comma-separated learning rates. Default: 1e-4,3e-4,6e-4,1e-3
  --devices <csv>         Comma-separated devices. Default: cuda:0
                          If only one device is given, it is reused for all lrs.
                          Otherwise, device count must equal lr count.
  --dry-run               Print commands only, do not execute.
  --max-parallel <n>      Max concurrent runs. Default: device count (or 1 if single device).
  -h, --help              Show this help.

Examples:
  bash script/lr_sweep.sh \
    --impl-name my_impl \
    --lrs 1e-4,3e-4,6e-4 \
    --devices cuda:0,cuda:1,cuda:0

  bash script/lr_sweep.sh \
    --config config/train_small_story.yaml \
    --impl-name exp_v2 \
    --lrs 5e-5,1e-4 \
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
    --lrs)
      LRS_CSV="$2"
      shift 2
      ;;
    --devices)
      DEVICES_CSV="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --max-parallel)
      MAX_PARALLEL="$2"
      shift 2
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

IFS=',' read -r -a LRS <<< "$LRS_CSV"
IFS=',' read -r -a DEVICES <<< "$DEVICES_CSV"

if [[ ${#LRS[@]} -eq 0 ]]; then
  echo "No learning rates provided."
  exit 1
fi

if [[ ${#DEVICES[@]} -ne 1 && ${#DEVICES[@]} -ne ${#LRS[@]} ]]; then
  echo "Device count must be 1 or equal to lr count."
  echo "lr count: ${#LRS[@]}, device count: ${#DEVICES[@]}"
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

for i in "${!LRS[@]}"; do
  lr="${LRS[$i]}"
  if [[ ${#DEVICES[@]} -eq 1 ]]; then
    device="${DEVICES[0]}"
  else
    device="${DEVICES[$i]}"
  fi

  lr_tag="$(sanitize "$lr")"
  device_tag="$(sanitize "$device")"
  exp_name="${IMPL_NAME}_lr_${lr_tag}_dev_${device_tag}"
  out_dir="${OUTPUT_ROOT}/${IMPL_NAME}/${exp_name}"

  cmd=(
    python cs336_basics/train.py
    --config "$CONFIG_PATH"
    --lr "$lr"
    --device "$device"
    --exp_name "$exp_name"
    --output_path "$out_dir"
  )

  echo "[run $((i + 1))/${#LRS[@]}] lr=$lr device=$device exp=$exp_name"
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


# bash script/lr_sweep.sh --impl-name my_impl --lrs 1e-4,3e-4 --devices cuda:0,cuda:1 --dry-run
