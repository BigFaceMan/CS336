#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

CONFIG_PATH="config/infer.yaml"
uv run python cs336_basics/inference.py --config $CONFIG_PATH
