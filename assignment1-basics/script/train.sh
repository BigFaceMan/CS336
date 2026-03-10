#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

CONFIG_PATH="config/train_full_story.yaml"
python cs336_basics/train.py --config $CONFIG_PATH

