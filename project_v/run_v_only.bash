#!/bin/bash

# Helper wrapper to rerun main.bash only for videos that start with "V"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VIDEO_FILTER_REGEX='^V'
bash "$SCRIPT_DIR/main.bash" "$@"
