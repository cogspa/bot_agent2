#!/bin/bash

# Configuration
BRIDGE_DIR="/Users/joem/.gemini/antigravity/scratch/blender_bridge"
PAYLOAD_FILE="$BRIDGE_DIR/payload.py"

# Help
if [ -z "$1" ]; then
    echo "Usage: ./run_in_blender.sh <script_to_run.py>"
    exit 1
fi

SOURCE_FILE="$1"

# Check if source exists
if [ ! -f "$SOURCE_FILE" ]; then
    echo "Error: File '$SOURCE_FILE' not found."
    exit 1
fi

# Copy content to payload
# We used 'cat' > file to ensure we write a new modification time
cat "$SOURCE_FILE" > "$PAYLOAD_FILE"

echo "Deployed '$SOURCE_FILE' to Blender bridge."
