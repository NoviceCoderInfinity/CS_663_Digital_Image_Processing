#!/bin/bash

# Check if a directory is passed as an argument
if [ -z "$1" ]; then
  echo "Please provide a directory path."
  exit 1
fi

# Navigate to the specified directory
TARGET_DIR="$1"

# Ensure the target directory exists
if [ ! -d "$TARGET_DIR" ]; then
  echo "Directory '$TARGET_DIR' does not exist."
  exit 1
fi

# Start cleanup in the specified directory
find "$TARGET_DIR" -type f \( \
  -name "*.aux" -o \
  -name "*.fls" -o \
  -name "*.out" -o \
  -name "*.synctex.gz" -o \
  -name "*.fdb_latexmk" -o \
  -name "*.log" \
\) -exec rm -v {} +

echo "Cleanup complete in '$TARGET_DIR'."
