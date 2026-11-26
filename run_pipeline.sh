#!/bin/bash

# Predictive Maintenance RUL Prediction - Complete Pipeline

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "================================"
echo "RUL Prediction Pipeline"
echo "================================"

cd "$SCRIPT_DIR"

echo ""
echo "1. Downloading official NASA C-MAPSS data..."
"$PYTHON_BIN" src/download_data.py

echo ""
echo "2. Training the multi-source LSTM..."
"$PYTHON_BIN" src/train.py --require-cuda --epochs 20 --patience 5 --batch-size 256 --standardize-target --schedule --clip-grad 1 --validation-only

echo ""
echo "3. Evaluating saved checkpoint..."
"$PYTHON_BIN" src/evaluate.py --dataset ALL

echo ""
echo "================================"
echo "Pipeline completed successfully!"
echo "================================"
