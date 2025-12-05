#!/bin/bash
# 启动 Qwen2.5-32B 训练脚本

set -e

echo "=================================================="
echo "  Qwen2.5-32B Multi-Task Training Launcher"
echo "=================================================="
echo ""
echo "Configuration:"
echo "  - Model: Qwen/Qwen2.5-32B-Instruct"
echo "  - Total Steps: 300"
echo "  - Save Interval: Every 30 steps"
echo "  - Expected Checkpoints: 10"
echo "  - Evaluation: 30 samples per checkpoint"
echo ""
echo "=================================================="
echo ""

# Create necessary directories
mkdir -p logs checkpoints results

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export PYTHONUNBUFFERED=1

# Log file
LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"

echo "Starting training..."
echo "Log file: $LOG_FILE"
echo ""

# Run training in background
nohup python3 -u train_qwen32b.py > "$LOG_FILE" 2>&1 &

# Get PID
PID=$!
echo $PID > logs/training_pid.txt

echo "✓ Training started!"
echo "  PID: $PID"
echo "  Log: $LOG_FILE"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Check progress:"
echo "  ./monitor_training.sh"
echo ""
echo "Stop training:"
echo "  kill $PID"
echo ""
