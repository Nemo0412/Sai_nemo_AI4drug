#!/bin/bash
# 监控训练进度

LOG_DIR="logs"

echo "=================================================="
echo "  Training Progress Monitor"
echo "=================================================="
echo ""

# Find latest log file
if [ -f "$LOG_DIR/training_pid.txt" ]; then
    PID=$(cat "$LOG_DIR/training_pid.txt")
    
    # Check if process is running
    if ps -p $PID > /dev/null 2>&1; then
        echo "✓ Training process is running (PID: $PID)"
    else
        echo "⚠ Training process not found (PID: $PID may have stopped)"
    fi
    echo ""
else
    echo "⚠ No PID file found"
    echo ""
fi

# Find latest log file
LATEST_LOG=$(ls -t $LOG_DIR/training_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ No training log found"
    exit 1
fi

echo "Log file: $LATEST_LOG"
echo ""

# Extract training progress
echo "=================================================="
echo "  Recent Progress"
echo "=================================================="
tail -30 "$LATEST_LOG"
echo ""

# Check for checkpoints
echo "=================================================="
echo "  Saved Checkpoints"
echo "=================================================="
if [ -d "checkpoints/qwen32b_multitask_ckpts" ]; then
    CKPTS=$(ls -d checkpoints/qwen32b_multitask_ckpts/checkpoint-* 2>/dev/null | wc -l)
    echo "Total checkpoints: $CKPTS / 10"
    echo ""
    
    if [ $CKPTS -gt 0 ]; then
        echo "Checkpoint list:"
        ls -lh checkpoints/qwen32b_multitask_ckpts/ | grep "checkpoint-" | awk '{print "  " $9 " (" $5 ")"}'
    fi
else
    echo "No checkpoints directory found yet"
fi
echo ""

# Check evaluation results
echo "=================================================="
echo "  Evaluation Results"
echo "=================================================="
if [ -f "results/evaluation_results.json" ]; then
    echo "✓ Evaluation results file exists"
    
    # Extract summary if available
    python3 << 'EOF'
import json
import os

if os.path.exists('checkpoints/qwen32b_multitask_ckpts/evaluation_results.json'):
    with open('checkpoints/qwen32b_multitask_ckpts/evaluation_results.json', 'r') as f:
        results = json.load(f)
    
    print(f"\nEvaluated checkpoints: {len(results)}")
    print("\nStep | Eff MAE | Eff ±1 | Tox Acc | Tox F1")
    print("-" * 50)
    
    for r in results[-5:]:  # Show last 5
        step = r['step']
        eff_mae = r['results']['efficiency']['mae']
        eff_pm1 = r['results']['efficiency']['pm1_accuracy']
        tox_acc = r['results']['toxicity']['accuracy']
        tox_f1 = r['results']['toxicity'].get('f1', 0)
        print(f"{step:4d} | {eff_mae:7.3f} | {eff_pm1:6.1%} | {tox_acc:7.1%} | {tox_f1:6.3f}")
else:
    print("No evaluation results yet")
EOF
else
    echo "⚠ No evaluation results yet"
fi
echo ""

echo "=================================================="
echo "  Commands"
echo "=================================================="
echo "View live log:"
echo "  tail -f $LATEST_LOG"
echo ""
echo "GPU usage:"
echo "  nvidia-smi"
echo ""
