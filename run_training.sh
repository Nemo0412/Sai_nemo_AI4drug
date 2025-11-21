#!/bin/bash
# Script to run multi-task training with checkpoint evaluation

echo "=========================================="
echo "Starting Multi-Task Training"
echo "=========================================="
echo ""

# Run training
python multi_task_training.py

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo ""
echo "Evaluating all checkpoints..."
echo ""

# Evaluate all checkpoints
python evaluate_checkpoints.py --config training_config.yaml

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="
echo ""
echo "Results saved in: checkpoint_results/"
echo "Summary file: checkpoint_results/all_checkpoints_summary.json"
echo ""

