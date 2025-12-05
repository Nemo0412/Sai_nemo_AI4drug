#!/bin/bash
# 训练完成后评估所有checkpoints

echo "=================================================="
echo "  Evaluate All Checkpoints"
echo "=================================================="
echo ""

# Check if checkpoints exist
if [ ! -d "checkpoints/qwen32b_multitask_ckpts" ]; then
    echo "❌ No checkpoint directory found"
    exit 1
fi

CKPTS=$(ls -d checkpoints/qwen32b_multitask_ckpts/checkpoint-* 2>/dev/null | wc -l)
echo "Found $CKPTS checkpoints"
echo ""

if [ $CKPTS -eq 0 ]; then
    echo "❌ No checkpoints to evaluate"
    exit 1
fi

# Set environment
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1

# Run evaluation
LOG_FILE="logs/evaluation_$(date +%Y%m%d_%H%M%S).log"

echo "Starting evaluation..."
echo "Log file: $LOG_FILE"
echo ""

python3 -u evaluate_checkpoints.py \
    --checkpoint_dir checkpoints/qwen32b_multitask_ckpts \
    --base_model Qwen/Qwen2.5-32B-Instruct \
    --eff_test data/efficiency_test_data_rdkit.jsonl \
    --tox_test data/toxic_data/toxicity_test_data_rdkit.jsonl \
    --num_samples 30 \
    --output results/evaluation_results.json \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=================================================="
echo "✓ Evaluation complete!"
echo "  Results: results/evaluation_results.json"
echo "  Log: $LOG_FILE"
echo "=================================================="
