#!/bin/bash
#SBATCH --job-name=multi_task_finetune
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err

# Create logs directory
mkdir -p logs
mkdir -p checkpoint_results

# Load modules
module load cuda/12.1.1

# Print GPU info
echo "=========================================="
echo "Multi-Task Finetuning Job Started"
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo ""
nvidia-smi
echo ""

# Change to project directory
cd /users/7/li003385/workspace/Ai4drug/Sai_nemo_AI4drug

# Print configuration
echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "Config file: training_config.yaml"
echo "Save every: 5 steps"
echo "Evaluate every: 5 steps"
echo "Output directory: ./qwen_lora_finetuned_multi_task"
echo ""

# Run training
echo "=========================================="
echo "Starting Training"
echo "=========================================="
python3 multi_task_training.py

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training Completed Successfully"
    echo "=========================================="
    echo ""
    
    # Evaluate all checkpoints
    echo "=========================================="
    echo "Evaluating All Checkpoints"
    echo "=========================================="
    python3 evaluate_checkpoints.py --config training_config.yaml
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "Evaluation Completed"
        echo "=========================================="
        echo "Results saved in: checkpoint_results/"
        echo "Summary file: checkpoint_results/all_checkpoints_summary.json"
    else
        echo ""
        echo "Warning: Evaluation failed, but training completed successfully"
    fi
else
    echo ""
    echo "=========================================="
    echo "Training Failed"
    echo "=========================================="
    exit 1
fi

echo ""
echo "Job completed at: $(date)"
echo "=========================================="

