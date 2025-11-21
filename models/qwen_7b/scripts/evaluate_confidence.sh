#!/bin/bash
#SBATCH --job-name=eval_conf_7b
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/eval_confidence_%j.out
#SBATCH --error=logs/eval_confidence_%j.err

# Load CUDA
module load cuda

# Navigate to project directory
cd /users/7/li003385/workspace/Ai4drug/Sai_nemo_AI4drug

# Run evaluation with confidence
echo "Starting evaluation with confidence extraction..."
python3 evaluate_with_confidence.py

echo "Evaluation complete!"

