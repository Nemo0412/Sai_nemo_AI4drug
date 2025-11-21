# Training with Checkpoint Evaluation

This guide explains how to run training with automatic checkpoint saving and evaluation.

## Configuration

The training is configured to:
- **Save checkpoint every 5 steps** (`save_steps: 5`)
- **Evaluate every 5 steps** (`eval_steps: 5`)
- **Keep up to 100 checkpoints** (`save_total_limit: 100`)

Configuration file: `training_config.yaml`

## Running Training

### Option 1: Run training only
```bash
python multi_task_training.py
```

This will:
1. Train the model with multi-task loss (toxicity + efficiency)
2. Save checkpoints every 5 steps to `./qwen_lora_finetuned_multi_task/checkpoint-{step}/`
3. Print instructions for evaluating checkpoints

### Option 2: Run training + automatic evaluation
```bash
./run_training.sh
```

or

```bash
bash run_training.sh
```

This will:
1. Run training (saves checkpoints every 5 steps)
2. Automatically evaluate all saved checkpoints after training completes
3. Save evaluation results to `checkpoint_results/`

## Evaluating Checkpoints

### Evaluate all checkpoints after training
```bash
python evaluate_checkpoints.py --config training_config.yaml
```

### Evaluate specific checkpoint directory
```bash
python evaluate_checkpoints.py \
    --config training_config.yaml \
    --output_dir ./qwen_lora_finetuned_multi_task
```

### Custom results directory
```bash
python evaluate_checkpoints.py \
    --config training_config.yaml \
    --results_dir my_results
```

## Evaluation Results

### Output Structure
```
checkpoint_results/
├── checkpoint-5_results.json      # Results for step 5
├── checkpoint-10_results.json     # Results for step 10
├── checkpoint-15_results.json     # Results for step 15
├── ...
└── all_checkpoints_summary.json   # Summary of all checkpoints
```

### Metrics Computed

For each checkpoint, the following metrics are computed:

**Toxicity Prediction:**
- `toxicity_accuracy`: Accuracy for binary classification (0/1)

**Efficiency Prediction:**
- `efficiency_accuracy`: Exact match accuracy (1-10)
- `efficiency_mae`: Mean Absolute Error
- `efficiency_rmse`: Root Mean Squared Error
- `efficiency_within_1`: Accuracy within ±1
- `efficiency_within_2`: Accuracy within ±2

### Example Result File
```json
{
  "checkpoint": "./qwen_lora_finetuned_multi_task/checkpoint-5",
  "step": 5,
  "toxicity_accuracy": 0.85,
  "toxicity_samples": 200,
  "efficiency_accuracy": 0.42,
  "efficiency_mae": 1.23,
  "efficiency_rmse": 1.56,
  "efficiency_within_1": 0.78,
  "efficiency_within_2": 0.92,
  "efficiency_samples": 600
}
```

## Viewing Results

### View summary
```bash
cat checkpoint_results/all_checkpoints_summary.json | python -m json.tool
```

### Plot accuracy over steps (optional)
You can create a simple Python script to plot the accuracy trends:
```python
import json
import matplotlib.pyplot as plt

with open('checkpoint_results/all_checkpoints_summary.json', 'r') as f:
    results = json.load(f)

steps = [r['step'] for r in results]
tox_acc = [r.get('toxicity_accuracy', 0) for r in results]
eff_acc = [r.get('efficiency_accuracy', 0) for r in results]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(steps, tox_acc, 'b-o')
plt.xlabel('Step')
plt.ylabel('Toxicity Accuracy')
plt.title('Toxicity Accuracy vs Step')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(steps, eff_acc, 'r-o')
plt.xlabel('Step')
plt.ylabel('Efficiency Accuracy')
plt.title('Efficiency Accuracy vs Step')
plt.grid(True)

plt.tight_layout()
plt.savefig('accuracy_curves.png')
```

## Notes

1. **Checkpoint Evaluation**: Evaluation happens after training completes. During training, checkpoints are saved but not evaluated (to save time).

2. **Memory**: Each checkpoint evaluation loads the model into memory. If you have many checkpoints, consider evaluating them in batches.

3. **Skip Existing**: The evaluation script skips checkpoints that have already been evaluated (if result file exists).

4. **Output Constraints**: 
   - Toxicity: Must be 0 or 1
   - Efficiency: Must be 1-10 (discrete integers)

