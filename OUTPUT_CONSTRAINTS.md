# Output Constraints

## Toxicity Prediction
- **Output Type**: Discrete binary value
- **Valid Values**: 0 or 1 only
- **0**: Non-toxic
- **1**: Toxic
- **Enforcement**: 
  - Labels are clamped to [0, 1] during training
  - Predictions are rounded to 0 or 1 during evaluation

## Efficiency Prediction
- **Output Type**: Discrete integer value
- **Valid Values**: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 only
- **Enforcement**:
  - Labels are clamped to [1, 10] and converted to integers during training
  - Predictions are clamped to [1, 10] and converted to integers during evaluation
  - Efficiency loss is only computed for non-toxic samples (toxicity == 0)

## Implementation Details

### Training (`multi_task_training.py`)
- Toxicity labels: Clamped to [0, 1] before loss calculation
- Efficiency labels: Clamped to [1, 10] and converted to integers before loss calculation
- Efficiency loss mask: Only non-toxic samples (toxicity == 0) contribute to efficiency loss

### Evaluation (`evaluate_multi_task.py`)
- Toxicity extraction: Returns 0 or 1 only
- Efficiency extraction: Returns integer in range [1, 10] only
- All predictions are validated and clamped to valid ranges

