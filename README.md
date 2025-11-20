# LipoAgent - Multi-Agent Drug Discovery Framework

Multi-agent framework for lipid nanoparticle (LNP) delivery systems with conditional multi-task loss training.

## Quick Start

### 1. Finetune Training (Multi-Task Loss)
```bash
python multi_task_training.py
```
- **Config**: `training_config.yaml`
- **Output**: `./qwen_lora_finetuned_multi_task/`
- **Data**: `data/toxic_data/toxicity_train_data.jsonl` (200 samples)

### 2. Multi-Agent Prediction Loop
```bash
python multi_agent_pipeline.py
```
- **Config**: `config.yaml`
- **Output**: `output/` directory
- **Requires**: Trained model from step 1

## Complete Workflow

```bash
# Step 1: Prepare data (if needed)
python process_toxicity_data.py

# Step 2: Train model
python multi_task_training.py

# Step 3: Run multi-agent prediction
python multi_agent_pipeline.py
```

## Architecture

**Multi-Agent Framework:**
- **Predictor Agent**: Predicts toxicity + efficiency with reasoning (uses finetuned model)
- **Verifier Agent**: Validates logical consistency (uses base model)
- **Negotiation**: Up to 2 loops, then human feedback if no agreement

**Training:**
- **Conditional Multi-Task Loss**: `loss = loss_tox + alpha * loss_eff`
  - Toxicity loss: Binary cross-entropy (all samples)
  - Efficiency loss: Cross-entropy (only non-toxic samples)

## Configuration

### Training Config (`training_config.yaml`)
```yaml
loss:
  alpha: 1.0  # Efficiency loss weight
  efficiency_num_classes: 10

training:
  num_epochs: 3
  learning_rate: 2e-4
  batch_size: 4
```

### Multi-Agent Config (`config.yaml`)
```yaml
model:
  base_model_path: "Qwen/Qwen-7B-Chat"
  lora_path: "./qwen_lora_finetuned"  # Path to trained model

multi_agent:
  max_negotiation_loops: 2
  low_confidence_threshold: 6.0
```

## Output Files

**Training:**
- Model saved to: `./qwen_lora_finetuned_multi_task/`

**Multi-Agent:**
- `output/agreed_molecules.json` - Agent agreements
- `output/disagreed_molecules.json` - Agent disagreements
- `output/human_feedback_molecules.json` - Human feedback cases
- `output/predictions.json` - All predictions
- `output/feedback_history.json` - Feedback history

## Verify Setup

```bash
python verify_code.py
```

## Requirements

- Python 3.8+
- PyTorch, Transformers, PEFT
- See `requirements.txt`

## Key Files

| File | Purpose |
|------|---------|
| `multi_task_training.py` | Finetune with conditional loss |
| `multi_agent_pipeline.py` | Multi-agent prediction loop |
| `predictor_agent.py` | Predictor agent implementation |
| `verifier_agent.py` | Verifier agent implementation |
| `training_config.yaml` | Training hyperparameters |
| `config.yaml` | Multi-agent configuration |
