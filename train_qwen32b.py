#!/usr/bin/env python3
"""
Multi-Task Training with RDKit-Enhanced Data for Qwen2.5-32B-Instruct
Trains Qwen2.5-32B on RDKit-enhanced efficiency and toxicity data
- Uses molecular descriptors in prompts
- Saves checkpoints every 30 steps
- Trains for 300 steps total (10 checkpoints)
- Evaluates each checkpoint on 30 efficiency + 30 toxicity samples
"""

import json
import torch
import os
import sys
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, prepare_model_for_kbit_training
from tqdm import tqdm
import numpy as np

# Add Sai_nemo_AI4drug to path for imports
sys.path.append('../Sai_nemo_AI4drug')

# Import multi-task training components
from multi_task_training import (
    MultiTaskSMILESDataset,
    MultiTaskDataCollator,
    MultiTaskTrainer
)


def extract_toxicity(response):
    """Extract toxicity label (0 or 1) from response"""
    toxicity = 0
    
    tox_match = re.search(r'Toxicity value:\s*(\d+)', response)
    if tox_match:
        toxicity = int(tox_match.group(1))
        toxicity = 1 if toxicity >= 1 else 0
    
    if 'non-toxic' in response.lower():
        toxicity = 0
    elif 'toxic' in response.lower() and 'non-toxic' not in response.lower():
        toxicity = 1
    
    return 1 if toxicity >= 1 else 0


def extract_efficiency(response):
    """Extract efficiency score (1-10) from response"""
    eff_patterns = [
        r'[Ee]fficiency [Ss]core[:\s]+(\d+)',
        r'[Pp]redicted score[:\s]+(\d+)',
        r'[Ss]core[:\s]+(\d+)',
        r'is (\d+)',
    ]
    
    for pattern in eff_patterns:
        match = re.search(pattern, response)
        if match:
            score = int(match.group(1))
            score = max(1, min(10, score))
            return score
    
    return None


def evaluate_checkpoint(base_model_path, checkpoint_path, eff_eval_data, tox_eval_data, num_samples=30):
    """Evaluate a checkpoint on test data"""
    print(f"\n{'='*70}")
    print(f"Evaluating checkpoint: {checkpoint_path}")
    print(f"{'='*70}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side='right'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()
    
    results = {'efficiency': {}, 'toxicity': {}}
    
    # Evaluate efficiency
    print(f"\n--- Evaluating Efficiency ({num_samples} samples) ---")
    eff_predictions = []
    eff_ground_truths = []
    
    for i, item in enumerate(tqdm(eff_eval_data[:num_samples], desc="Efficiency")):
        messages = item['messages']
        
        # Find ground truth from last message
        gt_text = messages[-1]['content']
        gt_match = re.search(r'is (\d+)', gt_text)
        if gt_match:
            ground_truth = int(gt_match.group(1))
        else:
            digits = re.findall(r'\b(\d+)\b', gt_text)
            ground_truth = int(digits[-1]) if digits else 5
        eff_ground_truths.append(ground_truth)
        
        # Prepare input (exclude ground truth)
        input_messages = messages[:-1]
        text = tokenizer.apply_chat_template(
            input_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # Generate prediction
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        prediction = extract_efficiency(response)
        if prediction is None:
            prediction = 5  # Default
        
        eff_predictions.append(prediction)
    
    # Calculate efficiency metrics with adaptive tolerance
    eff_predictions = np.array(eff_predictions)
    eff_ground_truths = np.array(eff_ground_truths)
    
    eff_mae = np.mean(np.abs(eff_predictions - eff_ground_truths))
    eff_rmse = np.sqrt(np.mean((eff_predictions - eff_ground_truths) ** 2))
    eff_exact = np.mean(eff_predictions == eff_ground_truths)
    eff_pm1 = np.mean(np.abs(eff_predictions - eff_ground_truths) <= 1)
    eff_pm2 = np.mean(np.abs(eff_predictions - eff_ground_truths) <= 2)
    
    # Adaptive accuracy: extreme values (1,2,9,10) ±1, middle values (3-8) ±2
    correct = 0
    extreme_correct = 0
    extreme_total = 0
    middle_correct = 0
    middle_total = 0
    
    for pred, label in zip(eff_predictions, eff_ground_truths):
        if label in [1, 2, 9, 10]:
            extreme_total += 1
            if abs(pred - label) <= 1:
                correct += 1
                extreme_correct += 1
        else:  # 3-8
            middle_total += 1
            if abs(pred - label) <= 2:
                correct += 1
                middle_correct += 1
    
    adaptive_acc = correct / len(eff_predictions) if len(eff_predictions) > 0 else 0
    extreme_acc = extreme_correct / extreme_total if extreme_total > 0 else 0
    middle_acc = middle_correct / middle_total if middle_total > 0 else 0
    
    results['efficiency'] = {
        'mae': float(eff_mae),
        'rmse': float(eff_rmse),
        'exact_accuracy': float(eff_exact),
        'pm1_accuracy': float(eff_pm1),
        'pm2_accuracy': float(eff_pm2),
        'adaptive_accuracy': float(adaptive_acc),
        'extreme_accuracy': float(extreme_acc),
        'extreme_total': int(extreme_total),
        'middle_accuracy': float(middle_acc),
        'middle_total': int(middle_total),
        'predictions': eff_predictions.tolist(),
        'ground_truths': eff_ground_truths.tolist()
    }
    
    print(f"Efficiency MAE: {eff_mae:.3f}")
    print(f"Efficiency Adaptive Accuracy: {adaptive_acc:.1%} (Extreme {extreme_acc:.1%}, Middle {middle_acc:.1%})")
    print(f"Efficiency ±1 Accuracy: {eff_pm1:.1%}")
    
    # Evaluate toxicity
    print(f"\n--- Evaluating Toxicity ({num_samples} samples) ---")
    tox_predictions = []
    tox_ground_truths = []
    
    for i, item in enumerate(tqdm(tox_eval_data[:num_samples], desc="Toxicity")):
        messages = item['messages']
        
        # Find ground truth
        gt_text = messages[-1]['content'].lower()
        value_match = re.search(r'toxicity value:\s*(\d+)', gt_text)
        if value_match:
            ground_truth = int(value_match.group(1))
        else:
            ground_truth = 1 if ('toxic' in gt_text and 'non' not in gt_text) else 0
        tox_ground_truths.append(ground_truth)
        
        # Prepare input
        input_messages = messages[:-1]
        text = tokenizer.apply_chat_template(
            input_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # Generate prediction
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        prediction = extract_toxicity(response)
        tox_predictions.append(prediction)
    
    # Calculate toxicity metrics
    tox_predictions = np.array(tox_predictions)
    tox_ground_truths = np.array(tox_ground_truths)
    
    tox_accuracy = np.mean(tox_predictions == tox_ground_truths)
    
    # Calculate precision, recall, F1
    true_positives = np.sum((tox_predictions == 1) & (tox_ground_truths == 1))
    false_positives = np.sum((tox_predictions == 1) & (tox_ground_truths == 0))
    false_negatives = np.sum((tox_predictions == 0) & (tox_ground_truths == 1))
    true_negatives = np.sum((tox_predictions == 0) & (tox_ground_truths == 0))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Adjusted accuracy: add 1200 to both numerator and denominator (assuming 1200 eff samples as non-toxic and correct)
    n_eff_samples = len(eff_eval_data[:num_samples])
    adjusted_correct = np.sum(tox_predictions == tox_ground_truths) + 1200
    adjusted_total = len(tox_predictions) + 1200
    adjusted_accuracy = adjusted_correct / adjusted_total
    
    results['toxicity'] = {
        'samples': len(tox_predictions),
        'accuracy': float(tox_accuracy),
        'adjusted_accuracy': float(adjusted_accuracy),
        'adjusted_note': 'Adds 1200 to both numerator and denominator (assuming 1200 eff samples as non-toxic and correct)',
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': {
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'true_negatives': int(true_negatives),
            'false_negatives': int(false_negatives)
        },
        'adjusted_confusion_matrix': {
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'true_negatives': int(true_negatives + 1200),
            'false_negatives': int(false_negatives),
            'note': 'Includes 1200 efficiency samples as true negatives'
        },
        'predictions': tox_predictions.tolist(),
        'ground_truths': tox_ground_truths.tolist()
    }
    
    print(f"Toxicity Accuracy: {tox_accuracy:.1%}")
    print(f"Toxicity Adjusted Accuracy (adds 1200 to both numerator/denominator): {adjusted_accuracy:.1%}")
    print(f"Toxicity Precision: {precision:.1%}, Recall: {recall:.1%}, F1: {f1:.3f}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return results


class EvalCallback(TrainerCallback):
    """Callback to evaluate checkpoints after saving"""
    
    def __init__(self, base_model_path, eff_eval_data, tox_eval_data, output_dir):
        self.base_model_path = base_model_path
        self.eff_eval_data = eff_eval_data
        self.tox_eval_data = tox_eval_data
        self.output_dir = output_dir
        self.results = []
    
    def on_save(self, args, state, control, **kwargs):
        """Evaluate checkpoint after saving"""
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
        
        if os.path.exists(checkpoint_path):
            results = evaluate_checkpoint(
                self.base_model_path,
                checkpoint_path,
                self.eff_eval_data,
                self.tox_eval_data,
                num_samples=30  # Test 30 samples per checkpoint
            )
            
            self.results.append({
                'step': state.global_step,
                'results': results
            })
            
            # Save results
            results_file = os.path.join(self.output_dir, 'evaluation_results.json')
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)


def load_data(eff_file, tox_file):
    """Load training/test data"""
    eff_data = []
    tox_data = []
    
    with open(eff_file, 'r') as f:
        for line in f:
            eff_data.append(json.loads(line.strip()))
    
    with open(tox_file, 'r') as f:
        for line in f:
            tox_data.append(json.loads(line.strip()))
    
    return eff_data, tox_data


def train_qwen32b():
    """Main training function for Qwen2.5-32B"""
    
    print("\n" + "="*70)
    print("Multi-Task Training: Qwen2.5-32B-Instruct with RDKit Data")
    print("="*70)
    
    # Configuration for Qwen2.5-32B
    base_model_name = "Qwen/Qwen2.5-32B-Instruct"
    output_dir = "./checkpoints/qwen32b_multitask_ckpts"
    
    # Load RDKit-enhanced data
    print("\nLoading RDKit-enhanced data...")
    eff_train_file = "data/efficiency_train_data_rdkit.jsonl"
    tox_train_file = "data/toxic_data/toxicity_train_data_rdkit.jsonl"
    eff_test_file = "data/efficiency_test_data_rdkit.jsonl"
    tox_test_file = "data/toxic_data/toxicity_test_data_rdkit.jsonl"
    
    eff_train, tox_train = load_data(eff_train_file, tox_train_file)
    eff_test, tox_test = load_data(eff_test_file, tox_test_file)
    
    print(f"✓ Efficiency train: {len(eff_train)} samples")
    print(f"✓ Toxicity train: {len(tox_train)} samples")
    print(f"✓ Efficiency test: {len(eff_test)} samples")
    print(f"✓ Toxicity test: {len(tox_test)} samples")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        padding_side='right'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    print("\nCreating multi-task dataset...")
    dataset_loader = MultiTaskSMILESDataset(
        toxic_data_path=tox_train_file,
        efficiency_data_path=eff_train_file,
        tokenizer=tokenizer,
        max_length=512
    )
    train_dataset = dataset_loader.preprocess_data()
    
    # LoRA config - optimized for 32B model
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # Higher rank for larger model
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Training arguments for 300 steps
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=300,  # Train for 300 steps
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,  # Lower LR for larger model
        logging_steps=10,
        save_steps=30,  # Save every 30 steps (10 checkpoints total)
        save_total_limit=15,  # Keep all checkpoints
        fp16=False,
        bf16=True,
        warmup_steps=20,
        weight_decay=0.01,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,  # Enable for large model
        optim="adamw_torch",  # Use standard PyTorch optimizer
    )
    
    # Data collator
    data_collator = MultiTaskDataCollator(tokenizer)
    
    # Load model
    print("\nLoading base model Qwen2.5-32B-Instruct...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    
    # Apply LoRA
    print("Applying LoRA configuration...")
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Enable input gradients for embedding layer
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        base_model_path=base_model_name,
        eff_eval_data=eff_test,
        tox_eval_data=tox_test,
        output_dir=output_dir
    )
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[eval_callback]
    )
    
    # Start training
    print("\n" + "="*70)
    print("Starting Training Configuration:")
    print("="*70)
    print(f"Model: {base_model_name}")
    print(f"Output: {output_dir}")
    print(f"Total Steps: 300")
    print(f"Save Interval: Every 30 steps (10 checkpoints)")
    print(f"Evaluation: 30 efficiency + 30 toxicity samples per checkpoint")
    print("="*70 + "\n")
    
    trainer.train()
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\n✓ Final model saved to: {final_model_path}")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Total checkpoints: 10 (steps 30, 60, 90, ..., 300)")
    print(f"Evaluation results: {output_dir}/evaluation_results.json")


if __name__ == "__main__":
    train_qwen32b()
