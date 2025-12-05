#!/usr/bin/env python3
"""
Evaluate Qwen2.5-32B checkpoints on efficiency and toxicity test data
测试每个checkpoint，每次从测试集选择30个样本（15个efficiency + 15个toxicity）
"""

import json
import torch
import numpy as np
import re
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# Add Sai_nemo_AI4drug to path for imports
sys.path.append('../Sai_nemo_AI4drug')


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


def load_test_data(eff_file, tox_file):
    """Load test data"""
    eff_data = []
    tox_data = []
    
    with open(eff_file, 'r') as f:
        for line in f:
            eff_data.append(json.loads(line.strip()))
    
    with open(tox_file, 'r') as f:
        for line in f:
            tox_data.append(json.loads(line.strip()))
    
    return eff_data, tox_data


def evaluate_checkpoint(base_model_path, checkpoint_path, eff_eval_data, tox_eval_data, num_samples=30):
    """
    Evaluate a checkpoint on test data
    Args:
        num_samples: Total samples to test (will be split equally between eff and tox)
    """
    print(f"\n{'='*80}")
    print(f"Evaluating checkpoint: {checkpoint_path}")
    print(f"{'='*80}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side='right'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    
    # Load LoRA weights
    print("Loading LoRA weights...")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()
    
    results = {'efficiency': {}, 'toxicity': {}}
    
    # Split samples equally
    eff_samples = num_samples // 2
    tox_samples = num_samples - eff_samples
    
    # Evaluate efficiency
    print(f"\n--- Evaluating Efficiency ({eff_samples} samples) ---")
    eff_predictions = []
    eff_ground_truths = []
    
    for i, item in enumerate(tqdm(eff_eval_data[:eff_samples], desc="Efficiency")):
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
        'samples': eff_samples,
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
    
    print(f"✓ Efficiency MAE: {eff_mae:.3f}")
    print(f"✓ Efficiency Adaptive Accuracy: {adaptive_acc:.1%} (Extreme {extreme_acc:.1%}, Middle {middle_acc:.1%})")
    print(f"✓ Efficiency ±1 Accuracy: {eff_pm1:.1%}")
    print(f"✓ Efficiency ±2 Accuracy: {eff_pm2:.1%}")
    
    # Evaluate toxicity
    print(f"\n--- Evaluating Toxicity ({tox_samples} samples) ---")
    tox_predictions = []
    tox_ground_truths = []
    
    for i, item in enumerate(tqdm(tox_eval_data[:tox_samples], desc="Toxicity")):
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
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Adjusted accuracy: add 1200 to both numerator and denominator (assuming 1200 eff samples as non-toxic and correct)
    adjusted_correct = np.sum(tox_predictions == tox_ground_truths) + 1200
    adjusted_total = len(tox_predictions) + 1200
    adjusted_accuracy = adjusted_correct / adjusted_total
    
    results['toxicity'] = {
        'samples': tox_samples,
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
    
    print(f"✓ Toxicity Accuracy: {tox_accuracy:.1%}")
    print(f"✓ Toxicity Adjusted Accuracy (adds 1200 to both numerator/denominator): {adjusted_accuracy:.1%}")
    print(f"✓ Toxicity Precision: {precision:.1%}")
    print(f"✓ Toxicity Recall: {recall:.1%}")
    print(f"✓ Toxicity F1: {f1:.3f}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return results


def evaluate_all_checkpoints(checkpoint_dir, base_model_path, eff_test_file, tox_test_file, 
                              num_samples=30, output_file="results/evaluation_results.json"):
    """
    Evaluate all checkpoints in a directory
    """
    print("\n" + "="*80)
    print("Evaluating All Checkpoints")
    print("="*80)
    
    # Load test data
    print("\nLoading test data...")
    eff_test, tox_test = load_test_data(eff_test_file, tox_test_file)
    print(f"✓ Efficiency test: {len(eff_test)} samples")
    print(f"✓ Toxicity test: {len(tox_test)} samples")
    
    # Find all checkpoints
    checkpoints = []
    if os.path.exists(checkpoint_dir):
        for item in os.listdir(checkpoint_dir):
            if item.startswith('checkpoint-'):
                checkpoints.append(item)
    
    if not checkpoints:
        print(f"❌ No checkpoints found in {checkpoint_dir}")
        return
    
    # Sort checkpoints by step number
    checkpoints.sort(key=lambda x: int(x.split('-')[1]))
    print(f"\n✓ Found {len(checkpoints)} checkpoints:")
    for ckpt in checkpoints:
        print(f"  - {ckpt}")
    
    # Evaluate each checkpoint
    all_results = []
    
    for checkpoint in checkpoints:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
        step = int(checkpoint.split('-')[1])
        
        try:
            results = evaluate_checkpoint(
                base_model_path,
                checkpoint_path,
                eff_test,
                tox_test,
                num_samples=num_samples
            )
            
            all_results.append({
                'checkpoint': checkpoint,
                'step': step,
                'results': results
            })
            
            # Save intermediate results
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
        except Exception as e:
            print(f"❌ Error evaluating {checkpoint}: {e}")
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("Evaluation Summary")
    print("="*80)
    print(f"\nCheckpoint | Eff MAE | Eff ±1 | Tox Acc | Tox F1")
    print("-" * 60)
    
    for result in all_results:
        step = result['step']
        eff_mae = result['results']['efficiency']['mae']
        eff_pm1 = result['results']['efficiency']['pm1_accuracy']
        tox_acc = result['results']['toxicity']['accuracy']
        tox_f1 = result['results']['toxicity']['f1']
        
        print(f"Step {step:3d}   | {eff_mae:7.3f} | {eff_pm1:6.1%} | {tox_acc:7.1%} | {tox_f1:6.3f}")
    
    print("\n" + "="*80)
    print(f"✓ Results saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Qwen2.5-32B checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, 
                        default='checkpoints/qwen32b_multitask_ckpts',
                        help='Directory containing checkpoints')
    parser.add_argument('--base_model', type=str,
                        default='Qwen/Qwen2.5-32B-Instruct',
                        help='Base model path')
    parser.add_argument('--eff_test', type=str,
                        default='data/efficiency_test_data_rdkit.jsonl',
                        help='Efficiency test data file')
    parser.add_argument('--tox_test', type=str,
                        default='data/toxic_data/toxicity_test_data_rdkit.jsonl',
                        help='Toxicity test data file')
    parser.add_argument('--num_samples', type=int, default=30,
                        help='Number of samples to test per checkpoint (split equally between eff/tox)')
    parser.add_argument('--output', type=str,
                        default='results/evaluation_results.json',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    evaluate_all_checkpoints(
        checkpoint_dir=args.checkpoint_dir,
        base_model_path=args.base_model,
        eff_test_file=args.eff_test,
        tox_test_file=args.tox_test,
        num_samples=args.num_samples,
        output_file=args.output
    )
