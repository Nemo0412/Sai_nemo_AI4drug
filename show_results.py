#!/usr/bin/env python3
"""
快速查看评估结果
"""
import json
import sys

if len(sys.argv) > 1:
    result_file = sys.argv[1]
else:
    result_file = 'qwen_multitask_rdkit_checkpoints/checkpoint_140_full_evaluation.json'

try:
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    checkpoint_name = results.get('checkpoint', 'Unknown').split('/')[-1]
    
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS - {checkpoint_name}")
    print(f"{'='*80}")
    
    # Efficiency
    eff = results['efficiency']
    if 'adaptive_tolerance' in eff:
        eff_acc = eff['adaptive_tolerance']['overall_accuracy']
        eff_note = "(极端值±1, 中间值±2)"
    else:
        eff_acc = eff['pm2_accuracy']
        eff_note = "(±2)"
    
    print(f"  Efficiency MAE:          {eff['mae']:.3f}")
    print(f"  Efficiency Accuracy:     {eff_acc:.1%} {eff_note}")
    
    # Toxicity
    tox = results['toxicity']
    print(f"  Toxicity Accuracy:       {tox['adjusted_accuracy']:.1%} (包含{eff['total_samples']}个eff样本)")
    
    print(f"{'='*80}\n")
    
except FileNotFoundError:
    print(f"错误: 找不到文件 {result_file}")
except Exception as e:
    print(f"错误: {e}")
