"""
Split efficiency data into training (600) and test (600) sets with shuffle
"""

import json
import random
import os


def split_efficiency_data(train_file="data/train_data.jsonl", 
                         test_file="data/test_data.jsonl",
                         train_output="data/efficiency_train_data.jsonl",
                         test_output="data/efficiency_test_data.jsonl",
                         train_size=600,
                         test_size=600,
                         seed=42):
    """
    Combine and split efficiency data into training and test sets
    
    Args:
        train_file: Original training data file
        test_file: Original test data file
        train_output: Output path for new training data
        test_output: Output path for new test data
        train_size: Number of training samples
        test_size: Number of test samples
        seed: Random seed for reproducibility
    """
    
    print("="*80)
    print("SPLITTING EFFICIENCY DATA")
    print("="*80)
    
    # Set random seed
    random.seed(seed)
    
    # Load all data
    all_data = []
    
    # Load from training file
    if os.path.exists(train_file):
        print(f"\nLoading data from: {train_file}")
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                all_data.append(json.loads(line))
        print(f"  Loaded {len(all_data)} samples")
    else:
        print(f"Warning: {train_file} not found")
    
    # Load from test file
    if os.path.exists(test_file):
        print(f"\nLoading data from: {test_file}")
        initial_count = len(all_data)
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                all_data.append(json.loads(line))
        print(f"  Loaded {len(all_data) - initial_count} additional samples")
    else:
        print(f"Warning: {test_file} not found")
    
    total_samples = len(all_data)
    print(f"\nTotal samples available: {total_samples}")
    print(f"Required: {train_size} training + {test_size} test = {train_size + test_size} samples")
    
    if total_samples < train_size + test_size:
        print(f"\n⚠ Warning: Not enough samples! Have {total_samples}, need {train_size + test_size}")
        print(f"Adjusting split...")
        # Adjust sizes proportionally
        available = total_samples
        train_size = min(train_size, available - test_size)
        test_size = available - train_size
        print(f"New split: {train_size} training + {test_size} test")
    
    # Shuffle data
    print(f"\nShuffling data (seed={seed})...")
    random.shuffle(all_data)
    
    # Split data
    train_data = all_data[:train_size]
    test_data = all_data[train_size:train_size + test_size]
    
    print(f"\nSplit results:")
    print(f"  Training samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")
    
    # Save training data
    os.makedirs(os.path.dirname(train_output), exist_ok=True)
    with open(train_output, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"\n✓ Saved training data to: {train_output}")
    
    # Save test data
    os.makedirs(os.path.dirname(test_output), exist_ok=True)
    with open(test_output, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✓ Saved test data to: {test_output}")
    
    # Show sample distribution
    print(f"\nSample efficiency scores in training set (first 10):")
    import re
    for i, item in enumerate(train_data[:10]):
        msg = item['messages'][-1]['content']
        match = re.search(r'is (\d+)', msg)
        if match:
            score = int(match.group(1))
            print(f"  Sample {i+1}: {score}")
    
    print(f"\nSample efficiency scores in test set (first 10):")
    for i, item in enumerate(test_data[:10]):
        msg = item['messages'][-1]['content']
        match = re.search(r'is (\d+)', msg)
        if match:
            score = int(match.group(1))
            print(f"  Sample {i+1}: {score}")
    
    print("\n" + "="*80)
    print("DATA SPLIT COMPLETED")
    print("="*80)
    
    return train_data, test_data


if __name__ == "__main__":
    split_efficiency_data(
        train_file="data/train_data.jsonl",
        test_file="data/test_data.jsonl",
        train_output="data/efficiency_train_data.jsonl",
        test_output="data/efficiency_test_data.jsonl",
        train_size=600,
        test_size=600,
        seed=42
    )

