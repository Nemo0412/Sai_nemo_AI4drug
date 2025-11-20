"""
Process toxicity data from Carcinogenicity_Carcinogenicity.csv
Extract first 400 samples, split into 200 train and 200 test
"""

import pandas as pd
import json
import os
from pathlib import Path


def process_toxicity_data():
    """
    Process toxicity data:
    - Extract first 400 samples
    - Split into 200 train and 200 test
    - Canonical SMILES as drug ID
    - Toxicity Value (0=non-toxic, 1=toxic)
    """
    
    # File paths
    input_file = "data/toxic_data/Carcinogenicity_Carcinogenicity.csv"
    output_dir = "data/toxic_data"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("PROCESSING TOXICITY DATA")
    print("="*80)
    
    # Read CSV file
    print(f"\nReading data from: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"Total rows in file: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Remove rows with missing SMILES or Toxicity Value first
    df_clean = df.dropna(subset=['Canonical SMILES', 'Toxicity Value']).copy()
    print(f"\nAfter removing missing values: {len(df_clean)} valid samples")
    
    # Extract first 400 valid samples
    if len(df_clean) < 400:
        print(f"Warning: Only {len(df_clean)} valid samples available, using all of them")
        df_subset = df_clean.copy()
        n_samples = len(df_subset)
        n_train = n_samples // 2
        n_test = n_samples - n_train
    else:
        df_subset = df_clean.head(400).copy()
        n_samples = 400
        n_train = 200
        n_test = 200
    
    print(f"Extracted {len(df_subset)} valid samples for processing")
    
    # Split into train and test
    df_train = df_subset.head(n_train)
    df_test = df_subset.tail(n_test)
    
    print(f"\nSplit:")
    print(f"  Training set: {len(df_train)} samples")
    print(f"  Test set: {len(df_test)} samples")
    
    # Check distribution
    print(f"\nTraining set toxicity distribution:")
    print(df_train['Toxicity Value'].value_counts().sort_index())
    print(f"\nTest set toxicity distribution:")
    print(df_test['Toxicity Value'].value_counts().sort_index())
    
    # Convert to JSONL format (similar to existing training data format)
    def create_jsonl(df_split, output_file, split_name):
        """Create JSONL file from dataframe"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, row in df_split.iterrows():
                smiles = str(row['Canonical SMILES']).strip()
                toxicity = int(row['Toxicity Value'])
                
                # Map toxicity: 0 = non-toxic, 1 = toxic
                toxicity_label = "non-toxic" if toxicity == 0 else "toxic"
                
                # Create training format similar to existing format
                conversation = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant specialized in drug discovery and molecular analysis. You can predict molecular toxicity based on their SMILES structures."
                        },
                        {
                            "role": "user",
                            "content": f"Is this molecular structure toxic? {smiles}"
                        },
                        {
                            "role": "assistant",
                            "content": f"This molecular structure {smiles} is {toxicity_label}. Toxicity value: {toxicity}."
                        }
                    ]
                }
                
                f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
        
        print(f"  Saved {split_name} to: {output_file}")
    
    # Create JSONL files
    train_jsonl = os.path.join(output_dir, 'toxicity_train_data.jsonl')
    test_jsonl = os.path.join(output_dir, 'toxicity_test_data.jsonl')
    
    create_jsonl(df_train, train_jsonl, "Training set")
    create_jsonl(df_test, test_jsonl, "Test set")
    
    # Also save as CSV for easy inspection
    train_csv = os.path.join(output_dir, 'toxicity_train_data.csv')
    test_csv = os.path.join(output_dir, 'toxicity_test_data.csv')
    
    df_train.to_csv(train_csv, index=False, encoding='utf-8')
    df_test.to_csv(test_csv, index=False, encoding='utf-8')
    
    print(f"\n  Saved training CSV to: {train_csv}")
    print(f"  Saved test CSV to: {test_csv}")
    
    # Save statistics
    stats = {
        'total_samples': len(df_subset),
        'train_samples': len(df_train),
        'test_samples': len(df_test),
        'train_toxicity_distribution': df_train['Toxicity Value'].value_counts().to_dict(),
        'test_toxicity_distribution': df_test['Toxicity Value'].value_counts().to_dict(),
        'toxicity_mapping': {
            '0': 'non-toxic',
            '1': 'toxic'
        }
    }
    
    stats_file = os.path.join(output_dir, 'toxicity_data_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n  Saved statistics to: {stats_file}")
    
    # Display sample data
    print(f"\n{'='*80}")
    print("SAMPLE DATA")
    print(f"{'='*80}")
    print("\nTraining set sample (first 3):")
    for idx, row in df_train.head(3).iterrows():
        print(f"  SMILES: {row['Canonical SMILES']}")
        print(f"  Toxicity: {row['Toxicity Value']} ({'toxic' if row['Toxicity Value'] == 1 else 'non-toxic'})")
        print()
    
    print("\nTest set sample (first 3):")
    for idx, row in df_test.head(3).iterrows():
        print(f"  SMILES: {row['Canonical SMILES']}")
        print(f"  Toxicity: {row['Toxicity Value']} ({'toxic' if row['Toxicity Value'] == 1 else 'non-toxic'})")
        print()
    
    print("="*80)
    print("DATA PROCESSING COMPLETED")
    print("="*80)
    
    return df_train, df_test


if __name__ == "__main__":
    process_toxicity_data()

