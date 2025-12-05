#!/usr/bin/env python3
"""
使用RDKit增强训练数据
为每个SMILES添加分子描述符，帮助模型更好地理解分子结构
"""

import json
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
from tqdm import tqdm

def calculate_molecular_features(smiles: str) -> dict:
    """计算关键分子描述符"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        features = {
            'molecular_weight': round(Descriptors.MolWt(mol), 1),
            'logP': round(Crippen.MolLogP(mol), 2),  # 脂溶性
            'hbd': Lipinski.NumHDonors(mol),  # 氢键供体
            'hba': Lipinski.NumHAcceptors(mol),  # 氢键受体
            'tpsa': round(Descriptors.TPSA(mol), 1),  # 极性表面积
            'rotatable_bonds': Lipinski.NumRotatableBonds(mol),
            'aromatic_rings': Lipinski.NumAromaticRings(mol),
            'num_atoms': mol.GetNumAtoms(),
        }
        return features
    except Exception as e:
        print(f"  Error calculating features for {smiles}: {e}")
        return None

def create_feature_description(features: dict) -> str:
    """创建人类可读的特征描述"""
    desc = f"Molecular properties: MW={features['molecular_weight']}Da, LogP={features['logP']}, "
    desc += f"H-bond donors={features['hbd']}, H-bond acceptors={features['hba']}, "
    desc += f"TPSA={features['tpsa']}Ų, Rotatable bonds={features['rotatable_bonds']}, "
    desc += f"Aromatic rings={features['aromatic_rings']}, Total atoms={features['num_atoms']}"
    return desc

def enhance_training_data(input_file: str, output_file: str, task_type: str):
    """增强训练数据"""
    print(f"\n{'='*80}")
    print(f"Enhancing {task_type} data with RDKit features")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"{'='*80}\n")
    
    enhanced_data = []
    failed_count = 0
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    for line in tqdm(lines, desc=f"Processing {task_type}"):
        item = json.loads(line.strip())
        messages = item['messages']
        
        # 提取SMILES（从user message中）
        user_content = messages[1]['content']
        
        # 提取SMILES
        if task_type == "efficiency":
            smiles = user_content.split(": ")[1].strip().rstrip("?")
        else:  # toxicity
            smiles = user_content.split("? ")[1].strip()
        
        # 计算分子特征
        features = calculate_molecular_features(smiles)
        
        if features is None:
            failed_count += 1
            # 保留原始数据
            enhanced_data.append(item)
            continue
        
        # 创建特征描述
        feature_desc = create_feature_description(features)
        
        # 增强user message
        if task_type == "efficiency":
            new_user_content = f"What is the predicted score for this molecular structure: {smiles}? {feature_desc}"
        else:  # toxicity
            new_user_content = f"Is this molecular structure toxic? {smiles} {feature_desc}"
        
        # 更新messages
        messages[1]['content'] = new_user_content
        enhanced_data.append({'messages': messages})
    
    # 保存增强后的数据
    with open(output_file, 'w') as f:
        for item in enhanced_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Completed!")
    print(f"   Total samples: {len(lines)}")
    print(f"   Successfully enhanced: {len(lines) - failed_count}")
    print(f"   Failed: {failed_count}")
    print(f"   Output saved to: {output_file}\n")

def main():
    print("\n" + "="*80)
    print("🔬 RDKit Feature Enhancement for Training Data")
    print("="*80)
    
    # Check RDKit installation
    try:
        mol = Chem.MolFromSmiles("CCO")
        print("✓ RDKit is installed and working\n")
    except Exception as e:
        print(f"❌ RDKit error: {e}")
        print("Please install: pip install rdkit --break-system-packages")
        return
    
    # Enhance efficiency training data
    enhance_training_data(
        input_file="data/efficiency_train_data.jsonl",
        output_file="data/efficiency_train_data_rdkit.jsonl",
        task_type="efficiency"
    )
    
    # Enhance efficiency test data
    enhance_training_data(
        input_file="data/efficiency_test_data.jsonl",
        output_file="data/efficiency_test_data_rdkit.jsonl",
        task_type="efficiency"
    )
    
    # Enhance toxicity training data
    enhance_training_data(
        input_file="data/toxic_data/toxicity_train_data.jsonl",
        output_file="data/toxic_data/toxicity_train_data_rdkit.jsonl",
        task_type="toxicity"
    )
    
    # Enhance toxicity test data
    enhance_training_data(
        input_file="data/toxic_data/toxicity_test_data.jsonl",
        output_file="data/toxic_data/toxicity_test_data_rdkit.jsonl",
        task_type="toxicity"
    )
    
    print("\n" + "="*80)
    print("🎉 All data enhanced with RDKit features!")
    print("="*80)
    print("\nNext steps:")
    print("1. Use the *_rdkit.jsonl files for training")
    print("2. Modify train_multitask_with_eval.py to use new data files")
    print("3. Train a new model and compare results")

if __name__ == "__main__":
    main()
