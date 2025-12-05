#!/usr/bin/env python3
"""
使用checkpoint-140预测virtual library的mRNA转染效率
对于efficiency score >= 7的分子生成推理理由
"""

import json
import torch
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import warnings
warnings.filterwarnings('ignore')


def get_rdkit_features(smiles):
    """提取RDKit分子特征"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        features = {
            'mw': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'tpsa': Descriptors.TPSA(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'aromatic_rings': Descriptors.NumAromaticRings(mol),
            'total_atoms': mol.GetNumAtoms()
        }
        return features
    except:
        return None


def extract_score(text):
    """从模型输出提取效率分数"""
    patterns = [
        r'[Ee]fficiency [Ss]core[:\s]+(\d+)',
        r'[Ss]core[:\s]+(\d+)',
        r'is (\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            score = int(match.group(1))
            return max(1, min(10, score))
    return None


def predict_efficiency(model, tokenizer, smiles, rdkit_features, generate_reason=False):
    """预测效率分数，可选生成理由 - 使用和训练时完全一致的格式"""
    
    # 使用和训练数据完全一致的格式
    user_prompt = f"What is the predicted score for this molecular structure: {smiles}? Molecular properties: MW={rdkit_features['mw']:.1f}Da, LogP={rdkit_features['logp']:.2f}, H-bond donors={rdkit_features['hbd']}, H-bond acceptors={rdkit_features['hba']}, TPSA={rdkit_features['tpsa']:.1f}Ų, Rotatable bonds={rdkit_features['rotatable_bonds']}, Aromatic rings={rdkit_features['aromatic_rings']}, Total atoms={rdkit_features['total_atoms']}"
    
    if generate_reason:
        # 请求详细推理
        user_prompt += " Please also explain your reasoning for this score."
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant specialized in drug discovery and molecular analysis. You can predict molecular scores based on their SMILES structures."},
        {"role": "user", "content": user_prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    max_tokens = 256 if generate_reason else 128
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    score = extract_score(response)
    
    if score is None:
        score = 5  # 默认值
    
    return score, response if generate_reason else ""


def main():
    print("=" * 80)
    print("Virtual Library mRNA转染效率预测")
    print("=" * 80)
    
    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    df = pd.read_excel('data_ann/virtual library-2500.xlsx')
    print(f"  加载了 {len(df)} 个分子")
    
    # 2. 加载模型
    print("\n[2/4] 加载模型 (checkpoint-140)...")
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model = PeftModel.from_pretrained(base_model, "qwen_multitask_rdkit_checkpoints/checkpoint-140")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct", trust_remote_code=True)
    print("  模型加载完成")
    
    # 3. 预测所有分子
    print(f"\n[3/4] 预测效率分数...")
    results = []
    failed_count = 0
    
    for idx, row in df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"  进度: {idx + 1}/{len(df)}")
        
        smiles = row['smiles']
        
        # 获取RDKit特征
        rdkit_features = get_rdkit_features(smiles)
        if rdkit_features is None:
            failed_count += 1
            continue
        
        # 预测效率分数（不生成理由）
        score, _ = predict_efficiency(model, tokenizer, smiles, rdkit_features, generate_reason=False)
        
        result = {
            'number': int(row['Number']),
            'smiles': smiles,
            'efficiency_score': score,
            'rdkit_features': rdkit_features
        }
        results.append(result)
    
    print(f"  完成预测: {len(results)} 个成功, {failed_count} 个失败")
    
    # 4. 对高分分子生成理由
    print(f"\n[4/4] 为高分分子生成理由...")
    high_score_results = [r for r in results if r['efficiency_score'] >= 7]
    print(f"  发现 {len(high_score_results)} 个效率≥7的分子")
    
    for idx, result in enumerate(high_score_results):
        print(f"  生成理由: {idx + 1}/{len(high_score_results)}")
        _, reasoning = predict_efficiency(
            model, tokenizer, 
            result['smiles'], 
            result['rdkit_features'], 
            generate_reason=True
        )
        result['reasoning'] = reasoning
    
    # 按效率分数降序排序
    results_sorted = sorted(results, key=lambda x: x['efficiency_score'], reverse=True)
    
    # 保存结果
    output_data = []
    for r in results_sorted:
        output_item = {
            'number': r['number'],
            'smiles': r['smiles'],
            'efficiency_score': r['efficiency_score'],
            'molecular_weight': r['rdkit_features']['mw'],
            'logp': r['rdkit_features']['logp']
        }
        
        # 只有≥7分的才有reasoning字段
        if 'reasoning' in r:
            output_item['reasoning'] = r['reasoning']
        
        output_data.append(output_item)
    
    output_file = 'data_ann/result_new.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # 显示统计
    print("\n" + "=" * 80)
    print("预测完成!")
    print("=" * 80)
    print(f"总分子数:        {len(results_sorted)}")
    print(f"效率≥7分:        {len([r for r in results_sorted if r['efficiency_score'] >= 7])}")
    print(f"效率≥8分:        {len([r for r in results_sorted if r['efficiency_score'] >= 8])}")
    print(f"效率≥9分:        {len([r for r in results_sorted if r['efficiency_score'] >= 9])}")
    print(f"\n分数分布:")
    for score in range(10, 0, -1):
        count = len([r for r in results_sorted if r['efficiency_score'] == score])
        if count > 0:
            print(f"  分数 {score:2d}: {count:4d} 个分子")
    print(f"\n结果已保存: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
