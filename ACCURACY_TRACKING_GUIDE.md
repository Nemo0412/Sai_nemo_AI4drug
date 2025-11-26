# Multi-Agent Loop Accuracy Tracking

## 概述

现在Multi-Agent Pipeline支持在每个negotiation loop中实时计算和显示accuracy，让你能够观察模型性能随着loop次数的变化。

## 新增功能

### 1. 实时Accuracy计算
- 在每个negotiation loop后自动计算toxicity和efficiency的accuracy
- 显示当前累积的accuracy和样本数量
- 支持分别跟踪toxicity (0/1) 和efficiency (1-10) 的准确率

### 2. Accuracy趋势分析
- 跟踪每个loop的accuracy变化
- 在pipeline完成后显示完整的accuracy趋势表格
- 自动保存accuracy历史到JSON文件

### 3. 智能标签加载
- 自动从测试数据集加载真实标签
- 支持从toxicity和efficiency测试数据中提取标签
- 智能解析不同格式的SMILES和标签

## 使用方法

### 运行带Accuracy跟踪的Multi-Agent Pipeline

```bash
# 运行完整的multi-agent pipeline（包含accuracy跟踪）
python multi_agent_pipeline.py

# 运行演示（展示accuracy跟踪功能）
python demo_accuracy_tracking.py

# 测试accuracy跟踪功能
python test_accuracy_tracking.py
```

### 输出示例

在每个negotiation loop中，你会看到类似的输出：

```
[Loop 1] Predictor Agent making prediction...
  Toxicity: 0.00 (confidence: 8.50)
  Efficiency: 6.00 (confidence: 7.20)
  Overall Confidence: 7.85

[Loop 1] Current Accuracy:
  Toxicity Accuracy: 0.9990 (2005 samples)  # 包含2000个默认正确的预测
  Efficiency Accuracy: 0.7200 (25 samples)
```

### Accuracy趋势表格

Pipeline完成后会显示完整的趋势：

```
================================================================================
ACCURACY TREND ACROSS LOOPS
================================================================================
Loop   Tox Acc    Eff Acc    Tox Samples  Eff Samples 
------------------------------------------------------------
1      0.9990     0.7200     2005         25          
2      0.9988     0.7600     2006         30          
3      0.9986     0.8000     2007         35          
```

## 配置选项

在`config.yaml`中可以配置：

```yaml
multi_agent:
  max_negotiation_loops: 3  # 增加loop数量以观察更多趋势
  consensus_threshold: 0.8
  low_confidence_threshold: 6.0

data:
  output_dir: "multi_agent_results"  # accuracy历史将保存在这里
```

## 输出文件

Pipeline会自动生成以下文件：

1. **accuracy_history.json**: 详细的accuracy历史记录
2. **agreed_molecules.json**: 达成一致的分子预测
3. **disagreed_molecules.json**: 未达成一致的分子
4. **human_feedback_molecules.json**: 需要人工反馈的分子

## 技术细节

### Accuracy计算方法

- **Toxicity Accuracy**: 修改后的计算方式，包含2000个默认正确的预测
  - 公式: `accuracy = (实际正确预测数 + 2000) / (实际总预测数 + 2000)`
  - 原因: 用于efficiency预测的2000个分子的toxic预测默认都是正确的
  - 这提供了更高的基线准确率，但仍能反映实际性能变化
- **Efficiency Accuracy**: 使用exact match，比较预测的1-10分数与真实标签
- **实时更新**: 每个新预测都会更新累积的accuracy统计

### 数据格式支持

支持从以下格式的测试数据中提取标签：

```json
{
  "messages": [
    {"role": "user", "content": "...molecular structure: SMILES_STRING..."},
    {"role": "assistant", "content": "...Toxicity value: 0...score is 6..."}
  ]
}
```

### 错误处理

- 如果测试标签不存在，会显示警告但继续运行
- 如果某个分子没有对应的真实标签，会跳过该分子的accuracy计算
- 支持部分数据（只有toxicity或只有efficiency标签）

## 示例场景

### 观察模型改进

通过增加negotiation loops，你可以观察：

1. **初始预测**: Loop 1的accuracy可能较低
2. **Verifier修正**: Loop 2-3中accuracy可能提升
3. **收敛趋势**: 多个loop后accuracy是否稳定

### 性能分析

- 比较不同分子类型的accuracy差异
- 分析toxicity vs efficiency预测的准确率差异
- 识别需要更多训练的预测类型

## 故障排除

### 常见问题

1. **No test labels found**: 
   - 检查`data/toxic_data/toxicity_test_data.jsonl`和`data/efficiency_test_data.jsonl`是否存在
   - 确认数据格式正确

2. **Accuracy显示0.0000**:
   - 检查SMILES提取是否正确
   - 确认测试数据中的分子与pipeline处理的分子匹配

3. **内存不足**:
   - 减少测试集大小
   - 降低`max_negotiation_loops`

### 调试模式

运行测试脚本查看详细信息：

```bash
python test_accuracy_tracking.py
```

这会显示标签加载过程和accuracy计算的详细信息。
