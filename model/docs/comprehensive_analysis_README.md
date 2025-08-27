# 全面特征重要性统计分析工具

## 概述

这是一个专为热力图ICAS分类模型设计的全面可解释性分析工具，通过统计分析所有图像的Grad-CAM结果，深入理解模型的特征重要性和决策模式。

## 🎯 分析目标

### 1. **特征层重要性分析**
- 比较不同网络层的激活强度
- 评估各层对分类决策的贡献度
- 识别最关键的特征提取层

### 2. **位置模式分析**
- 统计激活区域的空间分布
- 分析注意力中心的位置模式
- 识别模型关注的解剖区域

### 3. **类别差异分析**
- 比较ICAS和Non-ICAS的特征差异
- 进行统计显著性检验
- 识别具有区分性的特征

### 4. **预测性能分析**
- 评估模型的整体准确性
- 分析不同类别的预测表现
- 关联特征重要性与预测准确性

## 🚀 快速开始

### 一键完整分析 (推荐)

```bash
# 在项目根目录下运行
cd /path/to/IR-image
python model/run_full_analysis.py
```

这将自动执行：
1. 单图Grad-CAM分析
2. 全面统计分析
3. 综合报告生成

### 单独运行统计分析

```bash
python model/comprehensive_feature_analysis.py
```

### Python API使用

```python
from model.comprehensive_feature_analysis import FeatureImportanceAnalyzer

# 创建分析器
analyzer = FeatureImportanceAnalyzer()

# 运行全面分析
analyzer.run_comprehensive_analysis(
    model_path="path/to/model.pth",  # 可选，自动查找
    dataset_dir="path/to/dataset"    # 可选，自动查找
)
```

## 📊 分析内容

### 1. 层重要性指标

#### **激活强度指标**
- `avg_max_activation`: 平均最大激活值
- `avg_mean_activation`: 平均激活值
- `avg_activation_area`: 平均激活区域比例

#### **一致性指标**
- `activation_consistency`: 激活一致性 (1 - CV)
- `accuracy_correlation`: 与预测准确性的相关性

#### **区分性指标**
- `max_activation_diff`: ICAS与Non-ICAS的激活差异
- `significant_differences`: 统计显著的差异特征

### 2. 位置模式分析

#### **空间分布**
- `mean_center`: 激活中心的平均位置
- `std_center`: 激活中心的标准差
- `center_spread`: 位置分散程度

#### **聚类分析**
- `cluster_centers`: K-means聚类中心
- `cluster_labels`: 样本聚类标签

### 3. 统计检验

#### **类别差异检验**
- t检验比较ICAS和Non-ICAS组
- p值 < 0.05 标记为显著差异
- 效应量计算

## 📁 输出结果

### 目录结构
```
dataset/datasets/feature_importance_analysis/
├── statistics/                          # 统计数据
├── visualizations/                      # 可视化图表
│   ├── layer_importance.png            # 层重要性对比
│   ├── position_patterns.png           # 位置模式分析
│   ├── class_differences.png           # 类别差异分析
│   ├── prediction_analysis.png         # 预测性能分析
│   └── comprehensive_heatmap.png       # 综合特征热力图
├── heatmap_clusters/                    # 热力图聚类
├── layer_comparisons/                   # 层间比较
├── comprehensive_analysis_results.json  # 详细结果数据
└── comprehensive_analysis_report.txt    # 文本分析报告
```

### 最终综合报告
```
dataset/datasets/final_analysis_report/
├── executive_summary.md                 # 执行摘要 (推荐阅读)
├── single_image_summary.txt            # 单图分析汇总
├── statistical_analysis_report.txt     # 统计分析报告
└── key_visualizations/                 # 关键图表
    ├── layer_importance.png
    ├── class_differences.png
    ├── comprehensive_heatmap.png
    └── prediction_analysis.png
```

## 📈 关键可视化图表

### 1. 层重要性对比图
- **内容**: 4个子图显示不同层的重要性指标
- **用途**: 识别最重要的特征提取层
- **解读**: 数值越高表示该层越重要

### 2. 位置模式分析图
- **内容**: 激活中心的空间分布和聚类
- **用途**: 理解模型关注的解剖区域
- **解读**: 聚集区域表示模型一致关注的位置

### 3. 类别差异分析图
- **内容**: ICAS和Non-ICAS在各层的特征差异
- **用途**: 识别具有区分性的特征层
- **解读**: 带*号表示统计显著差异 (p<0.05)

### 4. 综合特征热力图
- **内容**: 所有层和指标的标准化热力图
- **用途**: 全局比较各层的综合重要性
- **解读**: 颜色越深表示该层该指标越重要

## 🔍 分析层级

### ResNet层级结构
- `backbone.layer1.1.conv2`: **浅层特征** (边缘、纹理)
- `backbone.layer2.1.conv2`: **中层特征** (局部模式)
- `backbone.layer3.1.conv2`: **深层特征** (复杂模式)
- `backbone.layer4.1.conv2`: **最深层特征** (高级语义)

### 特征层次解释
1. **浅层**: 检测基本的边缘和纹理信息
2. **中层**: 组合基本特征形成局部模式
3. **深层**: 识别复杂的解剖结构
4. **最深层**: 提取高级语义特征用于分类

## 📊 统计分析方法

### 1. 描述性统计
- 均值、标准差、分位数
- 激活强度分布分析
- 位置分布统计

### 2. 推断性统计
- 独立样本t检验
- 效应量计算 (Cohen's d)
- 多重比较校正

### 3. 聚类分析
- K-means聚类
- 主成分分析 (PCA)
- 轮廓系数评估

### 4. 相关性分析
- Pearson相关系数
- 特征重要性与预测准确性关联
- 层间特征相关性

## 🎯 应用场景

### 1. 模型调试
- **问题**: 模型预测不准确
- **分析**: 检查哪些层的特征提取有问题
- **解决**: 针对性调整网络结构或训练策略

### 2. 临床解释
- **问题**: 医生需要理解模型决策依据
- **分析**: 展示模型关注的解剖区域
- **解决**: 提供可视化证据支持诊断

### 3. 模型优化
- **问题**: 提高模型性能
- **分析**: 识别最重要的特征层
- **解决**: 重点优化关键层的设计

### 4. 研究分析
- **问题**: 理解ICAS的影像学特征
- **分析**: 比较不同类别的特征差异
- **解决**: 发现新的诊断标志物

## ⚙️ 配置选项

### 分析参数
```python
# 可在代码中调整的参数
ANALYSIS_CONFIG = {
    'target_layers': [
        'backbone.layer1.1.conv2',
        'backbone.layer2.1.conv2', 
        'backbone.layer3.1.conv2',
        'backbone.layer4.1.conv2'
    ],
    'clustering_params': {
        'n_clusters': 3,
        'random_state': 42
    },
    'statistical_tests': {
        'alpha': 0.05,  # 显著性水平
        'correction': 'bonferroni'  # 多重比较校正
    }
}
```

### 可视化参数
```python
VISUALIZATION_CONFIG = {
    'figure_size': (15, 12),
    'dpi': 300,
    'color_scheme': 'YlOrRd',
    'font_size': 12
}
```

## 🔧 故障排除

### 常见问题

1. **内存不足**
   ```
   解决方案: 减少批量分析的图像数量，或使用更大内存的机器
   ```

2. **分析时间过长**
   ```
   解决方案: 使用GPU加速，或先用小样本测试
   ```

3. **可视化显示异常**
   ```
   解决方案: 检查matplotlib后端设置，确保支持中文字体
   ```

4. **统计结果不显著**
   ```
   解决方案: 增加样本量，或检查数据质量
   ```

## 📚 结果解读指南

### 层重要性解读
- **高激活强度**: 该层对输入敏感，提取丰富特征
- **高一致性**: 该层在不同样本间表现稳定
- **显著类别差异**: 该层有助于区分ICAS和Non-ICAS

### 位置模式解读
- **集中分布**: 模型关注特定解剖区域
- **分散分布**: 模型关注多个区域或不够聚焦
- **聚类模式**: 不同类型样本的关注区域差异

### 预测性能解读
- **整体准确率**: 模型的总体性能
- **类别准确率**: 各类别的检测能力
- **混淆矩阵**: 具体的分类错误模式

## 📞 技术支持

如有问题或建议，请：
1. 查看详细的错误日志
2. 检查输入数据格式
3. 参考故障排除指南
4. 联系开发团队

---

*本工具为热力图ICAS分类模型可解释性分析的专业工具，旨在提高模型的可信度和临床应用价值。*
