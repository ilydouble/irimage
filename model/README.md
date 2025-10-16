# Model Directory - 热力图ICAS分类模型工具集

## 📋 目录概览

本目录包含了完整的热力图ICAS分类模型训练、分析和预测工具链，涵盖从传统机器学习到深度学习的多种方法。

## 🎯 核心功能模块

### 1. 🧠 模型训练脚本

| 脚本名称 | 技术栈 | 主要特点 | 推荐场景 |
|---------|--------|----------|----------|
| **train_cnn_classifier.py** | 深度学习CNN | ResNet/EfficientNet + Focal Loss | 标准深度学习分类 |
| **train_feature_ml_classifier.py** | 传统机器学习 | 手工特征 + 多种ML算法 | 特征工程研究 |
| **train_multimodal_classifier.py** | 多模态融合 | 图像特征 + 临床数据 | 临床应用场景 |
| **train_contrastive_classifier.py** | 对比学习 | 两阶段训练：对比学习 + 分类 | 数据稀缺场景 |
| **train_contrastive_mask_classifier.py** | 对比学习 + Mask | 智能人脸Mask + Attention | 高精度要求 |
| **train_contrastive_split_classifier.py** | 对比学习 + 分割 | 改进数据集分割策略 | 实验对比 |

### 2. 🔍 模型分析脚本

| 脚本名称 | 功能 | 输出 | 用途 |
|---------|------|------|------|
| **interpretability_analysis.py** | Grad-CAM可解释性分析 | 热力图、叠加图像 | 理解模型决策 |
| **run_interpretability_analysis.py** | 可解释性分析运行器 | 批量分析结果 | 快速分析工具 |
| **comprehensive_feature_analysis.py** | 全面特征重要性统计 | 统计报告、可视化 | 深度分析 |

### 3. 🎯 模型预测脚本

| 脚本名称 | 功能 | 输入 | 输出 |
|---------|------|------|------|
| **predict_yolo11.py** | YOLOv11人脸检测预测 | 热力图像 | 检测框、分割掩码 |

### 4. 🎤 语音处理脚本

| 脚本名称 | 功能 | 特点 | 用途 |
|---------|------|------|------|
| **voice_asr.py** | 阿里云语音识别 | 支持OSS/本地文件 | 语音数据处理 |
| **extract_asr_simple.py** | ASR结果提取 | 交互式界面 | 结果整理 |

### 5. 🛠️ 辅助工具脚本

| 脚本名称 | 功能 | 用途 |
|---------|------|------|
| **train_yolo11.py** | YOLOv11模型训练 | 人脸检测模型训练 |
| **verify_mask_consistency.py** | Mask一致性验证 | 调试工具 |
| **visualize_smart_mask.py** | 智能Mask可视化 | 效果展示 |

## 🚀 快速开始

### 环境准备

```bash
# 安装基础依赖
pip install torch torchvision torchaudio
pip install opencv-python matplotlib seaborn
pip install scikit-learn pandas numpy
pip install ultralytics  # YOLOv11
```

### 模型训练

#### 1. 深度学习CNN分类器 (推荐)

```bash
python model/train_cnn_classifier.py
```

**特点：**
- 支持ResNet18/34/50和EfficientNet B0-B3
- 集成Focal Loss处理类别不平衡
- 自动早停和模型保存
- 完整的训练可视化

#### 2. 对比学习分类器 (高精度)

```bash
python model/train_contrastive_classifier.py
```

**特点：**
- 两阶段训练：对比学习预训练 + 分类微调
- 支持不对称分析（左右脸分别处理）
- 适合小样本学习

#### 3. 多模态融合分类器 (临床应用)

```bash
python model/train_multimodal_classifier.py
```

**特点：**
- 结合图像特征和临床数据
- 从数据库自动提取患者信息
- 适合实际临床部署

#### 4. 传统机器学习分类器 (特征研究)

```bash
python model/train_feature_ml_classifier.py
```

**特点：**
- 手工特征提取（纹理、形状、统计特征）
- 多种ML算法对比（RF、SVM、XGBoost等）
- 特征重要性分析

### 模型分析

#### 1. 快速可解释性分析

```bash
python model/run_interpretability_analysis.py
```

**功能：**
- 自动查找最新训练的模型
- 生成Grad-CAM热力图
- 批量分析多张图像
- 交互式操作界面

#### 2. 全面特征重要性分析

```bash
python model/comprehensive_feature_analysis.py
```

**功能：**
- 统计所有图像的特征重要性
- 层级重要性排名
- 类别差异分析
- 位置模式统计

### 模型预测

#### 1. YOLOv11人脸检测

```bash
python model/predict_yolo11.py
```

**配置文件：** 修改脚本中的config字典
```python
config = {
    'model_path': 'path/to/your/model.pt',
    'source_dir': 'path/to/images',
    'output_dir': 'path/to/output'
}
```

### 语音处理

#### 1. 语音识别

```bash
python model/voice_asr.py
```

**支持模式：**
- OSS文件识别
- 本地文件识别（自动上传）
- 批量目录处理

#### 2. 结果提取

```bash
python model/extract_asr_simple.py
```

## 📊 训练结果目录结构

```
model/
├── contrastive_thermal_classifier_results/    # 对比学习模型结果
│   └── run_YYYYMMDD_HHMMSS/
│       ├── best_classifier.pth               # 最佳分类器
│       ├── best_contrastive_encoder.pth      # 最佳编码器
│       ├── training_history.json             # 训练历史
│       └── config.json                       # 训练配置
├── cnn_thermal_classifier_results/           # CNN模型结果
├── multimodal_thermal_classifier_results/    # 多模态模型结果
└── feature_ml_classifier_results/            # 传统ML结果
```

## 🔬 分析结果目录结构

```
dataset/datasets/
├── interpretability_analysis/                # 可解释性分析结果
│   ├── gradcam_heatmaps/                     # Grad-CAM热力图
│   ├── overlay_images/                       # 叠加可视化
│   ├── analysis_results.json                 # 详细结果
│   └── summary_report.txt                    # 汇总报告
├── feature_importance_analysis/              # 特征重要性分析
│   ├── statistics/                           # 统计数据
│   ├── visualizations/                       # 可视化图表
│   └── comprehensive_analysis_report.txt     # 综合报告
└── quick_analysis/                           # 快速分析结果
```

## ⚙️ 配置说明

### 数据路径配置

大多数脚本使用以下默认路径：
- **训练数据：** `./dataset/datasets/thermal_classification_cropped`
- **数据库：** `./web/database/patientcare.db`
- **输出目录：** `./model/{script_name}_results`

### 模型参数配置

每个训练脚本都有内置的配置字典，可根据需要修改：

```python
# 示例：CNN分类器配置
config = {
    'model_name': 'resnet18',
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 100,
    'patience': 15
}
```

## 🔧 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减小batch_size
   config['batch_size'] = 16  # 或更小
   ```

2. **模型文件未找到**
   ```bash
   # 检查模型路径
   ls model/contrastive_thermal_classifier_results/
   ```

3. **数据集路径错误**
   ```bash
   # 确认数据集存在
   ls dataset/datasets/thermal_classification_cropped/
   ```

### 调试工具

- **Mask一致性验证：** `python model/verify_mask_consistency.py`
- **Mask效果可视化：** `python model/visualize_smart_mask.py`

## 📚 技术文档

### 核心技术

1. **Focal Loss：** 处理类别不平衡问题
2. **对比学习：** 提高特征表示质量
3. **Grad-CAM：** 模型可解释性分析
4. **智能Mask：** 基于内容的人脸区域检测
5. **多模态融合：** 图像+临床数据结合

### 模型架构

- **CNN骨干网络：** ResNet、EfficientNet
- **对比学习编码器：** 基于ResNet的Siamese网络
- **注意力机制：** 自适应特征加权
- **多模态融合：** 特征级和决策级融合

## 🔄 工作流程建议

### 1. 新手入门流程
```bash
# 1. 训练基础CNN模型
python model/train_cnn_classifier.py

# 2. 分析模型可解释性
python model/run_interpretability_analysis.py

# 3. 查看分析结果
ls dataset/datasets/interpretability_analysis/
```

### 2. 高级研究流程
```bash
# 1. 对比学习预训练
python model/train_contrastive_classifier.py

# 2. 全面特征分析
python model/comprehensive_feature_analysis.py

# 3. 多模态融合
python model/train_multimodal_classifier.py
```

### 3. 生产部署流程
```bash
# 1. 训练最佳模型
python model/train_contrastive_mask_classifier.py

# 2. 模型验证
python model/run_interpretability_analysis.py

# 3. 性能评估
python model/comprehensive_feature_analysis.py
```

## 📁 备份和历史文件

### backup_original_classifiers/ 目录
包含了原始训练脚本的备份版本：
- `train_thermal_classifier.py` → `train_cnn_classifier.py`
- `train_thermal_classifier1.py` → `train_feature_ml_classifier.py`
- `train_thermal_classifier2.py` → `train_multimodal_classifier.py`
- `train_thermal_classifier3.py` → `train_contrastive_classifier.py`
- `train_thermal_classifier4.py` → `train_contrastive_mask_classifier.py`
- `train_thermal_classifier5.py` → `train_contrastive_split_classifier.py`

### docs/ 目录
包含详细的技术文档：
- **comprehensive_analysis_README.md** - 全面特征重要性分析工具文档
- **interpretability_analysis_README.md** - 可解释性分析工具文档
- **mask_consistency_fix.md** - Mask一致性修复技术文档
- **multimodal_features_documentation.md** - 多模态特征详细说明
- **smart_mask_summary.md** - 智能Mask功能技术总结
- **thermal_classification_methods_report.md** - 分类方法对比报告

## 🔍 脚本功能详细说明

### 训练脚本详细对比

#### 1. train_cnn_classifier.py - 深度学习CNN分类器
**技术栈：** PyTorch + ResNet/EfficientNet + Focal Loss
**特点：**
- 支持6种预训练模型（ResNet18/34/50, EfficientNet B0-B3）
- 集成Focal Loss处理类别不平衡
- 自动早停机制（patience=15）
- 完整的训练可视化和指标记录
- 支持CUDA和MPS（Apple Silicon）

**适用场景：** 标准深度学习分类任务，作为基线模型

#### 2. train_feature_ml_classifier.py - 传统机器学习分类器
**技术栈：** OpenCV + scikit-learn + 手工特征工程
**特点：**
- 26维手工特征（温度、纹理、形状、统计特征）
- 8种ML算法对比（RF、SVM、XGBoost、LightGBM等）
- Focal Loss包装器处理类别不平衡
- 特征重要性分析和可视化
- 训练速度快（约30秒）

**适用场景：** 特征工程研究、快速原型验证、可解释性要求高

#### 3. train_multimodal_classifier.py - 多模态融合分类器
**技术栈：** 图像特征 + 临床数据 + 机器学习
**特点：**
- 324维图像特征 + 13维临床特征
- 自动从数据库提取患者临床信息
- 特征级融合策略
- 最佳性能（74.83%准确率）
- 适合实际临床部署

**适用场景：** 临床应用、多模态数据融合研究

#### 4. train_contrastive_classifier.py - 对比学习分类器
**技术栈：** 对比学习 + 两阶段训练
**特点：**
- 两阶段训练：对比学习预训练 + 分类微调
- 支持不对称分析（左右脸分别处理）
- 自监督学习，适合小样本场景
- 强泛化能力

**适用场景：** 数据稀缺、需要强泛化能力的场景

#### 5. train_contrastive_mask_classifier.py - 对比学习+Mask分类器
**技术栈：** 对比学习 + 智能人脸Mask + Attention机制
**特点：**
- 智能人脸Mask（基于内容检测）
- 空间Attention机制
- 双重策略：预处理Mask + 运行时Attention
- 最高可解释性

**适用场景：** 高精度要求、需要可解释性的研究

#### 6. train_contrastive_split_classifier.py - 对比学习+数据分割分类器
**技术栈：** 对比学习 + 改进数据分割策略
**特点：**
- 改进的数据集分割策略
- 确保实验可重复性
- 精确的数据控制

**适用场景：** 科研实验、方法对比研究

### 分析脚本详细说明

#### 1. interpretability_analysis.py - 核心可解释性分析引擎
**功能：**
- Grad-CAM算法实现
- 多层特征可视化（layer1-layer4）
- 热力图生成和叠加可视化
- 特征统计计算（激活强度、重心等）
- 支持6通道和3通道模式自动检测

#### 2. run_interpretability_analysis.py - 可解释性分析运行器
**功能：**
- 自动查找最新训练模型
- 智能模型模式检测（6通道/3通道）
- 交互式和自定义分析模式
- 批量图像处理
- 结果汇总和报告生成

#### 3. comprehensive_feature_analysis.py - 全面特征重要性统计
**功能：**
- 统计所有图像的Grad-CAM结果
- 层级重要性排名分析
- 类别差异统计检验
- 位置模式聚类分析
- 预测性能关联分析
- 限制分析图像数量（最多200张，避免过长时间）

### 辅助工具详细说明

#### 1. voice_asr.py - 阿里云语音识别
**功能：**
- 支持OSS文件识别
- 本地文件自动上传识别
- 批量目录处理
- 智能分轨和自适应采样率
- 结果保存为JSON和TXT格式

#### 2. extract_asr_simple.py - ASR结果提取工具
**功能：**
- 交互式文件搜索和提取
- 自动识别ASR结果文件
- 按类型组织输出（json_results/, txt_reports/）
- 生成处理摘要

#### 3. predict_yolo11.py - YOLOv11预测工具
**功能：**
- 批量图像人脸检测
- 支持检测框和分割掩码输出
- 结果自动组织和保存
- 可配置置信度阈值

#### 4. face_detect.py - MediaPipe人脸检测
**功能：**
- 人脸关键点检测（468个landmark）
- 面部区域分析（左眼、右眼、鼻子、嘴巴等）
- GLCM纹理特征提取
- 检测框可视化

#### 5. seg_all.py - SAM图像分割
**功能：**
- 基于SAM模型的图像分割
- 多点提示分割
- 掩码优化和后处理
- 批量图像处理

#### 6. verify_mask_consistency.py - Mask一致性验证
**功能：**
- 验证训练脚本中mask使用的一致性
- 检查对比学习和分类阶段的mask逻辑
- 代码质量检查和建议

#### 7. visualize_smart_mask.py - 智能Mask可视化
**功能：**
- 可视化不同mask方法的效果
- 对比智能检测、椭圆、矩形、自适应mask
- 效果展示和调试

## 🎯 使用建议和最佳实践

### 新手入门建议
1. **从CNN分类器开始**：`train_cnn_classifier.py`提供稳定的基线
2. **理解可解释性**：使用`run_interpretability_analysis.py`查看模型关注区域
3. **对比不同方法**：尝试传统ML方法了解特征重要性

### 研究人员建议
1. **对比学习探索**：使用`train_contrastive_classifier.py`探索自监督学习
2. **多模态融合**：结合临床数据提升性能
3. **深度分析**：使用`comprehensive_feature_analysis.py`进行全面分析

### 临床应用建议
1. **多模态部署**：`train_multimodal_classifier.py`最适合临床环境
2. **可解释性要求**：使用mask和attention机制提供决策解释
3. **性能监控**：定期使用分析工具验证模型性能

---

**注意：** 运行脚本前请确保在项目根目录下，并且已正确配置数据路径和模型参数。详细的技术文档请参考docs/目录下的相关文件。
