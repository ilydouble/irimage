# 热力图对比学习模型可解释性分析工具

## 概述

本工具基于Grad-CAM技术，为训练好的对比学习热力图分类模型提供可解释性分析。通过可视化模型的注意力区域，帮助理解模型的决策过程，提高模型的可信度和临床应用价值。

## 功能特性

### 🔍 **Grad-CAM可视化**
- 支持多层特征图的Grad-CAM分析
- 自动选择关键卷积层进行可视化
- 生成高质量的注意力热力图

### 🎨 **图像叠加融合**
- 智能调整原图透明度，减少鲜艳色彩干扰
- 热力图与原图的自然融合
- 支持多种颜色映射方案

### 📊 **批量分析**
- 支持单张图像和批量图像分析
- 自动生成分析报告和统计信息
- 结构化的结果存储

### 📁 **结果管理**
- 自动创建分类存储目录
- 详细的分析结果JSON记录
- 可视化结果的系统化管理

## 安装依赖

```bash
# 基础依赖已在requirements.txt中
pip install opencv-python matplotlib seaborn

# 如果需要更好的可视化效果
pip install plotly scikit-image
```

## 使用方法

### 1. 快速开始 (推荐)

```bash
# 在项目根目录下运行
cd /path/to/IR-image
python model/run_interpretability_analysis.py
```

这将启动交互式界面，自动查找最新的模型和数据集。

### 2. 命令行使用

#### 分析单张图像
```bash
python model/interpretability_analysis.py \
    --model_path "./model/contrastive_thermal_classifier_results/run_20250826_234950/best_contrastive_encoder.pth" \
    --image_path "./dataset/datasets/thermal_classification_cropped/icas/patient_001.jpg" \
    --use_asymmetry
```

#### 批量分析目录
```bash
python model/interpretability_analysis.py \
    --model_path "./model/contrastive_thermal_classifier_results/run_20250826_234950/best_contrastive_encoder.pth" \
    --image_dir "./dataset/datasets/thermal_classification_cropped/icas/" \
    --pattern "*.jpg" \
    --use_asymmetry
```

#### 分析所有类别
```bash
python model/interpretability_analysis.py \
    --model_path "./model/contrastive_thermal_classifier_results/run_20250826_234950/best_contrastive_encoder.pth" \
    --image_dir "./dataset/datasets/thermal_classification_cropped/" \
    --pattern "*/*.jpg" \
    --use_asymmetry
```

### 3. Python API使用

```python
from model.interpretability_analysis import ThermalInterpretabilityAnalyzer

# 创建分析器
analyzer = ThermalInterpretabilityAnalyzer(
    model_path="path/to/your/model.pth",
    use_asymmetry_analysis=True
)

# 分析单张图像
result = analyzer.analyze_single_image("path/to/image.jpg")

# 批量分析
results = analyzer.batch_analyze("path/to/image/directory", "*.jpg")
```

## 输出结果

### 目录结构
```
dataset/datasets/interpretability_analysis/
├── gradcam_heatmaps/           # Grad-CAM热力图
│   ├── patient_001_backbone_layer4_1_conv2_ICAS_heatmap.png
│   └── ...
├── overlay_images/             # 叠加可视化图像
│   ├── patient_001_backbone_layer4_1_conv2_ICAS_overlay.png
│   ├── patient_001_backbone_layer4_1_conv2_ICAS_overlay_plt.png
│   └── ...
├── feature_maps/               # 特征图统计
├── analysis_results.json       # 详细分析结果
└── summary_report.txt          # 汇总报告
```

### 结果文件说明

#### 1. Grad-CAM热力图 (`gradcam_heatmaps/`)
- 纯热力图可视化
- 显示模型关注的区域
- 使用jet颜色映射

#### 2. 叠加图像 (`overlay_images/`)
- `*_overlay.png`: OpenCV版本的叠加图像
- `*_overlay_plt.png`: Matplotlib版本的叠加图像 (推荐)
- 原图透明度: 30%，热力图透明度: 70%

#### 3. 分析结果 (`analysis_results.json`)
```json
{
  "image_path": "path/to/image.jpg",
  "predicted_class": 1,
  "confidence": 0.8542,
  "gradcam_results": {
    "backbone.layer4.1.conv2": {
      "heatmap_path": "path/to/heatmap.png",
      "overlay_path": "path/to/overlay.png",
      "feature_statistics": {
        "max_activation": 1.0,
        "mean_activation": 0.3245,
        "std_activation": 0.2156,
        "activation_area": 0.1234,
        "center_of_mass": [112, 89]
      }
    }
  }
}
```

#### 4. 汇总报告 (`summary_report.txt`)
- 分析图像总数统计
- 预测类别分布
- 置信度统计信息
- 输出目录说明

## 技术细节

### Grad-CAM实现
- **目标层**: 自动选择ResNet的关键卷积层
  - `backbone.layer1.1.conv2`: 浅层特征 (低级纹理)
  - `backbone.layer2.1.conv2`: 中层特征 (局部模式)
  - `backbone.layer3.1.conv2`: 深层特征 (复杂模式)
  - `backbone.layer4.1.conv2`: 最深层特征 (高级语义)

### 图像处理
- **不对称分析模式**: 支持6通道输入 (左脸+右脸)
- **标准模式**: 3通道RGB输入
- **尺寸标准化**: 自动调整到模型输入尺寸

### 可视化优化
- **透明度调整**: 原图30%，热力图70%
- **颜色映射**: 使用jet colormap突出关注区域
- **分辨率**: 300 DPI高质量输出

## 参数说明

### 命令行参数
- `--model_path`: 训练好的模型文件路径
- `--image_path`: 单张图像路径
- `--image_dir`: 图像目录路径
- `--use_asymmetry`: 是否使用不对称分析模式
- `--pattern`: 图像文件匹配模式 (默认: "*.jpg")

### 配置选项
```python
# 可在代码中调整的参数
TARGET_LAYERS = [
    'backbone.layer1.1.conv2',
    'backbone.layer2.1.conv2', 
    'backbone.layer3.1.conv2',
    'backbone.layer4.1.conv2'
]

TRANSPARENCY_CONFIG = {
    'original_alpha': 0.3,    # 原图透明度
    'heatmap_alpha': 0.7      # 热力图透明度
}
```

## 应用场景

### 1. 模型调试
- 检查模型是否关注正确的解剖区域
- 识别模型的偏见和错误模式
- 验证模型的泛化能力

### 2. 临床解释
- 为医生提供模型决策的可视化解释
- 增强模型预测的可信度
- 辅助临床诊断决策

### 3. 研究分析
- 比较不同模型的注意力模式
- 分析特征层次的语义信息
- 评估模型的解释性能力

## 注意事项

1. **模型兼容性**: 确保模型是使用`train_thermal_classifier3.py`训练的对比学习模型
2. **内存使用**: 批量分析大量图像时注意内存使用
3. **GPU支持**: 建议使用GPU加速分析过程
4. **图像质量**: 输入图像质量会影响可视化效果

## 故障排除

### 常见问题

1. **模型加载失败**
   ```
   解决方案: 检查模型路径和模型架构是否匹配
   ```

2. **CUDA内存不足**
   ```
   解决方案: 减少批量大小或使用CPU模式
   ```

3. **图像处理错误**
   ```
   解决方案: 检查图像格式和路径是否正确
   ```

4. **可视化效果不佳**
   ```
   解决方案: 调整透明度参数或尝试不同的颜色映射
   ```

## 更新日志

- **v1.0**: 初始版本，支持基础Grad-CAM分析
- **v1.1**: 添加批量分析和报告生成功能
- **v1.2**: 优化图像叠加效果和透明度处理

---

如有问题或建议，请联系开发团队或提交Issue。
