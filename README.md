# IR-Image: 基于热力图和语音的ICAS智能诊断系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

## 📋 项目概述

IR-Image是一个基于热力图和语音的ICAS（颅内动脉粥样硬化性狭窄）智能诊断系统，集成了深度学习、计算机视觉、语音识别和多模态数据融合技术，为临床医生提供高精度的ICAS风险评估工具。

### 🎯 核心功能

- **🔍 人脸检测**: 基于YOLOv11的热力图人脸区域自动检测
- **🧠 智能分类**: 多种机器学习方法的ICAS风险分类
- **📊 多模态融合**: 热力图特征与临床数据的深度融合
- **🎤 语音识别**: 集成阿里云ASR的语音数据处理和分析
- **🌐 Web应用**: 完整的患者管理和预测系统
- **📈 实时分析**: 支持批量处理和实时预测
- **☁️ 云端集成**: OSS存储和语音识别服务

### 🏆 技术亮点

- **准确率**: 多模态融合模型达到74.83%的分类准确率
- **效率**: 单张图像预测时间<1秒，语音识别实时处理
- **鲁棒性**: 支持多种图像质量和拍摄条件，智能语音降噪
- **可扩展**: 模块化设计，易于集成新算法
- **云端服务**: 集成阿里云OSS和ASR服务，支持大规模数据处理

## 🏗️ 系统架构

```
IR-Image System
├── 数据层 (Dataset)
│   ├── 热力图数据收集与标注
│   ├── 语音数据收集与处理
│   ├── 临床数据整合
│   └── 数据质量控制
├── 算法层 (Model)
│   ├── YOLOv11人脸检测
│   ├── 传统特征分类
│   ├── 多模态融合分类
│   ├── 深度学习分类
│   ├── 对比学习分类
│   └── 语音识别与分析
├── 云服务层 (Cloud)
│   ├── 阿里云OSS存储
│   ├── 阿里云ASR语音识别
│   ├── 文件上传与管理
│   └── 批量处理服务
├── 应用层 (Web)
│   ├── 患者管理系统
│   ├── 预测服务API
│   ├── 数据分析仪表板
│   └── 文件管理系统
└── 部署层 (Deployment)
    ├── 模型服务化
    ├── 性能监控
    └── 持续优化
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- Node.js 16+
- SQLite 3
- CUDA 11.8+ (可选，用于GPU加速)

### 1. 克隆项目

```bash
git clone https://github.com/your-repo/IR-image.git
cd IR-image
```

### 2. 安装Python依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# GPU版本 (可选)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. 设置Web应用

```bash
cd web

# 安装后端依赖
npm install

# 安装前端依赖
cd frontend
npm install
cd ..

# 初始化数据库
npm run init-db
npm run seed-db
```

### 4. 启动服务

```bash
# 启动后端服务 (端口 3001)
npm start

# 新终端启动前端服务 (端口 3000)
cd frontend
npm run dev
```

### 5. 访问系统

- 🌐 Web界面: http://localhost:3000
- 🔌 API接口: http://localhost:3001/api
- 📊 健康检查: http://localhost:3001/api/health

## 📁 项目结构

```
IR-image/
├── 📂 dataset/                 # 数据集管理和构建 📊
│   ├── 📄 build_datasets.py      # 数据集构建主脚本
│   ├── 📄 import_thermal_24h.py  # 24小时热力图数据导入
│   ├── 📄 import_thermal_25h.py  # 25小时热力图数据导入
│   ├── 📄 import_voice_25h.py    # 25小时语音数据导入
│   ├── 📄 build_yolo_dataset.py  # YOLO检测数据集构建
│   ├── 📄 extract_segmented_regions.py  # 分割区域提取
│   ├── 📄 README.md              # 数据集详细文档
│   └── 📂 datasets/              # 构建好的分类数据集
│       ├── thermal_classification_cropped/  # 裁剪后的分类数据
│       ├── asr_results/          # ASR识别结果（清理版）
│       ├── yolo_detection/       # YOLO检测数据集
│       └── dataset_summary.json  # 数据集统计信息
├── 📂 model/                   # 机器学习模型训练和分析 🧠
│   ├── 🧠 训练脚本 (6个)
│   │   ├── 📄 train_cnn_classifier.py           # 深度学习CNN分类器
│   │   ├── 📄 train_feature_ml_classifier.py    # 传统机器学习分类器
│   │   ├── 📄 train_multimodal_classifier.py    # 多模态融合分类器
│   │   ├── 📄 train_contrastive_classifier.py   # 对比学习分类器
│   │   ├── 📄 train_contrastive_mask_classifier.py  # 对比学习+Mask分类器
│   │   ├── 📄 train_contrastive_split_classifier.py # 对比学习+数据分割分类器
│   │   └── 📄 train_yolo11.py                   # YOLOv11人脸检测模型训练
│   ├── 🔍 分析脚本 (3个)
│   │   ├── 📄 interpretability_analysis.py      # Grad-CAM可解释性分析核心引擎
│   │   ├── 📄 run_interpretability_analysis.py  # 可解释性分析运行器
│   │   └── 📄 comprehensive_feature_analysis.py # 全面特征重要性统计
│   ├── 🎯 预测脚本 (1个)
│   │   └── 📄 predict_yolo11.py                 # YOLOv11人脸检测预测
│   ├── 🎤 语音处理脚本 (2个)
│   │   ├── 📄 voice_asr.py                      # 阿里云语音识别服务
│   │   └── 📄 extract_asr_simple.py             # ASR结果提取工具
│   ├── �️ 辅助工具脚本 (4个)
│   │   ├── 📄 seg_all.py                        # SAM图像分割
│   │   ├── 📄 verify_mask_consistency.py        # Mask一致性验证
│   │   └── 📄 visualize_smart_mask.py           # 智能Mask可视化
│   ├── 📂 backup_original_classifiers/          # 原始训练脚本备份
│   ├── 📂 docs/                                 # 技术文档 (6个文档)
│   │   ├── 📄 comprehensive_analysis_README.md  # 全面特征分析文档
│   │   ├── 📄 interpretability_analysis_README.md # 可解释性分析文档
│   │   ├── 📄 mask_consistency_fix.md           # Mask一致性修复文档
│   │   ├── 📄 multimodal_features_documentation.md # 多模态特征详细说明
│   │   ├── 📄 smart_mask_summary.md             # 智能Mask功能总结
│   │   └── 📄 thermal_classification_methods_report.md # 分类方法对比报告
│   ├── 📂 *_results/                            # 各种训练结果目录
│   └── 📄 README.md                             # 模型工具集详细文档
├── 📂 web/                     # Web应用系统 🌐
│   ├── 📄 server.js              # Express后端服务器
│   ├── 📂 frontend/              # Next.js前端应用
│   │   ├── 📂 src/pages/         # 页面组件
│   │   ├── 📂 src/components/    # UI组件
│   │   └── 📄 package.json       # 前端依赖
│   ├── 📂 routes/                # API路由定义
│   ├── 📂 services/              # 业务逻辑服务
│   ├── 📂 database/              # SQLite数据库
│   │   └── 📄 patientcare.db     # 患者数据库
│   ├── 📂 uploads/               # 文件上传目录
│   ├── 📄 package.json           # 后端依赖
│   └── 📄 README.md              # Web应用文档
├── 📂 results/                 # 统一结果输出目录 ⭐
│   ├── 📂 asr_results/           # 语音识别原始结果（待清理）
│   ├── 📂 extracted_results/     # 提取的结果文件
│   ├── 📂 oss_backups/          # OSS备份文件
│   └── 📂 logs/                 # 系统日志文件
├── 📂 docs/                    # 项目文档 📚
│   └── 📄 algorithm_pipeline_documentation.md # 算法流程详细文档
├── 📂 thermal_segmentation/    # 热力图分割相关 🔬
│   └── (分割算法和工具)
├── 📄 cleanup_oss.py           # OSS存储清理工具（完整版）
├── 📄 cleanup_oss_simple.py    # OSS存储清理工具（简化版）
├── 📄 extract_asr_results.py   # ASR结果提取工具（完整版）
├── 📄 organize_project.py      # 项目结构整理工具
├── 📄 thermal_classification_starter.py # 热力图分类教学模板
├── 📄 requirements.txt         # Python依赖清单
└── 📄 README.md               # 项目主文档（本文件）
```

## � 目录功能详解

### 📊 dataset/ - 数据集管理中心
**功能：** 数据收集、处理、构建和管理
- **数据导入脚本**：支持热力图和语音数据的批量导入
- **数据集构建**：自动化构建分类和检测数据集
- **数据统计**：提供详细的数据集统计信息
- **质量控制**：数据验证和清理工具
- **📖 详细文档**：[dataset/README.md](dataset/README.md)

### 🧠 model/ - 机器学习核心
**功能：** 模型训练、分析、预测和工具集
- **6种训练方法**：从传统ML到深度学习的完整方法链
- **3种分析工具**：可解释性分析和特征重要性统计
- **语音处理**：ASR识别和结果提取
- **辅助工具**：分割、验证、可视化等专用工具
- **技术文档**：6个详细的技术文档
- **📖 详细文档**：[model/README.md](model/README.md)

### 🌐 web/ - Web应用系统
**功能：** 用户界面和API服务
- **前端应用**：Next.js构建的现代化Web界面
- **后端服务**：Express.js API服务器
- **数据库管理**：SQLite患者数据管理
- **文件上传**：支持图像和文件上传处理
- **📖 详细文档**：[web/README.md](web/README.md)

### 📈 results/ - 统一结果输出
**功能：** 所有处理结果的统一存储
- **ASR结果**：语音识别原始输出
- **训练结果**：模型训练输出和日志
- **分析结果**：可解释性分析和特征分析结果
- **备份文件**：OSS备份和提取的文件

### 📚 docs/ - 项目文档
**功能：** 项目级别的文档和说明
- **算法文档**：详细的算法流程说明
- **架构文档**：系统架构和设计文档

### 🔬 thermal_segmentation/ - 图像分割
**功能：** 热力图像分割相关算法和工具
- **分割算法**：专用的图像分割方法
- **后处理工具**：分割结果优化和处理

### 🛠️ 根目录工具脚本
**功能：** 项目级别的管理和维护工具
- **OSS管理**：云存储清理和管理工具
- **结果提取**：ASR结果提取和整理
- **项目整理**：项目结构优化工具
- **教学模板**：热力图分类学习模板

## �🔬 算法方法

### 1. 传统特征分类
- **特征**: 温度特征、LBP纹理、形状特征
- **分类器**: LogisticRegression, SVM, RandomForest, XGBoost
- **性能**: 准确率51.75%

### 2. 多模态融合分类 ⭐
- **图像特征**: 增强LBP纹理分析 (324维)
- **临床特征**: 生理指标与衍生特征 (13维)
- **最佳模型**: LightGBM
- **性能**: 准确率74.83%, F1分数53.85%

### 3. 深度学习分类
- **架构**: 自定义CNN + ResNet预训练
- **训练策略**: 数据增强、早停、学习率调度
- **状态**: 开发中

### 4. 对比学习分类
- **方法**: 两阶段训练 (对比预训练 + 分类微调)
- **特色**: 支持面部不对称分析
- **状态**: 开发中

## 📊 性能表现

| 方法 | 最佳模型 | 准确率 | F1分数 | AUC | 训练时间 |
|------|----------|--------|--------|-----|----------|
| **多模态融合** | **LightGBM** | **74.83%** | **53.85%** | **75.46%** | **28.51s** |
| 传统特征 | LogisticRegression | 51.75% | 37.84% | 54.14% | 28.12s |
| 深度学习 | ThermalCNN | 开发中 | - | - | - |
| 对比学习 | ContrastiveNet | 开发中 | - | - | - |

## 🛠️ 使用指南

### 数据准备

```bash
# 1. 导入热力图数据
cd dataset
python import_thermal_24h.py
python import_thermal_25h.py

# 2. 导入语音数据
python import_voice_25h.py

# 3. 构建分类数据集
python build_datasets.py
```

### 模型训练

```bash
cd model

# 1. 深度学习CNN分类器（推荐入门）
python train_cnn_classifier.py

# 2. 多模态融合分类器（最佳性能）
python train_multimodal_classifier.py

# 3. 对比学习分类器（高泛化能力）
python train_contrastive_classifier.py

# 4. 对比学习+智能Mask分类器（最高可解释性）
python train_contrastive_mask_classifier.py

# 5. 传统机器学习分类器（快速验证）
python train_feature_ml_classifier.py

# 6. YOLOv11人脸检测模型
python train_yolo11.py
```

### 模型预测

```bash
# 人脸检测
python predict_yolo11.py --input path/to/thermal/image.jpg

# ICAS风险预测
python model/predict.py --image path/to/face/image.jpg --clinical clinical_data.json
```

### 语音识别服务

```bash
cd model

# 1. 单个OSS文件识别
python voice_asr.py
# 选择模式 1: OSS文件识别

# 2. 本地文件识别（自动上传）
python voice_asr.py
# 选择模式 2: 本地文件识别

# 3. 批量处理本地目录
python voice_asr.py
# 选择模式 3: 批量处理本地目录

# 4. 提取ASR结果文件
python extract_asr_simple.py

# 5. 清理OSS存储（节省费用）
python ../cleanup_oss_simple.py
```

### Web API使用

```bash
# 上传热力图进行预测
curl -X POST http://localhost:3001/api/predictions \
  -F "thermal_image=@thermal.jpg" \
  -F "clinical_data={\"age\":45,\"bmi\":25.5}"

# 批量预测
curl -X POST http://localhost:3001/api/predictions/batch \
  -H "Content-Type: application/json" \
  -d '{"patient_ids": [1, 2, 3]}'
```

## 📈 数据集信息

### 热力图数据
- **总样本数**: 950张热力图
- **ICAS阳性**: 303张 (31.9%)
- **ICAS阴性**: 647张 (68.1%)
- **数据来源**: 2024年临床热成像数据
- **标注质量**: 专业医生标注，多人交叉验证

### 语音数据
- **总样本数**: 25小时语音数据
- **ICAS阳性**: 约8小时语音
- **ICAS阴性**: 约17小时语音
- **数据格式**: WAV格式，多种采样率
- **处理状态**: 已完成ASR识别和特征提取

## 🔧 配置说明

### 环境变量

创建 `.env` 文件:

```env
# 预测服务配置
PREDICTION_MODE=mock
PREDICTION_API_URL=http://localhost:5000/predict
PREDICTION_API_KEY=your_api_key

# 前端API配置
NEXT_PUBLIC_API_URL=http://localhost:3001/api

# 数据库配置
DATABASE_PATH=./database/database.db

# 阿里云服务配置
ALIYUN_AK_ID=your_access_key_id
ALIYUN_AK_SECRET=your_access_key_secret
ALIYUN_OSS_BUCKET=your_bucket_name
ALIYUN_OSS_ENDPOINT=oss-cn-beijing.aliyuncs.com
```

### 模型配置

```python
# model/config.py
MODEL_CONFIG = {
    'image_size': (224, 224),
    'batch_size': 32,
    'learning_rate': 1e-4,
    'epochs': 100,
    'early_stopping_patience': 10
}
```

## 🧪 测试

```bash
# 运行单元测试
python -m pytest tests/

# 运行集成测试
npm test

# 性能测试
python tests/performance_test.py
```

## 🛠️ 工具脚本详解

### 📊 数据集管理工具
```bash
cd dataset

# 数据导入
python import_thermal_24h.py        # 24小时热力图数据导入
python import_thermal_25h.py        # 25小时热力图数据导入
python import_voice_25h.py          # 25小时语音数据导入

# 数据集构建
python build_datasets.py            # 分类数据集构建
python build_yolo_dataset.py        # YOLO检测数据集构建
python extract_segmented_regions.py # 分割区域提取
```

### 🧠 模型分析工具
```bash
cd model

# 可解释性分析
python run_interpretability_analysis.py  # 交互式Grad-CAM分析
python interpretability_analysis.py      # 核心分析引擎
python comprehensive_feature_analysis.py # 全面特征重要性统计

# 模型预测
python predict_yolo11.py            # YOLOv11人脸检测预测

# 辅助工具
python verify_mask_consistency.py   # Mask一致性验证
python visualize_smart_mask.py      # 智能Mask可视化
python seg_all.py                   # SAM图像分割
```

### 🎤 语音处理工具
```bash
cd model

# 语音识别服务
python voice_asr.py
# 支持的功能：
# - OSS文件识别
# - 本地文件自动上传+识别
# - 批量目录处理
# - 断点续传
# - 结果提取和分析

# ASR结果提取
python extract_asr_simple.py        # 简化交互版
```

### 🛠️ 项目管理工具
```bash
# 项目结构整理
python organize_project.py

# ASR结果提取（根目录）
python extract_asr_results.py       # 完整版（命令行参数）

# OSS存储清理
python cleanup_oss_simple.py        # 简化交互版
python cleanup_oss.py               # 完整版（OOP架构）

# 教学模板
python thermal_classification_starter.py  # 学生作业模板
```

## 📚 文档导航

### 🏠 主要文档
- **[项目主文档](README.md)** - 项目概述、快速开始、系统架构（本文件）
- **[算法流程详细文档](docs/algorithm_pipeline_documentation.md)** - 完整的算法流程说明

### 📊 数据集文档
- **[数据集构建指南](dataset/README.md)** - 数据集管理和构建详细说明
  - 6个数据处理脚本的详细用法
  - 数据集统计信息和目录结构
  - 故障排除和最佳实践

### 🧠 模型文档
- **[模型工具集文档](model/README.md)** - 机器学习模型训练和分析工具集
  - 6种训练方法详细对比
  - 3种分析工具使用指南
  - 语音处理和辅助工具说明
  - 完整的使用示例和配置说明

#### 🔍 模型技术文档 (model/docs/)
- **[全面特征分析文档](model/docs/comprehensive_analysis_README.md)** - 特征重要性统计分析
- **[可解释性分析文档](model/docs/interpretability_analysis_README.md)** - Grad-CAM可解释性分析
- **[多模态特征说明](model/docs/multimodal_features_documentation.md)** - 324维图像特征+13维临床特征详解
- **[分类方法对比报告](model/docs/thermal_classification_methods_report.md)** - 三种分类方法性能对比
- **[智能Mask功能总结](model/docs/smart_mask_summary.md)** - 基于内容的智能人脸检测
- **[Mask一致性修复文档](model/docs/mask_consistency_fix.md)** - 技术修复说明

### 🌐 Web应用文档
- **[Web应用文档](web/README.md)** - 前后端系统说明
- **[API接口文档](web/docs/api.md)** - RESTful API详细说明

### 📈 结果和报告
- **[项目结构整理报告](results/organization_summary.txt)** - 项目重构总结
- **[数据集统计信息](dataset/datasets/dataset_summary.json)** - 数据集详细统计

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 👥 团队

- **算法开发**: ICAS算法开发团队
- **系统架构**: 系统设计团队
- **前端开发**: UI/UX团队
- **医学顾问**: 临床专家团队

## 📞 联系我们

- 📧 Email: team@icas-ai.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/IR-image/issues)
- 📖 Wiki: [项目Wiki](https://github.com/your-repo/IR-image/wiki)

## 🙏 致谢

感谢所有为本项目做出贡献的研究人员、开发者和临床专家。

---

⭐ 如果这个项目对您有帮助，请给我们一个星标！
