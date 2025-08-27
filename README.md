# IR-Image: 基于热力图的ICAS智能诊断系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

## 📋 项目概述

IR-Image是一个基于热力图的ICAS（颅内动脉粥样硬化性狭窄）智能诊断系统，集成了深度学习、计算机视觉和多模态数据融合技术，为临床医生提供高精度的ICAS风险评估工具。

### 🎯 核心功能

- **🔍 人脸检测**: 基于YOLOv11的热力图人脸区域自动检测
- **🧠 智能分类**: 多种机器学习方法的ICAS风险分类
- **📊 多模态融合**: 热力图特征与临床数据的深度融合
- **🌐 Web应用**: 完整的患者管理和预测系统
- **📈 实时分析**: 支持批量处理和实时预测

### 🏆 技术亮点

- **准确率**: 多模态融合模型达到74.83%的分类准确率
- **效率**: 单张图像预测时间<1秒
- **鲁棒性**: 支持多种图像质量和拍摄条件
- **可扩展**: 模块化设计，易于集成新算法

## 🏗️ 系统架构

```
IR-Image System
├── 数据层 (Dataset)
│   ├── 热力图数据收集与标注
│   ├── 临床数据整合
│   └── 数据质量控制
├── 算法层 (Model)
│   ├── YOLOv11人脸检测
│   ├── 传统特征分类
│   ├── 多模态融合分类
│   ├── 深度学习分类
│   └── 对比学习分类
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
├── 📂 dataset/                 # 数据集管理
│   ├── build_datasets.py      # 数据集构建
│   ├── import_thermal_*.py    # 数据导入脚本
│   └── datasets/              # 分类数据集
├── 📂 model/                   # 机器学习模型
│   ├── train_thermal_classifier*.py  # 分类模型训练
│   ├── train_yolo11.py        # 人脸检测模型训练
│   ├── predict_yolo11.py      # 预测脚本
│   └── *_results/             # 训练结果
├── 📂 web/                     # Web应用
│   ├── server.js              # 后端服务器
│   ├── frontend/              # Next.js前端
│   ├── routes/                # API路由
│   ├── services/              # 业务服务
│   └── database/              # SQLite数据库
├── 📂 docs/                    # 文档
│   └── algorithm_pipeline_documentation.md
├── 📂 thermal_segmentation/    # 图像分割
└── 📄 requirements.txt         # Python依赖
```

## 🔬 算法方法

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

# 2. 构建分类数据集
python build_datasets.py
```

### 模型训练

```bash
cd model

# 训练YOLOv11人脸检测模型
python train_yolo11.py

# 训练多模态融合分类器
python train_thermal_classifier2.py

# 训练对比学习模型
python train_thermal_classifier3.py
```

### 模型预测

```bash
# 人脸检测
python predict_yolo11.py --input path/to/thermal/image.jpg

# ICAS风险预测
python model/predict.py --image path/to/face/image.jpg --clinical clinical_data.json
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

- **总样本数**: 950张热力图
- **ICAS阳性**: 303张 (31.9%)
- **ICAS阴性**: 647张 (68.1%)
- **数据来源**: 2024年临床热成像数据
- **标注质量**: 专业医生标注，多人交叉验证

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

## 📚 文档

- [算法流程详细文档](docs/algorithm_pipeline_documentation.md)
- [数据集构建指南](dataset/README.md)
- [Web应用文档](web/README.md)
- [API接口文档](web/docs/api.md)
- [代码问题分析](model/docs/train_thermal_classifier3_issues.md)

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
