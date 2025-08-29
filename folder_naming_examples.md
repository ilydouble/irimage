# 训练脚本文件夹命名规范

## 🎯 **新的命名格式**

所有训练脚本现在使用描述性的文件夹命名，包含关键训练参数：

### 1. **CNN分类器** (`train_cnn_classifier.py`)
```
格式: CNN-{BACKBONE}-{LOSS_TYPE}-E{EPOCHS}-LR{LEARNING_RATE}-B{BATCH_SIZE}-{TIMESTAMP}

示例:
├── CNN-RESNET18-FOCAL-E100-LR0.0001-B32-20240829_143022/
├── CNN-EFFICIENTNET_B0-CE-E50-LR0.001-B16-20240829_150315/
└── CNN-YOLO11S-FOCAL-E75-LR0.0005-B24-20240829_162045/
```

### 2. **对比学习分类器** (`train_contrastive_classifier.py`)
```
格式: CONTRASTIVE-E{CONTRASTIVE_EPOCHS}+{CLASSIFICATION_EPOCHS}-LR{CONTRASTIVE_LR}+{CLASSIFICATION_LR}-B{BATCH_SIZE}-{ASYMMETRY_FLAG}-{TIMESTAMP}

示例:
├── CONTRASTIVE-E50+30-LR0.001+0.0001-B32-FULL-20240829_143022/
├── CONTRASTIVE-E100+50-LR0.0005+0.00005-B16-ASYM-20240829_150315/
└── CONTRASTIVE-E75+25-LR0.002+0.0002-B24-FULL-20240829_162045/
```

### 3. **带Mask的对比学习** (`train_contrastive_mask_classifier.py`)
```
格式: CONTRASTIVE-{MASK_FLAG}-{ATTENTION_FLAG}-E{CONTRASTIVE_EPOCHS}+{CLASSIFICATION_EPOCHS}-LR{CONTRASTIVE_LR}+{CLASSIFICATION_LR}-B{BATCH_SIZE}-{ASYMMETRY_FLAG}-{TIMESTAMP}

示例:
├── CONTRASTIVE-MASK-ELLIPSE-ATT-E50+30-LR0.001+0.0001-B32-FULL-20240829_143022/
├── CONTRASTIVE-MASK-CONTENT_BASED-NOATT-E75+40-LR0.0005+0.00005-B16-ASYM-20240829_150315/
└── CONTRASTIVE-NOMASK-ATT-E60+35-LR0.002+0.0002-B24-FULL-20240829_162045/
```

### 4. **分割对比学习** (`train_contrastive_split_classifier.py`)
```
格式: CONTRASTIVE-SPLIT-E{CONTRASTIVE_EPOCHS}+{CLASSIFICATION_EPOCHS}-LR{CONTRASTIVE_LR}+{CLASSIFICATION_LR}-B{BATCH_SIZE}-{ASYMMETRY_FLAG}-{TIMESTAMP}

示例:
├── CONTRASTIVE-SPLIT-E50+30-LR0.001+0.0001-B32-FULL-20240829_143022/
├── CONTRASTIVE-SPLIT-E100+50-LR0.0005+0.00005-B16-ASYM-20240829_150315/
└── CONTRASTIVE-SPLIT-E75+25-LR0.002+0.0002-B24-FULL-20240829_162045/
```

### 5. **特征ML分类器** (`train_feature_ml_classifier.py`)
```
格式: FEATURE-{BEST_MODEL}-FOCAL-A{ALPHA}-G{GAMMA}-{TIMESTAMP}

示例:
├── FEATURE-RANDOMFOREST-FOCAL-A0.25-G2.0-20240829_143022/
├── FEATURE-XGBOOST-FOCAL-A0.3-G1.5-20240829_150315/
└── FEATURE-SVM-FOCAL-A0.2-G2.5-20240829_162045/
```

### 6. **多模态分类器** (`train_multimodal_classifier.py`)
```
格式: MULTIMODAL-{BEST_MODEL}-FOCAL-A{ALPHA}-G{GAMMA}-{TIMESTAMP}

示例:
├── MULTIMODAL-RANDOMFOREST-FOCAL-A0.25-G2.0-20240829_143022/
├── MULTIMODAL-GRADIENTBOOSTING-FOCAL-A0.3-G1.5-20240829_150315/
└── MULTIMODAL-LOGISTICREGRESSION-FOCAL-A0.2-G2.5-20240829_162045/
```

## 📋 **参数说明**

### 通用参数
- **E{数字}**: 训练轮数 (Epochs)
- **LR{数字}**: 学习率 (Learning Rate)
- **B{数字}**: 批次大小 (Batch Size)
- **A{数字}**: Focal Loss Alpha参数
- **G{数字}**: Focal Loss Gamma参数
- **{TIMESTAMP}**: 时间戳 (YYYYMMDD_HHMMSS)

### CNN特定参数
- **BACKBONE**: 骨干网络 (RESNET18, EFFICIENTNET_B0, YOLO11S等)
- **LOSS_TYPE**: 损失函数类型 (FOCAL, CE)

### 对比学习特定参数
- **ASYMMETRY_FLAG**: 不对称分析标志 (ASYM, FULL)
- **MASK_FLAG**: Mask类型 (MASK-ELLIPSE, MASK-CONTENT_BASED, NOMASK)
- **ATTENTION_FLAG**: 注意力机制 (ATT, NOATT)

### 传统ML特定参数
- **BEST_MODEL**: 最佳模型名称 (RANDOMFOREST, XGBOOST, SVM等)

## 🎯 **优势**

1. **一目了然**: 从文件夹名就能看出训练配置
2. **易于比较**: 不同参数的实验结果容易对比
3. **便于管理**: 按参数组织，方便查找特定配置的结果
4. **避免混淆**: 不再有无意义的run_001, run_002等编号
5. **自动排序**: 按时间戳自然排序

## 📁 **结果目录结构示例**

```
results/training_results/
├── thermal_classifier_results/
│   ├── CNN-RESNET18-FOCAL-E100-LR0.0001-B32-20240829_143022/
│   │   ├── best_model.pth
│   │   ├── config.json
│   │   ├── confusion_matrix.png
│   │   ├── test_results.json
│   │   ├── training_curves.png
│   │   └── training_history.json
│   └── CNN-EFFICIENTNET_B0-CE-E50-LR0.001-B16-20240829_150315/
│
├── contrastive_thermal_classifier_results/
│   ├── CONTRASTIVE-E50+30-LR0.001+0.0001-B32-FULL-20240829_143022/
│   │   ├── best_classifier.pth
│   │   ├── best_contrastive_encoder.pth
│   │   ├── classification_training_curves.png
│   │   ├── confusion_matrix.png
│   │   ├── contrastive_training_curves.png
│   │   └── roc_curve.png
│   └── CONTRASTIVE-MASK-ELLIPSE-ATT-E75+40-LR0.0005+0.00005-B16-ASYM-20240829_150315/
│
├── thermal_feature_classifier_results/
│   └── FEATURE-RANDOMFOREST-FOCAL-A0.25-G2.0-20240829_143022/
│
└── multimodal_thermal_classifier_results/
    └── MULTIMODAL-XGBOOST-FOCAL-A0.3-G1.5-20240829_150315/
```

现在每个训练结果都有清晰、有意义的文件夹名称！🎉
