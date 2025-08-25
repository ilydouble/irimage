# 数据集构建工具

本目录包含用于构建分类数据集的脚本，基于患者数据库中的ICAS诊断结果进行分类。

## 脚本说明

### 1. build_datasets.py
构建分类数据集，将已导入的文件按ICAS状态分类到不同文件夹。

### 2. import_thermal_24h.py
导入24年热力图数据到数据库files表。

## 使用方法

### 导入24年热力图数据
```bash
cd dataset
python import_thermal_24h.py
```

### 构建分类数据集
```bash
cd dataset
python build_datasets.py
```

## 文件类型编码

- `thermal_24h`: 24年热力图
- `thermal_25h`: 25年热力图  
- `voice_25h`: 25年语音

## 数据目录结构

```
web/
├── 2025年合同完整数据/
│   └── 2024热成像数据/
│       ├── {patient_id}/
│       │   └── {patient_id}1.jpg
│       └── ...
└── database.db
```

## 数据集结构

构建完成后会生成以下目录结构：

```
datasets/
├── thermal_24h/
│   ├── icas/          # ICAS阳性患者的24年热力图
│   └── non_icas/      # ICAS阴性患者的24年热力图
├── thermal_25h/
│   ├── icas/          # ICAS阳性患者的25年热力图
│   └── non_icas/      # ICAS阴性患者的25年热力图
├── voice_25h/
│   ├── icas/          # ICAS阳性患者的25年语音
│   └── non_icas/      # ICAS阴性患者的25年语音
└── dataset_summary.json  # 数据集统计摘要
```