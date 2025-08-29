# 数据集构建工具

本目录包含用于构建和处理医学影像分类数据集的完整工具链，基于患者数据库中的ICAS诊断结果进行分类。

## 脚本说明

### 数据导入脚本

#### 1. import_thermal_24h.py
导入2024年热力图数据到数据库files表。
- 从数据库获取需要24年热力图的患者ID
- 在指定目录中查找对应的热力图文件
- 支持多种文件命名格式：`{patient_id}1.jpg`、`{patient_id}-1.jpg`、`{patient_id}-6.jpg`
- 将文件信息保存到数据库files表

#### 2. import_thermal_25h.py
导入2025年热力图数据到数据库files表。
- 从数据库获取需要25年热力图的患者ID
- 查找格式为`{patient_id}-正-1.jpg`或`{patient_id}-正-2.jpg`的文件
- 将文件信息保存到数据库files表

#### 3. import_voice_25h.py
导入2025年语音数据到数据库files表。
- 从数据库获取需要25年语音的患者ID
- 查找包含患者ID的语音文件
- 支持多种语音文件格式
- 将文件信息保存到数据库files表

### 数据集构建脚本

#### 4. build_datasets.py
构建基础分类数据集，将已导入的文件按ICAS状态分类。
- 从数据库files表获取文件信息和患者ICAS状态
- 按文件类型和ICAS状态分类复制文件
- 生成数据集统计摘要文件
- 支持三种数据类型：thermal_24h、thermal_25h、voice_25h

#### 5. build_yolo_dataset.py
构建YOLO格式的分割数据集。
- 从thermal_24h/icas目录构建YOLO训练数据集
- 查找图像-标签文件对（.jpg和.txt）
- 按指定比例分割训练集和验证集（默认8:2）
- 生成YOLO格式的目录结构和配置文件

#### 6. extract_segmented_regions.py
提取分割区域并生成裁剪后的分类数据集。
- 解析YOLO分割标签文件
- 根据分割掩码提取前景区域
- 将提取的区域调整为统一尺寸（512x512）
- 生成用于分类训练的裁剪数据集

## 使用方法

### 1. 数据导入流程
```bash
cd dataset

# 导入24年热力图数据
python import_thermal_24h.py

# 导入25年热力图数据
python import_thermal_25h.py

# 导入25年语音数据
python import_voice_25h.py
```

### 2. 构建基础分类数据集
```bash
cd dataset
python build_datasets.py
```

### 3. 构建YOLO分割数据集
```bash
cd dataset
python build_yolo_dataset.py
```

### 4. 提取分割区域生成裁剪数据集
```bash
cd dataset
python extract_segmented_regions.py
```

## 文件类型编码

- `thermal_24h`: 2024年热力图数据
- `thermal_25h`: 2025年热力图数据
- `voice_25h`: 2025年语音数据

## 源数据目录结构

```
../
├── 2025年合同完整数据/
│   ├── 2024热成像数据/
│   │   ├── {patient_id}/
│   │   │   └── {patient_id}1.jpg
│   │   └── ...
│   ├── 2025热成像数据/
│   │   └── 热成像照片/
│   │       ├── {patient_id}-正-1.jpg
│   │       └── ...
│   └── 语音数据/
│       ├── {patient_id}_xxx.wav
│       └── ...
└── web/database/patientcare.db
```

## 生成的数据集结构

构建完成后会生成以下目录结构：

```
datasets/
├── thermal_24h/                    # 24年热力图基础数据集
│   ├── icas/                      # ICAS阳性患者
│   └── non_icas/                  # ICAS阴性患者
├── thermal_25h/                    # 25年热力图基础数据集
│   ├── icas/                      # ICAS阳性患者
│   └── non_icas/                  # ICAS阴性患者
├── voice_25h/                      # 25年语音基础数据集
│   ├── icas/                      # ICAS阳性患者
│   └── non_icas/                  # ICAS阴性患者
├── thermal_24h_yolo/               # YOLO分割数据集
│   ├── images/
│   │   ├── train/                 # 训练图像
│   │   └── val/                   # 验证图像
│   ├── labels/
│   │   ├── train/                 # 训练标签
│   │   └── val/                   # 验证标签
│   └── dataset.yaml               # YOLO配置文件
├── thermal_classification/         # 合并的热力图分类数据集
│   ├── icas/                      # 包含24年和25年的ICAS阳性数据
│   └── non_icas/                  # 包含24年和25年的ICAS阴性数据
├── thermal_classification_cropped/ # 裁剪后的分类数据集
│   ├── icas/                      # 基于分割结果裁剪的ICAS阳性数据
│   └── non_icas/                  # 基于分割结果裁剪的ICAS阴性数据
└── dataset_summary.json            # 数据集统计摘要
```

## 数据集统计信息

根据最新的数据集摘要（dataset_summary.json）：

| 数据类型 | ICAS阳性 | ICAS阴性 | 总计 |
|---------|---------|---------|------|
| 24年热力图 | 774 | 1,703 | 2,477 |
| 25年热力图 | 423 | 901 | 1,324 |
| 25年语音 | 141 | 301 | 442 |
| **总计** | **1,338** | **2,905** | **4,243** |

## 其他目录说明

### seg_result/
包含分割结果的示例图像，用于验证分割算法的效果。

### datasets/子目录
- `asr_results/`: 语音识别结果
- `feature_importance_analysis/`: 特征重要性分析结果
- `interpretability_analysis/`: 可解释性分析结果
- `quick_analysis/`: 快速分析结果

## 注意事项

### 1. 数据库依赖
- 所有脚本都依赖于SQLite数据库 `../web/database/patientcare.db`
- 确保数据库中的patients表包含正确的ICAS诊断信息
- files表用于存储文件路径和元数据

### 2. 文件路径
- 所有脚本假设从dataset目录运行
- 源数据路径相对于dataset目录的上级目录
- 生成的数据集保存在 `./datasets/` 目录下

### 3. 文件命名规则
- **24年热力图**: `{patient_id}1.jpg`、`{patient_id}-1.jpg`、`{patient_id}-6.jpg`
- **25年热力图**: `{patient_id}-正-1.jpg`、`{patient_id}-正-2.jpg`
- **25年语音**: 文件名包含患者ID的任意格式

### 4. YOLO数据集要求
- 需要对应的.txt标签文件与.jpg图像文件配对
- 标签文件格式为YOLO分割格式（多边形坐标）
- 训练/验证分割比例可在脚本中调整（默认8:2）

### 5. 分割区域提取
- 依赖于YOLO分割模型的预测结果
- 输出图像统一调整为512x512像素
- 非分割区域填充为黑色

## 故障排除

### 常见问题
1. **数据库连接失败**: 检查数据库文件路径是否正确
2. **源文件不存在**: 验证源数据目录结构和文件命名
3. **权限问题**: 确保对目标目录有写入权限
4. **内存不足**: 处理大量图像时可能需要调整批处理大小

### 日志输出
所有脚本都会输出详细的处理日志，包括：
- 处理进度
- 成功/失败统计
- 错误信息和警告
- 最终结果摘要

## 扩展功能

### 自定义配置
可以通过修改脚本参数来自定义：
- 数据库路径
- 源数据目录
- 输出目录
- 训练/验证分割比例
- 图像尺寸设置

### 批处理支持
所有脚本都支持批量处理，可以处理大规模数据集而无需人工干预。