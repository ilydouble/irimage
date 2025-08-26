# 多模态特征文档 (Multimodal Features Documentation)

本文档详细介绍了 `train_thermal_classifier2.py` 生成的 `multimodal_features.csv` 文件中各个特征字段的含义和计算方法。

## 概述

多模态特征文件包含了从热力图图像和临床数据中提取的特征，用于ICAS（颅内动脉粥样硬化性狭窄）分类任务。特征分为三大类：
- **图像特征**：从热力图中提取的温度、纹理和形状特征
- **临床特征**：从数据库中获取的患者生理指标
- **标识字段**：患者ID和分类标签

## 1. 标识字段

| 字段名 | 类型 | 含义 |
|--------|------|------|
| `patient_id` | string | 患者唯一标识符，从图像文件名提取 |
| `label` | int | 分类标签 (0: non_icas, 1: icas) |
| `has_icas` | int | 数据库中的ICAS诊断结果 (0/1) |

## 2. 图像特征

### 2.1 温度特征 (Temperature Features)

基于HSV色彩空间的色调(H)通道和红蓝比值提取温度相关特征。

#### 2.1.1 基于色调的温度特征
```python
# 使用HSV色彩空间的色调通道作为温度代理
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hue = hsv[:, :, 0]  # 色调通道
temp_proxy = hue[mask]  # 排除背景后的前景区域
```

| 字段名 | 计算方法 | 含义 |
|--------|----------|------|
| `hue_temp_mean` | `np.mean(temp_proxy)` | 色调温度均值 |
| `hue_temp_std` | `np.std(temp_proxy)` | 色调温度标准差 |
| `hue_temp_min` | `np.min(temp_proxy)` | 色调温度最小值 |
| `hue_temp_max` | `np.max(temp_proxy)` | 色调温度最大值 |
| `hue_temp_median` | `np.median(temp_proxy)` | 色调温度中位数 |
| `hue_temp_range` | `max - min` | 色调温度范围 |
| `hue_temp_p25` | `np.percentile(temp_proxy, 25)` | 25%分位数 |
| `hue_temp_p75` | `np.percentile(temp_proxy, 75)` | 75%分位数 |
| `hue_temp_iqr` | `p75 - p25` | 四分位距 |
| `hue_temp_skewness` | `scipy.stats.skew(temp_proxy)` | 偏度 |
| `hue_temp_kurtosis` | `scipy.stats.kurtosis(temp_proxy)` | 峰度 |
| `hue_temp_entropy` | `shannon_entropy(temp_proxy)` | 香农熵 |

#### 2.1.2 基于红蓝比值的温度特征
```python
# 计算红蓝比值作为温度指标
red_channel = image[:, :, 2]  # BGR格式中红色通道
blue_channel = image[:, :, 0]  # 蓝色通道
red_blue_ratio = red_channel / (blue_channel + 1e-8)
```

| 字段名 | 计算方法 | 含义 |
|--------|----------|------|
| `rb_ratio_mean` | `np.mean(temp_ratio)` | 红蓝比值均值 |
| `rb_ratio_std` | `np.std(temp_ratio)` | 红蓝比值标准差 |
| `rb_ratio_max` | `np.max(temp_ratio)` | 红蓝比值最大值 |
| `rb_ratio_min` | `np.min(temp_ratio)` | 红蓝比值最小值 |

#### 2.1.3 区域特征
| 字段名 | 计算方法 | 含义 |
|--------|----------|------|
| `foreground_ratio` | `np.sum(mask) / mask.size` | 前景区域占比 |
| `high_temp_ratio` | 高温区域占前景比例 | 红色区域(色调0-30°, 150-180°)占比 |

### 2.2 纹理特征 (Texture Features)

基于灰度共生矩阵(GLCM)提取纹理特征，使用多个距离和角度组合。

```python
# GLCM参数设置
distances = [1, 2, 3]  # 像素距离
angles = [0, 45, 90, 135]  # 角度(度)

# 对每个距离-角度组合计算GLCM特征
glcm = graycomatrix(gray_reduced, distances=[dist], angles=[angle_rad], 
                   levels=64, symmetric=True, normed=True)
```

#### 2.2.1 GLCM特征命名规则
格式：`glcm_d{距离}_a{角度}_{特征名}`

**示例字段：**
- `glcm_d1_a0_contrast`: 距离1，角度0°的对比度
- `glcm_d2_a45_homogeneity`: 距离2，角度45°的同质性
- `glcm_d3_a90_energy`: 距离3，角度90°的能量

#### 2.2.2 GLCM特征类型
| 特征名 | 计算方法 | 含义 |
|--------|----------|------|
| `contrast` | `graycoprops(glcm, 'contrast')` | 对比度，衡量局部变化 |
| `dissimilarity` | `graycoprops(glcm, 'dissimilarity')` | 相异性 |
| `homogeneity` | `graycoprops(glcm, 'homogeneity')` | 同质性，衡量纹理均匀性 |
| `energy` | `graycoprops(glcm, 'energy')` | 能量，衡量纹理规律性 |
| `correlation` | `graycoprops(glcm, 'correlation')` | 相关性 |

### 2.3 形状特征 (Shape Features)

基于轮廓检测提取形状特征，使用多种阈值方法。

```python
# 使用多种阈值方法
thresholds = [cv2.THRESH_BINARY + cv2.THRESH_OTSU, 
             cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE]

# 对每种阈值方法提取轮廓特征
_, binary = cv2.threshold(gray, 0, 255, thresh_type)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

#### 2.3.1 形状特征命名规则
格式：`{特征名}_{阈值方法索引}`

**阈值方法：**
- `_0`: OTSU阈值
- `_1`: Triangle阈值

#### 2.3.2 形状特征类型
| 字段名 | 计算方法 | 含义 |
|--------|----------|------|
| `contour_area_{i}` | `cv2.contourArea(largest_contour)` | 最大轮廓面积 |
| `contour_perimeter_{i}` | `cv2.arcLength(largest_contour, True)` | 最大轮廓周长 |
| `contour_compactness_{i}` | `perimeter² / (4π × area)` | 紧凑度 |
| `bbox_aspect_ratio_{i}` | `width / height` | 边界框宽高比 |
| `bbox_extent_{i}` | `area / (width × height)` | 边界框填充率 |

## 3. 临床特征

从数据库 `patients` 表中提取的患者生理指标。

### 3.1 基础生理指标
| 字段名 | 单位 | 含义 |
|--------|------|------|
| `age` | 年 | 患者年龄 |
| `gender_encoded` | - | 性别编码 (0: male, 1: female) |
| `height` | cm | 身高 |
| `weight` | kg | 体重 |
| `bmi` | kg/m² | 体质指数 |
| `waist` | cm | 腰围 |
| `hip` | cm | 臀围 |
| `neck` | cm | 颈围 |

### 3.2 衍生特征
```python
# 计算衍生特征
df['waist_hip_ratio'] = df['waist'] / df['hip']
df['waist_height_ratio'] = df['waist'] / df['height']
df['neck_height_ratio'] = df['neck'] / df['height']
```

| 字段名 | 计算方法 | 含义 |
|--------|----------|------|
| `waist_hip_ratio` | `waist / hip` | 腰臀比 |
| `waist_height_ratio` | `waist / height` | 腰身比 |
| `neck_height_ratio` | `neck / height` | 颈身比 |

### 3.3 分类特征
```python
# BMI分类
def bmi_category(bmi):
    if bmi < 18.5: return 0    # 偏瘦
    elif bmi < 24: return 1    # 正常
    elif bmi < 28: return 2    # 超重
    else: return 3             # 肥胖

# 年龄分组
def age_group(age):
    if age < 30: return 0      # 青年
    elif age < 45: return 1    # 中年早期
    elif age < 60: return 2    # 中年晚期
    else: return 3             # 老年
```

| 字段名 | 取值范围 | 含义 |
|--------|----------|------|
| `bmi_category` | 0-3 | BMI分类 (0:偏瘦, 1:正常, 2:超重, 3:肥胖) |
| `age_group` | 0-3 | 年龄分组 (0:<30, 1:30-45, 2:45-60, 3:≥60) |

## 4. 数据预处理

### 4.1 缺失值处理
- **数值型特征**: 使用中位数填充
- **分类型特征**: 使用众数填充
- **图像特征**: 使用0填充

### 4.2 特征标准化
- **图像特征**: 使用RobustScaler (对异常值更鲁棒)
- **临床特征**: 使用StandardScaler

### 4.3 特征选择
- **方差阈值**: 移除方差小于0.01的特征
- **PCA降维**: 当图像特征维度>100时，降维到100维

## 5. 使用示例

```python
# 加载特征文件
import pandas as pd
features_df = pd.read_csv('multimodal_features.csv')

# 查看特征统计
print(f"总样本数: {len(features_df)}")
print(f"特征维度: {len(features_df.columns) - 3}")  # 排除patient_id, label, has_icas
print(f"ICAS分布: {features_df['label'].value_counts()}")

# 分离不同类型特征
image_features = [col for col in features_df.columns 
                 if col.startswith(('hue_', 'rb_', 'glcm_', 'contour_', 'bbox_', 'foreground_', 'high_temp_'))]
clinical_features = ['age', 'gender_encoded', 'height', 'weight', 'bmi', 'waist', 'hip', 'neck',
                    'waist_hip_ratio', 'waist_height_ratio', 'neck_height_ratio', 'bmi_category', 'age_group']
```

## 6. 注意事项

1. **图像质量**: 特征提取依赖于图像质量，低质量图像可能产生无效特征
2. **背景处理**: 使用饱和度和明度阈值排除背景，阈值设置为30
3. **数据匹配**: 图像特征与临床特征通过patient_id匹配，未匹配的样本用中位数/众数填充
4. **特征缩放**: 不同类型特征使用不同的标准化方法，使用时需注意
5. **类别不平衡**: 数据集可能存在类别不平衡，建议使用适当的采样或加权方法

---

*生成时间: 基于 train_thermal_classifier2.py 代码实现*
*版本: v1.0*