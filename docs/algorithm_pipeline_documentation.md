# ICAS热力图分析算法流程详细文档

## 概述

本文档详细介绍了基于热力图的ICAS（颅内动脉粥样硬化性狭窄）检测与分类的完整算法流程。整个流程包括数据集构建、人脸检测、图像分割、特征提取和分类预测等多个阶段，采用多种机器学习和深度学习方法实现高精度的ICAS风险评估。

## 1. 整体算法架构

```
原始热力图数据
    ↓
数据集构建与标注
    ↓
YOLOv11人脸检测模型训练
    ↓
人脸区域检测与分割
    ↓
人脸热力数据集构建
    ↓
多种分类方法并行训练
    ├── 传统图像特征分类
    ├── 多模态特征融合分类
    ├── 深度学习分类
    └── 对比学习分类
    ↓
模型性能评估与选择
    ↓
最终ICAS风险预测
```

## 2. 数据集构建阶段

### 2.1 原始数据收集
- **数据来源**: 2024年热成像数据
- **数据格式**: JPG格式热力图图像
- **数据规模**: 950张热力图
- **存储结构**: `2025年合同完整数据/2024热成像数据/{patient_id}/{patient_id}1.jpg`

### 2.2 数据导入与管理
使用 `dataset/import_thermal_24h.py` 脚本将热力图数据导入数据库：

```python
# 数据导入流程
1. 扫描热力图文件目录
2. 提取患者ID信息
3. 记录文件路径和元数据
4. 存储到数据库files表
```

### 2.3 分类数据集构建
使用 `dataset/build_datasets.py` 根据ICAS诊断结果构建分类数据集：

```
datasets/
├── thermal_24h/
│   ├── icas/          # ICAS阳性患者 (303张, 31.9%)
│   └── non_icas/      # ICAS阴性患者 (647张, 68.1%)
└── dataset_summary.json
```

## 3. 人脸检测与分割阶段

### 3.1 数据标注
- **标注工具**: 专业图像标注工具
- **标注内容**: 热力图中的人脸区域边界框
- **标注格式**: YOLO格式 (class_id, x_center, y_center, width, height)
- **质量控制**: 多人标注，交叉验证

### 3.2 YOLOv11模型训练
```python
# 训练配置
model_architecture: YOLOv11
input_size: 640x640
batch_size: 16
epochs: 100
optimizer: AdamW
learning_rate: 0.001
data_augmentation: 
  - rotation: ±15°
  - scaling: 0.8-1.2
  - brightness: ±20%
  - contrast: ±20%
```

### 3.3 人脸检测流程
```python
# 检测流程伪代码
def detect_face_regions(thermal_image):
    # 1. 图像预处理
    image = preprocess_thermal_image(thermal_image)
    
    # 2. YOLOv11推理
    detections = yolo_model.predict(image)
    
    # 3. 后处理
    face_boxes = post_process_detections(detections)
    
    # 4. 置信度筛选
    valid_faces = filter_by_confidence(face_boxes, threshold=0.5)
    
    return valid_faces
```

### 3.4 图像分割与裁剪
```python
# 分割流程
def segment_face_regions(image, face_boxes):
    face_regions = []
    for box in face_boxes:
        # 1. 根据检测框裁剪人脸区域
        face_region = crop_image(image, box)
        
        # 2. 尺寸标准化
        face_region = resize_image(face_region, target_size=(224, 224))
        
        # 3. 质量检查
        if quality_check(face_region):
            face_regions.append(face_region)
    
    return face_regions
```

## 4. 人脸热力数据集构建

### 4.1 数据预处理
```python
# 预处理流程
def preprocess_face_thermal_data():
    processed_data = []
    
    for image_path in thermal_images:
        # 1. 加载原始热力图
        thermal_img = load_thermal_image(image_path)
        
        # 2. 人脸检测
        face_boxes = detect_face_regions(thermal_img)
        
        # 3. 人脸分割
        face_regions = segment_face_regions(thermal_img, face_boxes)
        
        # 4. 数据增强
        augmented_faces = apply_augmentation(face_regions)
        
        # 5. 质量筛选
        valid_faces = quality_filter(augmented_faces)
        
        processed_data.extend(valid_faces)
    
    return processed_data
```

### 4.2 数据集划分
```python
# 数据集划分策略
train_ratio = 0.7    # 训练集70%
val_ratio = 0.15     # 验证集15%
test_ratio = 0.15    # 测试集15%

# 分层采样确保类别平衡
stratified_split(face_thermal_dataset, ratios=[0.7, 0.15, 0.15])
```

## 5. 分类模型方法选型

### 5.1 方法一：传统图像特征分类

#### 5.1.1 特征提取策略
基于 `train_thermal_classifier1.py` 实现：

```python
# 特征提取流程
def extract_traditional_features(face_image):
    features = []
    
    # 1. 温度特征 (基于HSV色彩空间)
    hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
    temp_features = extract_temperature_features(hsv)
    
    # 2. 纹理特征 (LBP)
    lbp_features = extract_lbp_features(face_image)
    
    # 3. 形状特征 (轮廓分析)
    shape_features = extract_shape_features(face_image)
    
    features.extend([temp_features, lbp_features, shape_features])
    return np.concatenate(features)
```

#### 5.1.2 分类器选择
```python
# 多种分类器对比
classifiers = {
    'LogisticRegression': LogisticRegression(),
    'SVM': SVC(kernel='rbf'),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier()
}
```

### 5.2 方法二：多模态特征融合分类

#### 5.2.1 特征融合架构
基于 `train_thermal_classifier2.py` 实现：

```python
# 多模态特征融合
def extract_multimodal_features(face_image, clinical_data):
    # 1. 图像特征 (324维)
    image_features = extract_enhanced_image_features(face_image)
    
    # 2. 临床特征 (13维)
    clinical_features = extract_clinical_features(clinical_data)
    
    # 3. 特征融合
    multimodal_features = np.concatenate([image_features, clinical_features])
    
    return multimodal_features

def extract_enhanced_image_features(image):
    # 温度特征 (~18维)
    temp_features = extract_temperature_features(image)
    
    # 分块LBP直方图 (288维)
    block_lbp = extract_block_lbp_histogram(image, grid_size=(4,4))
    
    # 全局LBP统计 (12维)
    global_lbp = extract_global_lbp_statistics(image)
    
    # 形状特征 (~6维)
    shape_features = extract_shape_features(image)
    
    return np.concatenate([temp_features, block_lbp, global_lbp, shape_features])
```

#### 5.2.2 临床特征工程
```python
def extract_clinical_features(patient_data):
    # 基础生理指标 (8维)
    basic_features = [
        patient_data['age'], patient_data['gender_encoded'],
        patient_data['height'], patient_data['weight'],
        patient_data['bmi'], patient_data['waist'],
        patient_data['hip'], patient_data['neck']
    ]
    
    # 衍生特征 (3维)
    derived_features = [
        patient_data['waist'] / patient_data['hip'],      # 腰臀比
        patient_data['waist'] / patient_data['height'],   # 腰身比
        patient_data['neck'] / patient_data['height']     # 颈身比
    ]
    
    # 分类特征 (2维)
    categorical_features = [
        categorize_bmi(patient_data['bmi']),              # BMI分类
        categorize_age(patient_data['age'])               # 年龄分组
    ]
    
    return np.concatenate([basic_features, derived_features, categorical_features])
```

### 5.3 方法三：深度学习分类

#### 5.3.1 CNN架构设计
```python
# 深度学习模型架构
class ThermalCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(ThermalCNN, self).__init__()
        
        # 特征提取层
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

#### 5.3.2 训练策略
```python
# 训练配置
training_config = {
    'optimizer': 'AdamW',
    'learning_rate': 1e-4,
    'batch_size': 32,
    'epochs': 100,
    'loss_function': 'CrossEntropyLoss',
    'scheduler': 'CosineAnnealingLR',
    'data_augmentation': True,
    'early_stopping': True,
    'patience': 10
}
```

### 5.4 方法四：对比学习分类

#### 5.4.1 对比学习架构
基于 `train_thermal_classifier3.py` 实现：

```python
# 对比学习模型
class ContrastiveThermalClassifier(nn.Module):
    def __init__(self, backbone='resnet50', projection_dim=128):
        super().__init__()
        
        # 骨干网络
        self.backbone = self._build_backbone(backbone)
        
        # 投影头
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
        
        # 分类头
        self.classification_head = nn.Linear(2048, 2)
    
    def forward(self, x, mode='classification'):
        features = self.backbone(x)
        
        if mode == 'contrastive':
            return self.projection_head(features)
        else:
            return self.classification_head(features)
```

#### 5.4.2 两阶段训练策略
```python
# 阶段1: 对比学习预训练
def contrastive_pretraining():
    for epoch in range(contrastive_epochs):
        for batch in contrastive_dataloader:
            # 数据增强生成正负样本对
            anchor, positive, negative = generate_triplets(batch)
            
            # 特征提取
            anchor_features = model(anchor, mode='contrastive')
            positive_features = model(positive, mode='contrastive')
            negative_features = model(negative, mode='contrastive')
            
            # 对比损失
            loss = contrastive_loss(anchor_features, positive_features, negative_features)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# 阶段2: 分类微调
def classification_finetuning():
    # 冻结骨干网络部分参数
    freeze_backbone_layers(model.backbone, freeze_ratio=0.8)
    
    for epoch in range(finetuning_epochs):
        for batch in classification_dataloader:
            images, labels = batch
            
            # 分类预测
            predictions = model(images, mode='classification')
            
            # 分类损失
            loss = classification_loss(predictions, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

## 6. 模型性能评估

### 6.1 评估指标
```python
# 评估指标计算
def evaluate_model(y_true, y_pred, y_prob):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'auc': roc_auc_score(y_true, y_prob[:, 1]),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    return metrics
```

### 6.2 方法性能对比

| 方法 | 最佳模型 | 测试准确率 | 测试F1分数 | 测试AUC | 训练时间 | 特征维度 |
|------|----------|------------|------------|---------|----------|----------|
| **多模态融合** | **LightGBM** | **74.83%** | **53.85%** | **75.46%** | **28.51秒** | **42维** |
| 传统特征 | LogisticRegression | 51.75% | 37.84% | 54.14% | 28.12秒 | 26维 |
| 深度学习 | ThermalCNN | *待测试* | *-* | *-* | *-* | *-* |
| 对比学习 | ContrastiveNet | *待测试* | *-* | *-* | *-* | *-* |

### 6.3 模型选择策略
```python
# 模型选择决策树
def select_best_model(performance_metrics, requirements):
    if requirements['accuracy_priority']:
        return max(performance_metrics, key=lambda x: x['accuracy'])
    elif requirements['speed_priority']:
        return min(performance_metrics, key=lambda x: x['inference_time'])
    elif requirements['interpretability_priority']:
        return filter_interpretable_models(performance_metrics)
    else:
        # 综合评分
        return calculate_weighted_score(performance_metrics)
```

## 7. 部署与应用

### 7.1 模型部署流程
```python
# 模型部署管道
class ICASPredictionPipeline:
    def __init__(self, face_detector, feature_extractor, classifier):
        self.face_detector = face_detector
        self.feature_extractor = feature_extractor
        self.classifier = classifier
    
    def predict(self, thermal_image, clinical_data=None):
        # 1. 人脸检测
        face_regions = self.face_detector.detect(thermal_image)
        
        if not face_regions:
            return {'error': 'No face detected'}
        
        # 2. 特征提取
        if clinical_data is not None:
            features = self.feature_extractor.extract_multimodal(
                face_regions[0], clinical_data
            )
        else:
            features = self.feature_extractor.extract_image_only(
                face_regions[0]
            )
        
        # 3. 风险预测
        risk_score = self.classifier.predict_proba(features)[0][1]
        risk_level = self._categorize_risk(risk_score)
        
        return {
            'risk_score': risk_score * 100,
            'risk_level': risk_level,
            'confidence': self.classifier.predict_proba(features).max(),
            'face_detected': True
        }
    
    def _categorize_risk(self, score):
        if score < 0.3:
            return 'low'
        elif score < 0.7:
            return 'medium'
        else:
            return 'high'
```

### 7.2 Web应用集成
```python
# Flask API接口
@app.route('/api/predict/icas', methods=['POST'])
def predict_icas():
    # 1. 接收上传的热力图
    thermal_image = request.files['thermal_image']
    clinical_data = request.json.get('clinical_data', None)
    
    # 2. 图像预处理
    image = preprocess_uploaded_image(thermal_image)
    
    # 3. 模型预测
    prediction = pipeline.predict(image, clinical_data)
    
    # 4. 返回结果
    return jsonify(prediction)
```

## 8. 质量控制与优化

### 8.1 数据质量控制
```python
# 图像质量检查
def quality_control_pipeline(image):
    checks = {
        'resolution_check': check_image_resolution(image),
        'brightness_check': check_brightness_range(image),
        'contrast_check': check_contrast_level(image),
        'noise_check': check_noise_level(image),
        'face_visibility_check': check_face_visibility(image)
    }
    
    quality_score = calculate_quality_score(checks)
    return quality_score > 0.7  # 质量阈值
```

### 8.2 模型持续优化
```python
# 在线学习与模型更新
class ModelUpdatePipeline:
    def __init__(self):
        self.feedback_buffer = []
        self.update_threshold = 100  # 累积100个反馈后更新
    
    def collect_feedback(self, prediction, ground_truth):
        self.feedback_buffer.append({
            'prediction': prediction,
            'ground_truth': ground_truth,
            'timestamp': datetime.now()
        })
        
        if len(self.feedback_buffer) >= self.update_threshold:
            self.update_model()
    
    def update_model(self):
        # 增量学习更新模型
        new_data = self.prepare_training_data(self.feedback_buffer)
        self.model.partial_fit(new_data['X'], new_data['y'])
        
        # 清空缓冲区
        self.feedback_buffer = []
```

## 9. 技术创新点

### 9.1 多模态特征融合
- **创新点**: 首次将热力图特征与临床特征深度融合
- **技术优势**: 相比单一图像特征，准确率提升23.08%
- **应用价值**: 更符合临床诊断的多维度评估需求

### 9.2 分块LBP纹理分析
- **创新点**: 4×4网格分块保留空间信息的LBP特征提取
- **技术优势**: 288维分块特征提供丰富的局部纹理描述
- **应用价值**: 捕捉人脸热力图的细微纹理变化

### 9.3 对比学习预训练
- **创新点**: 针对医学热力图的对比学习策略
- **技术优势**: 无监督预训练提升特征表达能力
- **应用价值**: 减少对标注数据的依赖

## 10. 总结

本算法流程通过系统性的数据处理、模型训练和性能优化，构建了一个完整的ICAS热力图分析系统。多模态特征融合方法在当前数据集上表现最佳，为临床ICAS风险评估提供了有效的技术支持。

**关键成果**:
- 构建了950张热力图的标准化数据集
- 实现了74.83%的分类准确率
- 开发了多种可选的分类方法
- 建立了完整的部署应用流程

**临床价值**:
- 提高ICAS早期筛查效率
- 辅助医生临床决策
- 降低漏诊误诊风险
- 支持个性化医疗建议

---

*文档版本: v1.0*  
*创建日期: 2025年1月*  
*维护团队: ICAS算法开发团队*