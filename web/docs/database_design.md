# PatientCare 数据库设计文档

## 1. 数据库概述

### 1.1 设计原则
- 数据完整性和一致性
- 支持多模态医疗数据存储
- 高效的查询性能
- 数据安全和隐私保护

### 1.2 技术选型
- **数据库类型：** 关系型数据库 (PostgreSQL/MySQL)
- **文件存储：** 对象存储 (用于图片、音频文件)
- **索引策略：** 复合索引优化查询性能

## 2. 核心数据表设计

### 2.1 患者基础信息表 (patients)
```sql
CREATE TABLE patients (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    patient_id VARCHAR(50) UNIQUE NOT NULL COMMENT '患者编号',
    name VARCHAR(100) NOT NULL COMMENT '患者姓名',
    gender ENUM('male', 'female') NOT NULL COMMENT '性别',
    age INT NOT NULL COMMENT '年龄',
    phone VARCHAR(20) COMMENT '联系电话',
    email VARCHAR(100) COMMENT '邮箱',
    address TEXT COMMENT '地址',
    emergency_contact VARCHAR(100) COMMENT '紧急联系人',
    emergency_phone VARCHAR(20) COMMENT '紧急联系电话',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_patient_id (patient_id),
    INDEX idx_name (name),
    INDEX idx_gender_age (gender, age)
);
```

### 2.2 体征测量表 (vital_signs)
```sql
CREATE TABLE vital_signs (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    patient_id VARCHAR(50) NOT NULL,
    height DECIMAL(5,2) COMMENT '身高(cm)',
    weight DECIMAL(5,2) COMMENT '体重(kg)',
    bmi DECIMAL(4,2) COMMENT 'BMI指数',
    waist_circumference DECIMAL(5,2) COMMENT '腰围(cm)',
    hip_circumference DECIMAL(5,2) COMMENT '臀围(cm)',
    neck_circumference DECIMAL(5,2) COMMENT '颈围(cm)',
    systolic_bp INT COMMENT '收缩压',
    diastolic_bp INT COMMENT '舒张压',
    heart_rate INT COMMENT '心率',
    measured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    INDEX idx_patient_measured (patient_id, measured_at)
);
```

### 2.3 医疗历史表 (medical_history)
```sql
CREATE TABLE medical_history (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    patient_id VARCHAR(50) NOT NULL,
    diabetes_type ENUM('none', 'type1', 'type2') DEFAULT 'none',
    hypertension BOOLEAN DEFAULT FALSE,
    smoking_status ENUM('never', 'former', 'current') DEFAULT 'never',
    alcohol_consumption ENUM('none', 'light', 'moderate', 'heavy') DEFAULT 'none',
    exercise_frequency ENUM('none', 'rare', 'regular', 'frequent') DEFAULT 'none',
    family_history_stroke BOOLEAN DEFAULT FALSE,
    family_history_diabetes BOOLEAN DEFAULT FALSE,
    medications TEXT COMMENT '当前用药',
    allergies TEXT COMMENT '过敏史',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    INDEX idx_patient_id (patient_id)
);
```

### 2.4 实验室检查表 (lab_results)
```sql
CREATE TABLE lab_results (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    patient_id VARCHAR(50) NOT NULL,
    test_date DATE NOT NULL,
    total_cholesterol DECIMAL(5,2) COMMENT '总胆固醇(mmol/L)',
    ldl_cholesterol DECIMAL(5,2) COMMENT '低密度脂蛋白(mmol/L)',
    hdl_cholesterol DECIMAL(5,2) COMMENT '高密度脂蛋白(mmol/L)',
    triglycerides DECIMAL(5,2) COMMENT '甘油三酯(mmol/L)',
    glucose DECIMAL(5,2) COMMENT '血糖(mmol/L)',
    hba1c DECIMAL(4,2) COMMENT '糖化血红蛋白(%)',
    creatinine DECIMAL(6,2) COMMENT '肌酐(μmol/L)',
    uric_acid DECIMAL(6,2) COMMENT '尿酸(μmol/L)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    INDEX idx_patient_test_date (patient_id, test_date)
);
```

### 2.5 热成像数据表 (thermal_imaging)
```sql
CREATE TABLE thermal_imaging (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    patient_id VARCHAR(50) NOT NULL,
    image_type ENUM('thermal_24h', 'thermal_25h') NOT NULL,
    file_path VARCHAR(500) NOT NULL COMMENT '文件存储路径',
    file_name VARCHAR(255) NOT NULL,
    file_size BIGINT COMMENT '文件大小(bytes)',
    image_features JSON COMMENT 'AI提取的图像特征',
    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed BOOLEAN DEFAULT FALSE,
    processed_at TIMESTAMP NULL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    INDEX idx_patient_type (patient_id, image_type),
    INDEX idx_processed (processed)
);
```

### 2.6 语音数据表 (voice_data)
```sql
CREATE TABLE voice_data (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    patient_id VARCHAR(50) NOT NULL,
    voice_type ENUM('voice_25h') NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_size BIGINT,
    duration DECIMAL(8,2) COMMENT '时长(秒)',
    voice_features JSON COMMENT 'AI提取的语音特征',
    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed BOOLEAN DEFAULT FALSE,
    processed_at TIMESTAMP NULL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    INDEX idx_patient_type (patient_id, voice_type),
    INDEX idx_processed (processed)
);
```

### 2.7 ICAS风险预测表 (icas_predictions)
```sql
CREATE TABLE icas_predictions (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    patient_id VARCHAR(50) NOT NULL,
    prediction_id VARCHAR(100) UNIQUE NOT NULL,
    risk_score DECIMAL(5,2) NOT NULL COMMENT '风险评分(0-100)',
    risk_level ENUM('low', 'medium', 'high') NOT NULL,
    confidence DECIMAL(5,2) NOT NULL COMMENT '置信度(%)',
    model_version VARCHAR(50) NOT NULL COMMENT '模型版本',
    input_features JSON NOT NULL COMMENT '输入特征数据',
    risk_factors JSON COMMENT '风险因子分析',
    recommendations JSON COMMENT '个性化建议',
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) COMMENT '创建者',
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    INDEX idx_patient_predicted (patient_id, predicted_at),
    INDEX idx_risk_level (risk_level),
    INDEX idx_prediction_id (prediction_id)
);
```

### 2.8 预测报告表 (prediction_reports)
```sql
CREATE TABLE prediction_reports (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    prediction_id VARCHAR(100) NOT NULL,
    report_type ENUM('pdf', 'json', 'xml') DEFAULT 'pdf',
    file_path VARCHAR(500),
    file_name VARCHAR(255),
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    downloaded_count INT DEFAULT 0,
    last_downloaded_at TIMESTAMP NULL,
    FOREIGN KEY (prediction_id) REFERENCES icas_predictions(prediction_id),
    INDEX idx_prediction_id (prediction_id)
);
```

### 2.9 系统日志表 (system_logs)
```sql
CREATE TABLE system_logs (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id VARCHAR(100),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(100),
    ip_address VARCHAR(45),
    user_agent TEXT,
    details JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_action (user_id, action),
    INDEX idx_created_at (created_at),
    INDEX idx_resource (resource_type, resource_id)
);
```

## 3. 数据关系图

```
patients (1) ←→ (N) vital_signs
patients (1) ←→ (1) medical_history  
patients (1) ←→ (N) lab_results
patients (1) ←→ (N) thermal_imaging
patients (1) ←→ (N) voice_data
patients (1) ←→ (N) icas_predictions
icas_predictions (1) ←→ (N) prediction_reports
```

## 4. 索引优化策略

### 4.1 查询优化索引
```sql
-- 患者搜索优化
CREATE INDEX idx_patient_search ON patients(name, patient_id, gender, age);

-- 预测结果查询优化
CREATE INDEX idx_prediction_search ON icas_predictions(patient_id, predicted_at, risk_level);

-- 文件处理状态查询
CREATE INDEX idx_file_processing ON thermal_imaging(processed, upload_time);
CREATE INDEX idx_voice_processing ON voice_data(processed, upload_time);
```

### 4.2 分区策略
```sql
-- 按时间分区大表
ALTER TABLE system_logs PARTITION BY RANGE (YEAR(created_at)) (
    PARTITION p2024 VALUES LESS THAN (2025),
    PARTITION p2025 VALUES LESS THAN (2026),
    PARTITION p_future VALUES LESS THAN MAXVALUE
);
```

## 5. 数据安全设计

### 5.1 敏感数据加密
- 患者姓名、联系方式等PII数据加密存储
- 医疗数据传输使用TLS加密
- 数据库连接使用SSL

### 5.2 访问控制
- 基于角色的权限控制(RBAC)
- 数据访问审计日志
- 定期数据备份和恢复测试

## 6. 性能优化建议

### 6.1 查询优化
- 使用适当的索引策略
- 避免全表扫描
- 合理使用缓存机制

### 6.2 存储优化
- 大文件使用对象存储
- 历史数据归档策略
- 定期清理临时数据

---

**文档版本：** v1.0  
**创建日期：** 2024年  
**维护团队：** PatientCare开发团队