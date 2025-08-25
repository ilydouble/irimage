# PatientCare 简化数据库设计

## 1. 患者信息表 (patients)
```sql
CREATE TABLE patients (
    id INT PRIMARY KEY AUTO_INCREMENT,
    patient_id VARCHAR(20) UNIQUE NOT NULL COMMENT '患者编号',
    name VARCHAR(50) NOT NULL COMMENT '姓名',
    gender ENUM('male', 'female') NOT NULL COMMENT '性别',
    age INT NOT NULL COMMENT '年龄',
    
    -- 基础体征
    height DECIMAL(5,2) COMMENT '身高(cm)',
    weight DECIMAL(5,2) COMMENT '体重(kg)',
    bmi DECIMAL(4,2) COMMENT 'BMI指数',
    waist DECIMAL(5,2) COMMENT '腰围(cm)',
    hip DECIMAL(5,2) COMMENT '臀围(cm)',
    neck DECIMAL(5,2) COMMENT '颈围(cm)',
    
    -- 检查项目
    thermal_24h BOOLEAN DEFAULT FALSE COMMENT '24年热成像',
    thermal_25h BOOLEAN DEFAULT FALSE COMMENT '25年热成像', 
    voice_25h BOOLEAN DEFAULT FALSE COMMENT '25年语音',
    
    -- ICAS相关
    has_icas BOOLEAN DEFAULT FALSE COMMENT 'ICAS诊断',
    icas_grade ENUM('无', '轻度', '中度', '重度') DEFAULT '无' COMMENT 'ICAS分级',
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

## 2. 预测结果表 (predictions)
```sql
CREATE TABLE predictions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    patient_id VARCHAR(20) NOT NULL,
    risk_score DECIMAL(5,2) NOT NULL COMMENT '风险评分(0-100)',
    risk_level ENUM('low', 'medium', 'high') NOT NULL COMMENT '风险等级',
    confidence DECIMAL(5,2) NOT NULL COMMENT '置信度(%)',
    factors JSON COMMENT '影响因子',
    recommendations TEXT COMMENT '建议',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);
```

## 3. 文件记录表 (files)
```sql
CREATE TABLE files (
    id INT PRIMARY KEY AUTO_INCREMENT,
    patient_id VARCHAR(20) NOT NULL,
    file_type ENUM('thermal_24h', 'thermal_25h', 'voice_25h') NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);
```

## 4. 索引
```sql
CREATE INDEX idx_patient_id ON patients(patient_id);
CREATE INDEX idx_patient_name ON patients(name);
CREATE INDEX idx_icas_status ON patients(has_icas, icas_grade);
```

## 5. 示例数据
```sql
INSERT INTO patients (patient_id, name, gender, age, height, weight, waist, hip, neck, thermal_24h, thermal_25h, voice_25h, has_icas, icas_grade) VALUES
('101EC', '张广磊', 'male', 42, 165.5, 91.7, 91.7, 97.8, 37.5, true, true, true, false, '无'),
('045EC', '徐淑燕', 'female', 43, 156.5, 78, 78, 101, 33, true, true, true, false, '无'),
('051FE', '赵波', 'male', 43, 173, 94, 94, 99, 38, true, true, true, false, '无');
```
