const database = require('../config/database');

async function rebuildPatientsTable() {
  try {
    await database.connect();
    
    console.log('开始重建patients表...');
    
    // 1. 创建新表结构（去掉不需要的字段）
    await database.run(`
      CREATE TABLE patients_new (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id VARCHAR(20) UNIQUE NOT NULL,
        name VARCHAR(50) NOT NULL,
        gender TEXT CHECK(gender IN ('male', 'female')) NOT NULL,
        age INTEGER,
        
        -- 基础体征
        height DECIMAL(5,2),
        weight DECIMAL(5,2),
        bmi DECIMAL(4,2),
        waist DECIMAL(5,2),
        hip DECIMAL(5,2),
        neck DECIMAL(5,2),
        
        -- 检查项目
        thermal_24h BOOLEAN DEFAULT 0,
        thermal_25h BOOLEAN DEFAULT 0,
        voice_25h BOOLEAN DEFAULT 0,
        
        -- ICAS相关
        has_icas BOOLEAN DEFAULT 0,
        icas_grade TEXT CHECK(icas_grade IN ('无', '轻度', '中度', '重度')) DEFAULT '无',
        
        -- 时间戳
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    `);
    
    // 2. 复制数据（只复制需要的字段）
    await database.run(`
      INSERT INTO patients_new (
        patient_id, name, gender, age, height, weight, bmi, waist, hip, neck,
        thermal_24h, thermal_25h, voice_25h, has_icas, icas_grade,
        created_at, updated_at
      )
      SELECT 
        patient_id, name, gender, age, height, weight, bmi, waist, hip, neck,
        thermal_24h, thermal_25h, voice_25h, has_icas, icas_grade,
        created_at, updated_at
      FROM patients
    `);
    
    // 3. 删除旧表
    await database.run('DROP TABLE patients');
    
    // 4. 重命名新表
    await database.run('ALTER TABLE patients_new RENAME TO patients');
    
    // 5. 重建索引
    await database.run('CREATE INDEX IF NOT EXISTS idx_patient_id ON patients(patient_id)');
    await database.run('CREATE INDEX IF NOT EXISTS idx_patient_name ON patients(name)');
    await database.run('CREATE INDEX IF NOT EXISTS idx_icas_status ON patients(has_icas, icas_grade)');
    
    // 6. 重建触发器
    await database.run(`
      CREATE TRIGGER IF NOT EXISTS update_patients_timestamp 
      AFTER UPDATE ON patients
      BEGIN
        UPDATE patients SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
      END
    `);
    
    console.log('✓ patients表重建完成');
    
  } catch (error) {
    console.error('重建表失败:', error);
  } finally {
    await database.close();
  }
}

rebuildPatientsTable();