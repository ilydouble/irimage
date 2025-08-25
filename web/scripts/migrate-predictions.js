const database = require('../config/database');

async function migratePredictions() {
  try {
    console.log('开始迁移预测数据...');
    
    // 连接数据库
    await database.connect();
    
    // 1. 首先检查当前数据
    const patientsWithPredictions = await database.all(`
      SELECT patient_id, predicted_icas, predicted_icas_grade, predicted_icas_confidence 
      FROM patients 
      WHERE predicted_icas IS NOT NULL OR predicted_icas_grade IS NOT NULL OR predicted_icas_confidence IS NOT NULL
    `);
    
    console.log(`找到 ${patientsWithPredictions.length} 个患者有预测数据`);
    
    // 2. 删除现有的predictions表并重新创建
    console.log('重新创建predictions表...');
    await database.run('DROP TABLE IF EXISTS predictions');
    
    await database.run(`
      CREATE TABLE predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id VARCHAR(20) NOT NULL,
        predicted_icas BOOLEAN DEFAULT 0,
        predicted_icas_grade TEXT CHECK(predicted_icas_grade IN ('无', '轻度', '中度', '重度')) DEFAULT '无',
        predicted_icas_confidence DECIMAL(5,2),
        factors TEXT,
        recommendations TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
      )
    `);
    
    await database.run('CREATE INDEX idx_predictions_patient ON predictions(patient_id)');
    
    // 3. 迁移数据到predictions表
    console.log('迁移预测数据到predictions表...');
    let migratedCount = 0;
    
    for (const patient of patientsWithPredictions) {
      // 只迁移有实际预测数据的患者
      if (patient.predicted_icas || patient.predicted_icas_grade !== '无' || patient.predicted_icas_confidence) {
        await database.run(`
          INSERT INTO predictions (
            patient_id, predicted_icas, predicted_icas_grade, predicted_icas_confidence, factors, recommendations
          ) VALUES (?, ?, ?, ?, ?, ?)
        `, [
          patient.patient_id,
          patient.predicted_icas ? 1 : 0,
          patient.predicted_icas_grade || '无',
          patient.predicted_icas_confidence,
          null, // factors - 可以后续填充
          null  // recommendations - 可以后续填充
        ]);
        migratedCount++;
      }
    }
    
    console.log(`成功迁移 ${migratedCount} 条预测记录`);
    
    // 4. 删除patients表中的预测字段
    console.log('从patients表中移除预测字段...');
    
    // SQLite不支持DROP COLUMN，所以需要重建表
    await database.run('BEGIN TRANSACTION');
    
    try {
      // 创建新的patients表结构
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
          
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
      `);
      
      // 复制数据（排除预测字段）
      await database.run(`
        INSERT INTO patients_new (
          id, patient_id, name, gender, age, height, weight, bmi, waist, hip, neck,
          thermal_24h, thermal_25h, voice_25h, has_icas, icas_grade, created_at, updated_at
        )
        SELECT 
          id, patient_id, name, gender, age, height, weight, bmi, waist, hip, neck,
          thermal_24h, thermal_25h, voice_25h, has_icas, icas_grade, created_at, updated_at
        FROM patients
      `);
      
      // 删除旧表，重命名新表
      await database.run('DROP TABLE patients');
      await database.run('ALTER TABLE patients_new RENAME TO patients');
      
      // 重新创建索引和触发器
      await database.run('CREATE INDEX idx_patient_id ON patients(patient_id)');
      await database.run('CREATE INDEX idx_patient_name ON patients(name)');
      await database.run('CREATE INDEX idx_icas_status ON patients(has_icas, icas_grade)');
      
      await database.run(`
        CREATE TRIGGER update_patients_timestamp 
        AFTER UPDATE ON patients
        BEGIN
          UPDATE patients SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
        END
      `);
      
      await database.run('COMMIT');
      console.log('成功重建patients表');
      
    } catch (error) {
      await database.run('ROLLBACK');
      throw error;
    }
    
    // 5. 验证迁移结果
    const finalPatientsCount = await database.get('SELECT COUNT(*) as count FROM patients');
    const finalPredictionsCount = await database.get('SELECT COUNT(*) as count FROM predictions');
    
    console.log('\n迁移完成！');
    console.log(`患者表记录数: ${finalPatientsCount.count}`);
    console.log(`预测表记录数: ${finalPredictionsCount.count}`);
    
    // 显示一些示例数据
    const samplePredictions = await database.all(`
      SELECT p.patient_id, p.name, pr.predicted_icas, pr.predicted_icas_grade, pr.predicted_icas_confidence
      FROM patients p
      LEFT JOIN predictions pr ON p.patient_id = pr.patient_id
      LIMIT 5
    `);
    
    console.log('\n示例数据:');
    samplePredictions.forEach(row => {
      console.log(`${row.patient_id} - ${row.name}: ICAS=${row.predicted_icas ? '是' : '否'}, 等级=${row.predicted_icas_grade || '无'}, 置信度=${row.predicted_icas_confidence || 'N/A'}`);
    });
    
  } catch (error) {
    console.error('迁移失败:', error);
    throw error;
  } finally {
    await database.close();
  }
}

// 运行迁移
if (require.main === module) {
  migratePredictions()
    .then(() => {
      console.log('数据迁移成功完成！');
      process.exit(0);
    })
    .catch((error) => {
      console.error('数据迁移失败:', error);
      process.exit(1);
    });
}

module.exports = { migratePredictions };
