const database = require('../config/database');

// SQL statements for creating tables
const createTables = {
  patients: `
    CREATE TABLE IF NOT EXISTS patients (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      patient_id VARCHAR(20) UNIQUE NOT NULL,
      name VARCHAR(50) NOT NULL,
      gender TEXT CHECK(gender IN ('male', 'female')) NOT NULL,
      age INTEGER, -- 移除NOT NULL约束
      
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
  `,

  predictions: `
    CREATE TABLE IF NOT EXISTS predictions (
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
  `,

  files: `
    CREATE TABLE IF NOT EXISTS files (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      patient_id VARCHAR(20) NOT NULL,
      file_type TEXT CHECK(file_type IN ('thermal_24h', 'thermal_25h', 'voice_25h')) NOT NULL,
      file_name VARCHAR(255) NOT NULL,
      file_path VARCHAR(500) NOT NULL,
      upload_time DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
    )
  `
};

// SQL statements for creating indexes
const createIndexes = [
  'CREATE INDEX IF NOT EXISTS idx_patient_id ON patients(patient_id)',
  'CREATE INDEX IF NOT EXISTS idx_patient_name ON patients(name)',
  'CREATE INDEX IF NOT EXISTS idx_icas_status ON patients(has_icas, icas_grade)',
  'CREATE INDEX IF NOT EXISTS idx_predictions_patient ON predictions(patient_id)',
  'CREATE INDEX IF NOT EXISTS idx_files_patient ON files(patient_id)',
  'CREATE INDEX IF NOT EXISTS idx_files_type ON files(file_type)'
];

// Trigger to update updated_at timestamp
const createTrigger = `
  CREATE TRIGGER IF NOT EXISTS update_patients_timestamp 
  AFTER UPDATE ON patients
  BEGIN
    UPDATE patients SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
  END
`;

async function initializeDatabase() {
  try {
    console.log('Initializing database...');
    
    // Connect to database
    await database.connect();
    
    // Create tables
    console.log('Creating tables...');
    for (const [tableName, sql] of Object.entries(createTables)) {
      await database.run(sql);
      console.log(`✓ Created table: ${tableName}`);
    }
    
    // Create indexes
    console.log('Creating indexes...');
    for (const sql of createIndexes) {
      await database.run(sql);
    }
    console.log('✓ Created indexes');
    
    // Create trigger
    await database.run(createTrigger);
    console.log('✓ Created triggers');
    
    console.log('Database initialization completed successfully!');
    
  } catch (error) {
    console.error('Error initializing database:', error);
    process.exit(1);
  } finally {
    await database.close();
  }
}

// Run initialization if this script is executed directly
if (require.main === module) {
  initializeDatabase();
}

module.exports = { initializeDatabase };
