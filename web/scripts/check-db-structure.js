const database = require('../config/database');

async function checkDatabaseStructure() {
  try {
    await database.connect();
    
    // 查看patients表结构
    const tableInfo = await database.all("PRAGMA table_info(patients)");
    console.log('当前patients表结构:');
    tableInfo.forEach(col => {
      console.log(`- ${col.name}: ${col.type} ${col.notnull ? 'NOT NULL' : ''} ${col.dflt_value ? `DEFAULT ${col.dflt_value}` : ''}`);
    });
    
    // 查看表中的数据量
    const count = await database.get("SELECT COUNT(*) as count FROM patients");
    console.log(`\n当前患者数量: ${count.count}`);
    
  } catch (error) {
    console.error('检查数据库结构失败:', error);
  } finally {
    await database.close();
  }
}

checkDatabaseStructure();