#!/usr/bin/env node

const fs = require('fs-extra');
const path = require('path');
const database = require('../config/database');

// Command line argument parsing
const args = process.argv.slice(2);
const command = args[0];

// Available commands
const commands = {
  backup: backupDatabase,
  restore: restoreDatabase,
  export: exportData,
  import: importData,
  reset: resetDatabase,
  status: showStatus,
  help: showHelp
};

// Main function
async function main() {
  if (!command || !commands[command]) {
    console.log('Invalid command. Use "help" to see available commands.');
    process.exit(1);
  }
  
  try {
    await commands[command]();
  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
}

// Backup database
async function backupDatabase() {
  const backupDir = path.join(__dirname, '..', 'backups');
  fs.ensureDirSync(backupDir);
  
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const backupFile = path.join(backupDir, `backup_${timestamp}.db`);
  
  const dbPath = path.join(__dirname, '..', 'database', 'patientcare.db');
  
  if (!fs.existsSync(dbPath)) {
    throw new Error('Database file not found. Run "npm run init-db" first.');
  }
  
  fs.copySync(dbPath, backupFile);
  console.log(`Database backed up to: ${backupFile}`);
}

// Restore database from backup
async function restoreDatabase() {
  const backupFile = args[1];
  
  if (!backupFile) {
    throw new Error('Please specify backup file path');
  }
  
  if (!fs.existsSync(backupFile)) {
    throw new Error('Backup file not found');
  }
  
  const dbPath = path.join(__dirname, '..', 'database', 'patientcare.db');
  
  // Create backup of current database
  if (fs.existsSync(dbPath)) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const currentBackup = path.join(path.dirname(dbPath), `current_backup_${timestamp}.db`);
    fs.copySync(dbPath, currentBackup);
    console.log(`Current database backed up to: ${currentBackup}`);
  }
  
  fs.copySync(backupFile, dbPath);
  console.log(`Database restored from: ${backupFile}`);
}

// Export data to JSON
async function exportData() {
  const exportType = args[1] || 'all';
  const outputFile = args[2] || `export_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
  
  await database.connect();
  
  let data = {};
  
  if (exportType === 'all' || exportType === 'patients') {
    data.patients = await database.all('SELECT * FROM patients ORDER BY created_at');
  }
  
  if (exportType === 'all' || exportType === 'predictions') {
    data.predictions = await database.all('SELECT * FROM predictions ORDER BY created_at');
  }
  
  if (exportType === 'all' || exportType === 'files') {
    data.files = await database.all('SELECT * FROM files ORDER BY upload_time');
  }
  
  await database.close();
  
  const exportDir = path.join(__dirname, '..', 'exports');
  fs.ensureDirSync(exportDir);
  
  const exportPath = path.join(exportDir, outputFile);
  fs.writeJsonSync(exportPath, data, { spaces: 2 });
  
  console.log(`Data exported to: ${exportPath}`);
  console.log(`Export type: ${exportType}`);
  
  if (data.patients) console.log(`Patients: ${data.patients.length}`);
  if (data.predictions) console.log(`Predictions: ${data.predictions.length}`);
  if (data.files) console.log(`Files: ${data.files.length}`);
}

// Import data from JSON
async function importData() {
  const importFile = args[1];
  
  if (!importFile) {
    throw new Error('Please specify import file path');
  }
  
  if (!fs.existsSync(importFile)) {
    throw new Error('Import file not found');
  }
  
  const data = fs.readJsonSync(importFile);
  
  await database.connect();
  
  let imported = { patients: 0, predictions: 0, files: 0 };
  
  // Import patients
  if (data.patients && Array.isArray(data.patients)) {
    console.log('Importing patients...');
    
    for (const patient of data.patients) {
      try {
        const sql = `
          INSERT OR REPLACE INTO patients (
            patient_id, name, gender, age, height, weight, bmi, waist, hip, neck,
            thermal_24h, thermal_25h, voice_25h, has_icas, icas_grade, created_at, updated_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `;
        
        await database.run(sql, [
          patient.patient_id, patient.name, patient.gender, patient.age,
          patient.height, patient.weight, patient.bmi, patient.waist,
          patient.hip, patient.neck, patient.thermal_24h, patient.thermal_25h,
          patient.voice_25h, patient.has_icas, patient.icas_grade,
          patient.created_at, patient.updated_at
        ]);
        
        imported.patients++;
      } catch (error) {
        console.warn(`Failed to import patient ${patient.patient_id}:`, error.message);
      }
    }
  }
  
  // Import predictions
  if (data.predictions && Array.isArray(data.predictions)) {
    console.log('Importing predictions...');
    
    for (const prediction of data.predictions) {
      try {
        const sql = `
          INSERT OR REPLACE INTO predictions (
            patient_id, risk_score, risk_level, confidence, factors, recommendations, created_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?)
        `;
        
        await database.run(sql, [
          prediction.patient_id, prediction.risk_score, prediction.risk_level,
          prediction.confidence, prediction.factors, prediction.recommendations,
          prediction.created_at
        ]);
        
        imported.predictions++;
      } catch (error) {
        console.warn(`Failed to import prediction for ${prediction.patient_id}:`, error.message);
      }
    }
  }
  
  // Import files
  if (data.files && Array.isArray(data.files)) {
    console.log('Importing file records...');
    
    for (const file of data.files) {
      try {
        const sql = `
          INSERT OR REPLACE INTO files (
            patient_id, file_type, file_name, file_path, upload_time
          ) VALUES (?, ?, ?, ?, ?)
        `;
        
        await database.run(sql, [
          file.patient_id, file.file_type, file.file_name,
          file.file_path, file.upload_time
        ]);
        
        imported.files++;
      } catch (error) {
        console.warn(`Failed to import file ${file.file_name}:`, error.message);
      }
    }
  }
  
  await database.close();
  
  console.log('Import completed:');
  console.log(`Patients: ${imported.patients}`);
  console.log(`Predictions: ${imported.predictions}`);
  console.log(`Files: ${imported.files}`);
}

// Reset database (clear all data)
async function resetDatabase() {
  const confirm = args[1];
  
  if (confirm !== '--confirm') {
    console.log('This will delete all data in the database!');
    console.log('Use: npm run db-manager reset --confirm');
    return;
  }
  
  await database.connect();
  
  // Clear all tables
  await database.run('DELETE FROM files');
  await database.run('DELETE FROM predictions');
  await database.run('DELETE FROM patients');
  
  // Reset auto-increment counters
  await database.run('DELETE FROM sqlite_sequence');
  
  await database.close();
  
  console.log('Database reset completed. All data has been deleted.');
}

// Show database status
async function showStatus() {
  await database.connect();
  
  const patientCount = await database.get('SELECT COUNT(*) as count FROM patients');
  const predictionCount = await database.get('SELECT COUNT(*) as count FROM predictions');
  const fileCount = await database.get('SELECT COUNT(*) as count FROM files');
  
  const dbPath = path.join(__dirname, '..', 'database', 'patientcare.db');
  const dbStats = fs.statSync(dbPath);
  
  console.log('Database Status:');
  console.log('================');
  console.log(`Database file: ${dbPath}`);
  console.log(`File size: ${(dbStats.size / 1024).toFixed(2)} KB`);
  console.log(`Last modified: ${dbStats.mtime.toISOString()}`);
  console.log('');
  console.log('Record counts:');
  console.log(`Patients: ${patientCount.count}`);
  console.log(`Predictions: ${predictionCount.count}`);
  console.log(`Files: ${fileCount.count}`);
  
  await database.close();
}

// Show help
function showHelp() {
  console.log('Database Manager Commands:');
  console.log('==========================');
  console.log('');
  console.log('backup                    - Create a backup of the database');
  console.log('restore <backup_file>     - Restore database from backup');
  console.log('export [type] [file]      - Export data to JSON (type: all|patients|predictions|files)');
  console.log('import <json_file>        - Import data from JSON file');
  console.log('reset --confirm           - Reset database (delete all data)');
  console.log('status                    - Show database status');
  console.log('help                      - Show this help message');
  console.log('');
  console.log('Examples:');
  console.log('npm run db-manager backup');
  console.log('npm run db-manager export patients patients.json');
  console.log('npm run db-manager import data.json');
  console.log('npm run db-manager restore backups/backup_2024-01-01.db');
}

// Run the main function
if (require.main === module) {
  main();
}

module.exports = {
  backupDatabase,
  restoreDatabase,
  exportData,
  importData,
  resetDatabase,
  showStatus
};
