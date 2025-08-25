const express = require('express');
const database = require('../config/database');

const router = express.Router();

// GET /api/analytics/overview - Get system overview statistics
router.get('/overview', async (req, res) => {
  try {
    await database.connect();
    
    // Get basic counts
    const patientCount = await database.get('SELECT COUNT(*) as count FROM patients');
    const predictionCount = await database.get('SELECT COUNT(*) as count FROM predictions');
    const fileCount = await database.get('SELECT COUNT(*) as count FROM files');
    
    // Get ICAS prediction distribution (using predicted_icas_grade as risk level)
    const riskDistribution = await database.all(`
      SELECT predicted_icas_grade as risk_level, COUNT(*) as count
      FROM predictions
      GROUP BY predicted_icas_grade
    `);
    
    // Get gender distribution
    const genderDistribution = await database.all(`
      SELECT gender, COUNT(*) as count 
      FROM patients 
      GROUP BY gender
    `);
    
    // Get ICAS distribution
    const icasDistribution = await database.all(`
      SELECT icas_grade, COUNT(*) as count 
      FROM patients 
      GROUP BY icas_grade
    `);
    
    // Get age distribution
    const ageDistribution = await database.all(`
      SELECT 
        CASE 
          WHEN age < 30 THEN '< 30'
          WHEN age < 40 THEN '30-39'
          WHEN age < 50 THEN '40-49'
          WHEN age < 60 THEN '50-59'
          ELSE '60+'
        END as age_group,
        COUNT(*) as count
      FROM patients 
      GROUP BY age_group
      ORDER BY age_group
    `);
    
    // Get recent activity (last 30 days)
    const recentPredictions = await database.get(`
      SELECT COUNT(*) as count 
      FROM predictions 
      WHERE created_at >= datetime('now', '-30 days')
    `);
    
    const recentUploads = await database.get(`
      SELECT COUNT(*) as count 
      FROM files 
      WHERE upload_time >= datetime('now', '-30 days')
    `);
    
    res.json({
      overview: {
        total_patients: patientCount.count,
        total_predictions: predictionCount.count,
        total_files: fileCount.count,
        recent_predictions: recentPredictions.count,
        recent_uploads: recentUploads.count
      },
      distributions: {
        risk_levels: riskDistribution,
        genders: genderDistribution,
        icas_grades: icasDistribution,
        age_groups: ageDistribution
      }
    });
    
  } catch (error) {
    console.error('Error fetching overview analytics:', error);
    res.status(500).json({ error: 'Failed to fetch analytics overview' });
  }
});

// GET /api/analytics/predictions - Get prediction analytics
router.get('/predictions', async (req, res) => {
  try {
    await database.connect();
    
    const { period = '30' } = req.query; // days
    
    // Predictions over time
    const predictionsOverTime = await database.all(`
      SELECT 
        DATE(created_at) as date,
        COUNT(*) as count,
        AVG(risk_score) as avg_risk_score
      FROM predictions 
      WHERE created_at >= datetime('now', '-${period} days')
      GROUP BY DATE(created_at)
      ORDER BY date
    `);
    
    // Risk score distribution
    const riskScoreDistribution = await database.all(`
      SELECT 
        CASE 
          WHEN risk_score < 20 THEN '0-19'
          WHEN risk_score < 40 THEN '20-39'
          WHEN risk_score < 60 THEN '40-59'
          WHEN risk_score < 80 THEN '60-79'
          ELSE '80-100'
        END as score_range,
        COUNT(*) as count
      FROM predictions 
      GROUP BY score_range
      ORDER BY score_range
    `);
    
    // Confidence distribution
    const confidenceStats = await database.get(`
      SELECT 
        AVG(confidence) as avg_confidence,
        MIN(confidence) as min_confidence,
        MAX(confidence) as max_confidence,
        COUNT(*) as total_predictions
      FROM predictions
    `);
    
    // Risk level trends
    const riskTrends = await database.all(`
      SELECT 
        DATE(created_at) as date,
        risk_level,
        COUNT(*) as count
      FROM predictions 
      WHERE created_at >= datetime('now', '-${period} days')
      GROUP BY DATE(created_at), risk_level
      ORDER BY date, risk_level
    `);
    
    res.json({
      predictions_over_time: predictionsOverTime,
      risk_score_distribution: riskScoreDistribution,
      confidence_stats: confidenceStats,
      risk_trends: riskTrends
    });
    
  } catch (error) {
    console.error('Error fetching prediction analytics:', error);
    res.status(500).json({ error: 'Failed to fetch prediction analytics' });
  }
});

// GET /api/analytics/patients - Get patient analytics
router.get('/patients', async (req, res) => {
  try {
    await database.connect();
    
    // BMI distribution
    const bmiDistribution = await database.all(`
      SELECT 
        CASE 
          WHEN bmi IS NULL THEN 'Unknown'
          WHEN bmi < 18.5 THEN 'Underweight'
          WHEN bmi < 24 THEN 'Normal'
          WHEN bmi < 28 THEN 'Overweight'
          ELSE 'Obese'
        END as bmi_category,
        COUNT(*) as count
      FROM patients 
      GROUP BY bmi_category
      ORDER BY bmi_category
    `);
    
    // File completion rates
    const fileCompletion = await database.get(`
      SELECT 
        SUM(thermal_24h) as thermal_24h_count,
        SUM(thermal_25h) as thermal_25h_count,
        SUM(voice_25h) as voice_25h_count,
        COUNT(*) as total_patients
      FROM patients
    `);
    
    // Age and gender correlation with ICAS
    const icasCorrelation = await database.all(`
      SELECT 
        gender,
        CASE 
          WHEN age < 40 THEN '< 40'
          WHEN age < 50 THEN '40-49'
          ELSE '50+'
        END as age_group,
        SUM(has_icas) as icas_count,
        COUNT(*) as total_count,
        ROUND(CAST(SUM(has_icas) AS FLOAT) / COUNT(*) * 100, 2) as icas_percentage
      FROM patients 
      GROUP BY gender, age_group
      ORDER BY gender, age_group
    `);
    
    // Physical measurements statistics
    const physicalStats = await database.get(`
      SELECT 
        AVG(height) as avg_height,
        AVG(weight) as avg_weight,
        AVG(bmi) as avg_bmi,
        AVG(waist) as avg_waist,
        AVG(hip) as avg_hip,
        AVG(neck) as avg_neck
      FROM patients 
      WHERE height IS NOT NULL AND weight IS NOT NULL
    `);
    
    res.json({
      bmi_distribution: bmiDistribution,
      file_completion: {
        thermal_24h_rate: (fileCompletion.thermal_24h_count / fileCompletion.total_patients * 100).toFixed(2),
        thermal_25h_rate: (fileCompletion.thermal_25h_count / fileCompletion.total_patients * 100).toFixed(2),
        voice_25h_rate: (fileCompletion.voice_25h_count / fileCompletion.total_patients * 100).toFixed(2),
        ...fileCompletion
      },
      icas_correlation: icasCorrelation,
      physical_stats: physicalStats
    });
    
  } catch (error) {
    console.error('Error fetching patient analytics:', error);
    res.status(500).json({ error: 'Failed to fetch patient analytics' });
  }
});

// GET /api/analytics/files - Get file analytics
router.get('/files', async (req, res) => {
  try {
    await database.connect();
    
    // File type distribution
    const fileTypeDistribution = await database.all(`
      SELECT file_type, COUNT(*) as count 
      FROM files 
      GROUP BY file_type
    `);
    
    // Upload trends (last 30 days)
    const uploadTrends = await database.all(`
      SELECT 
        DATE(upload_time) as date,
        file_type,
        COUNT(*) as count
      FROM files 
      WHERE upload_time >= datetime('now', '-30 days')
      GROUP BY DATE(upload_time), file_type
      ORDER BY date, file_type
    `);
    
    // Storage statistics (would need file size info)
    const storageStats = await database.all(`
      SELECT 
        file_type,
        COUNT(*) as file_count
      FROM files 
      GROUP BY file_type
    `);
    
    res.json({
      file_type_distribution: fileTypeDistribution,
      upload_trends: uploadTrends,
      storage_stats: storageStats
    });
    
  } catch (error) {
    console.error('Error fetching file analytics:', error);
    res.status(500).json({ error: 'Failed to fetch file analytics' });
  }
});

// GET /api/analytics/export - Export analytics data
router.get('/export', async (req, res) => {
  try {
    await database.connect();
    
    const { type = 'overview', format = 'json' } = req.query;
    
    let data = {};
    
    switch (type) {
      case 'patients':
        data = await database.all(`
          SELECT 
            patient_id, name, gender, age, bmi, has_icas, icas_grade,
            thermal_24h, thermal_25h, voice_25h, created_at
          FROM patients 
          ORDER BY created_at DESC
        `);
        break;
        
      case 'predictions':
        data = await database.all(`
          SELECT 
            p.patient_id, pt.name, p.risk_score, p.risk_level, 
            p.confidence, p.recommendations, p.created_at
          FROM predictions p
          LEFT JOIN patients pt ON p.patient_id = pt.patient_id
          ORDER BY p.created_at DESC
        `);
        break;
        
      case 'files':
        data = await database.all(`
          SELECT 
            f.patient_id, pt.name, f.file_type, f.file_name, f.upload_time
          FROM files f
          LEFT JOIN patients pt ON f.patient_id = pt.patient_id
          ORDER BY f.upload_time DESC
        `);
        break;
        
      default:
        // Overview export
        const patients = await database.all('SELECT * FROM patients');
        const predictions = await database.all(`
          SELECT p.*, pt.name as patient_name 
          FROM predictions p 
          LEFT JOIN patients pt ON p.patient_id = pt.patient_id
        `);
        const files = await database.all(`
          SELECT f.*, pt.name as patient_name 
          FROM files f 
          LEFT JOIN patients pt ON f.patient_id = pt.patient_id
        `);
        
        data = { patients, predictions, files };
    }
    
    if (format === 'csv' && Array.isArray(data)) {
      // Simple CSV conversion for arrays
      if (data.length === 0) {
        return res.status(404).json({ error: 'No data to export' });
      }
      
      const headers = Object.keys(data[0]).join(',');
      const rows = data.map(row => 
        Object.values(row).map(val => 
          typeof val === 'string' ? `"${val.replace(/"/g, '""')}"` : val
        ).join(',')
      );
      
      const csv = [headers, ...rows].join('\n');
      
      res.setHeader('Content-Type', 'text/csv');
      res.setHeader('Content-Disposition', `attachment; filename="${type}_export.csv"`);
      res.send(csv);
    } else {
      res.json(data);
    }
    
  } catch (error) {
    console.error('Error exporting analytics:', error);
    res.status(500).json({ error: 'Failed to export analytics data' });
  }
});

module.exports = router;
