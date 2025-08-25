const express = require('express');
const Joi = require('joi');
const database = require('../config/database');
const predictionService = require('../services/predictionService');

const router = express.Router();

// Validation schemas
const predictionRequestSchema = Joi.object({
  patient_id: Joi.string().max(20).required()
});

const predictionSchema = Joi.object({
  patient_id: Joi.string().required(),
  predicted_icas: Joi.boolean().required(),
  predicted_icas_grade: Joi.string().valid('无', '轻度', '中度', '重度').required(),
  predicted_icas_confidence: Joi.number().min(0).max(100).optional(),
  factors: Joi.string().optional(),
  recommendations: Joi.string().optional()
});

// GET /api/predictions - Get all predictions with optional filtering
router.get('/', async (req, res) => {
  try {
    // Ensure database is connected
    if (!database.db) {
      await database.connect();
    }

    const {
      patient_id,
      predicted_icas,
      predicted_icas_grade,
      page = 1,
      limit = 10
    } = req.query;

    let sql = `
      SELECT * FROM predictions
      WHERE 1=1
    `;
    const params = [];
    
    // Add filters
    if (patient_id) {
      sql += ' AND patient_id = ?';
      params.push(patient_id);
    }

    if (predicted_icas !== undefined) {
      sql += ' AND predicted_icas = ?';
      params.push(predicted_icas === 'true' ? 1 : 0);
    }

    if (predicted_icas_grade) {
      sql += ' AND predicted_icas_grade = ?';
      params.push(predicted_icas_grade);
    }
    
    // Add pagination
    const offset = (page - 1) * limit;
    sql += ' ORDER BY created_at DESC LIMIT ? OFFSET ?';
    params.push(parseInt(limit), offset);
    
    const predictions = await database.all(sql, params);

    // Add patient names to predictions
    const processedPredictions = [];
    for (const prediction of predictions) {
      const patient = await database.get('SELECT name, gender, age FROM patients WHERE patient_id = ?', [prediction.patient_id]);
      processedPredictions.push({
        ...prediction,
        patient_name: patient ? patient.name : null,
        gender: patient ? patient.gender : null,
        age: patient ? patient.age : null
      });
    }
    
    // Get total count
    let countSql = `
      SELECT COUNT(*) as total
      FROM predictions
      WHERE 1=1
    `;
    const countParams = [];

    if (patient_id) {
      countSql += ' AND patient_id = ?';
      countParams.push(patient_id);
    }

    if (predicted_icas_grade) {
      countSql += ' AND predicted_icas_grade = ?';
      countParams.push(predicted_icas_grade);
    }
    
    const countResult = await database.get(countSql, countParams);
    
    res.json({
      predictions: processedPredictions,
      pagination: {
        page: parseInt(page),
        limit: parseInt(limit),
        total: countResult.total,
        pages: Math.ceil(countResult.total / limit)
      }
    });
    
  } catch (error) {
    console.error('Error fetching predictions:', error);
    console.error('Error stack:', error.stack);
    res.status(500).json({ error: 'Failed to fetch predictions', details: error.message });
  }
});

// GET /api/predictions/:id - Get a specific prediction
router.get('/:id', async (req, res) => {
  try {
    
    const prediction = await database.get(`
      SELECT p.*, pt.name as patient_name, pt.gender, pt.age 
      FROM predictions p 
      LEFT JOIN patients pt ON p.patient_id = pt.patient_id 
      WHERE p.id = ?
    `, [req.params.id]);
    
    if (!prediction) {
      return res.status(404).json({ error: 'Prediction not found' });
    }
    
    // Parse JSON factors
    prediction.factors = prediction.factors ? JSON.parse(prediction.factors) : {};
    
    res.json(prediction);
    
  } catch (error) {
    console.error('Error fetching prediction:', error);
    res.status(500).json({ error: 'Failed to fetch prediction' });
  }
});

// POST /api/predictions - Create a new prediction
router.post('/', async (req, res) => {
  try {
    // Validate input
    const { error, value } = predictionSchema.validate(req.body);
    if (error) {
      return res.status(400).json({ 
        error: 'Validation failed', 
        details: error.details 
      });
    }

    // Check if patient exists
    const patient = await database.get(
      'SELECT * FROM patients WHERE patient_id = ?',
      [value.patient_id]
    );
    
    if (!patient) {
      return res.status(404).json({ error: 'Patient not found' });
    }
    
    // Get prediction from service
    console.log('Generating prediction for patient:', value.patient_id);
    const predictionResult = await predictionService.predict(patient);
    
    // Save prediction to database
    const sql = `
      INSERT INTO predictions (
        patient_id, risk_score, risk_level, confidence, factors, recommendations
      ) VALUES (?, ?, ?, ?, ?, ?)
    `;
    
    const result = await database.run(sql, [
      value.patient_id,
      predictionResult.risk_score,
      predictionResult.risk_level,
      predictionResult.confidence,
      JSON.stringify(predictionResult.factors),
      predictionResult.recommendations
    ]);
    
    // Fetch the created prediction
    const newPrediction = await database.get(`
      SELECT p.*, pt.name as patient_name, pt.gender, pt.age 
      FROM predictions p 
      LEFT JOIN patients pt ON p.patient_id = pt.patient_id 
      WHERE p.id = ?
    `, [result.id]);
    
    // Parse JSON factors
    newPrediction.factors = JSON.parse(newPrediction.factors);
    
    res.status(201).json({
      ...newPrediction,
      model_version: predictionResult.model_version,
      prediction_time: predictionResult.prediction_time
    });
    
  } catch (error) {
    console.error('Error creating prediction:', error);
    res.status(500).json({ error: 'Failed to create prediction' });
  }
});

// DELETE /api/predictions/:id - Delete a prediction
router.delete('/:id', async (req, res) => {
  try {
    
    // Check if prediction exists
    const existing = await database.get(
      'SELECT id FROM predictions WHERE id = ?',
      [req.params.id]
    );
    
    if (!existing) {
      return res.status(404).json({ error: 'Prediction not found' });
    }
    
    // Delete prediction
    await database.run('DELETE FROM predictions WHERE id = ?', [req.params.id]);
    
    res.json({ message: 'Prediction deleted successfully' });
    
  } catch (error) {
    console.error('Error deleting prediction:', error);
    res.status(500).json({ error: 'Failed to delete prediction' });
  }
});

// POST /api/predictions/batch - Create predictions for multiple patients
router.post('/batch', async (req, res) => {
  try {
    const { patient_ids } = req.body;
    
    if (!Array.isArray(patient_ids) || patient_ids.length === 0) {
      return res.status(400).json({ error: 'patient_ids must be a non-empty array' });
    }

    const results = [];
    const errors = [];
    
    for (const patient_id of patient_ids) {
      try {
        // Check if patient exists
        const patient = await database.get(
          'SELECT * FROM patients WHERE patient_id = ?',
          [patient_id]
        );
        
        if (!patient) {
          errors.push({ patient_id, error: 'Patient not found' });
          continue;
        }
        
        // Generate prediction
        const predictionResult = await predictionService.predict(patient);
        
        // Save to database
        const sql = `
          INSERT INTO predictions (
            patient_id, risk_score, risk_level, confidence, factors, recommendations
          ) VALUES (?, ?, ?, ?, ?, ?)
        `;
        
        const result = await database.run(sql, [
          patient_id,
          predictionResult.risk_score,
          predictionResult.risk_level,
          predictionResult.confidence,
          JSON.stringify(predictionResult.factors),
          predictionResult.recommendations
        ]);
        
        results.push({
          patient_id,
          prediction_id: result.id,
          risk_score: predictionResult.risk_score,
          risk_level: predictionResult.risk_level
        });
        
      } catch (error) {
        errors.push({ patient_id, error: error.message });
      }
    }
    
    res.json({
      success: results,
      errors: errors,
      summary: {
        total: patient_ids.length,
        successful: results.length,
        failed: errors.length
      }
    });
    
  } catch (error) {
    console.error('Error in batch prediction:', error);
    res.status(500).json({ error: 'Failed to process batch predictions' });
  }
});

// GET /api/predictions/service/health - Check prediction service health
router.get('/service/health', async (req, res) => {
  try {
    const health = await predictionService.healthCheck();
    res.json(health);
  } catch (error) {
    res.status(500).json({ 
      status: 'error', 
      message: error.message 
    });
  }
});

// POST /api/predictions/service/mode - Switch prediction service mode
router.post('/service/mode', async (req, res) => {
  try {
    const { mode } = req.body;
    
    if (!['mock', 'api'].includes(mode)) {
      return res.status(400).json({ error: 'Invalid mode. Use "mock" or "api"' });
    }
    
    predictionService.setMode(mode);
    
    res.json({ 
      message: `Prediction service switched to ${mode} mode`,
      mode: mode
    });
    
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;
