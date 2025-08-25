const express = require('express');
const Joi = require('joi');
const database = require('../config/database');

const router = express.Router();

// Validation schema for patient data
const patientSchema = Joi.object({
  patient_id: Joi.string().max(20).required(),
  name: Joi.string().max(50).required(),
  gender: Joi.string().valid('male', 'female').required(),
  age: Joi.number().integer().min(0).max(150).required(),
  height: Joi.number().precision(2).min(0).max(300).optional(),
  weight: Joi.number().precision(2).min(0).max(500).optional(),
  bmi: Joi.number().precision(2).min(0).max(100).optional(),
  waist: Joi.number().precision(2).min(0).max(200).optional(),
  hip: Joi.number().precision(2).min(0).max(200).optional(),
  neck: Joi.number().precision(2).min(0).max(100).optional(),
  thermal_24h: Joi.boolean().optional(),
  thermal_25h: Joi.boolean().optional(),
  voice_25h: Joi.boolean().optional(),
  has_icas: Joi.boolean().optional(),
  icas_grade: Joi.string().valid('无', '轻度', '中度', '重度').optional()
});

// GET /api/patients - Get all patients with optional filtering
router.get('/', async (req, res) => {
  try {
    await database.connect();
    
    const {
      search,
      gender,
      has_icas,
      icas_grade,
      age_min,
      age_max,
      bmi_min,
      bmi_max,
      page = 1,
      limit = 10
    } = req.query;
    
    let sql = 'SELECT * FROM patients WHERE 1=1';
    const params = [];
    
    // Add search filters
    if (search) {
      sql += ' AND (name LIKE ? OR patient_id LIKE ?)';
      params.push(`%${search}%`, `%${search}%`);
    }
    
    if (gender) {
      sql += ' AND gender = ?';
      params.push(gender);
    }
    
    if (has_icas !== undefined) {
      sql += ' AND has_icas = ?';
      params.push(has_icas === 'true' ? 1 : 0);
    }
    
    if (icas_grade) {
      sql += ' AND icas_grade = ?';
      params.push(icas_grade);
    }
    
    if (age_min) {
      sql += ' AND age >= ?';
      params.push(parseInt(age_min));
    }
    
    if (age_max) {
      sql += ' AND age <= ?';
      params.push(parseInt(age_max));
    }

    if (bmi_min) {
      sql += ' AND bmi >= ?';
      params.push(parseFloat(bmi_min));
    }

    if (bmi_max) {
      sql += ' AND bmi <= ?';
      params.push(parseFloat(bmi_max));
    }

    // Add thermal and voice filters
    if (req.query.thermal_24h !== undefined) {
      sql += ' AND thermal_24h = ?';
      params.push(req.query.thermal_24h === 'true' ? 1 : 0);
    }

    if (req.query.thermal_25h !== undefined) {
      sql += ' AND thermal_25h = ?';
      params.push(req.query.thermal_25h === 'true' ? 1 : 0);
    }

    if (req.query.voice_25h !== undefined) {
      sql += ' AND voice_25h = ?';
      params.push(req.query.voice_25h === 'true' ? 1 : 0);
    }

    // Add sorting
    const sortBy = req.query.sortBy || 'created_at';
    const sortOrder = req.query.sortOrder || 'desc';

    // 验证排序字段
    const allowedSortFields = [
      'patient_id', 'name', 'gender', 'age', 'height', 'weight', 'bmi',
      'waist', 'hip', 'neck', 'has_icas', 'icas_grade', 'created_at'
    ];

    const validSortBy = allowedSortFields.includes(sortBy) ? sortBy : 'created_at';
    const validSortOrder = ['asc', 'desc'].includes(sortOrder.toLowerCase()) ? sortOrder.toUpperCase() : 'DESC';

    // Add pagination
    const offset = (page - 1) * limit;
    sql += ` ORDER BY ${validSortBy} ${validSortOrder} LIMIT ? OFFSET ?`;
    params.push(parseInt(limit), offset);
    
    const patients = await database.all(sql, params);

    // Get predictions for each patient
    for (let patient of patients) {
      const predictions = await database.all(
        'SELECT * FROM predictions WHERE patient_id = ? ORDER BY created_at DESC',
        [patient.patient_id]
      );
      patient.predictions = predictions;
    }

    // Get total count for pagination
    let countSql = 'SELECT COUNT(*) as total FROM patients WHERE 1=1';
    const countParams = [];

    // Add search filters
    if (search) {
      countSql += ' AND (name LIKE ? OR patient_id LIKE ?)';
      countParams.push(`%${search}%`, `%${search}%`);
    }

    if (gender) {
      countSql += ' AND gender = ?';
      countParams.push(gender);
    }

    if (has_icas !== undefined) {
      countSql += ' AND has_icas = ?';
      countParams.push(has_icas === 'true' ? 1 : 0);
    }

    if (icas_grade) {
      countSql += ' AND icas_grade = ?';
      countParams.push(icas_grade);
    }

    if (age_min) {
      countSql += ' AND age >= ?';
      countParams.push(parseInt(age_min));
    }

    if (age_max) {
      countSql += ' AND age <= ?';
      countParams.push(parseInt(age_max));
    }

    if (bmi_min) {
      countSql += ' AND bmi >= ?';
      countParams.push(parseFloat(bmi_min));
    }

    if (bmi_max) {
      countSql += ' AND bmi <= ?';
      countParams.push(parseFloat(bmi_max));
    }

    // Add thermal and voice filters to count query
    if (req.query.thermal_24h !== undefined) {
      countSql += ' AND thermal_24h = ?';
      countParams.push(req.query.thermal_24h === 'true' ? 1 : 0);
    }

    if (req.query.thermal_25h !== undefined) {
      countSql += ' AND thermal_25h = ?';
      countParams.push(req.query.thermal_25h === 'true' ? 1 : 0);
    }

    if (req.query.voice_25h !== undefined) {
      countSql += ' AND voice_25h = ?';
      countParams.push(req.query.voice_25h === 'true' ? 1 : 0);
    }

    const countResult = await database.get(countSql, countParams);
    
    res.json({
      patients,
      pagination: {
        page: parseInt(page),
        limit: parseInt(limit),
        total: countResult.total,
        pages: Math.ceil(countResult.total / limit)
      }
    });
    
  } catch (error) {
    console.error('Error fetching patients:', error);
    res.status(500).json({ error: 'Failed to fetch patients' });
  }
});

// GET /api/patients/:id - Get a specific patient
router.get('/:id', async (req, res) => {
  try {
    await database.connect();
    
    const patient = await database.get(
      'SELECT * FROM patients WHERE patient_id = ?',
      [req.params.id]
    );
    
    if (!patient) {
      return res.status(404).json({ error: 'Patient not found' });
    }
    
    // Get associated predictions
    const predictions = await database.all(
      'SELECT * FROM predictions WHERE patient_id = ? ORDER BY created_at DESC',
      [req.params.id]
    );
    
    // Get associated files
    const files = await database.all(
      'SELECT * FROM files WHERE patient_id = ? ORDER BY upload_time DESC',
      [req.params.id]
    );
    
    res.json({
      ...patient,
      predictions,
      files
    });
    
  } catch (error) {
    console.error('Error fetching patient:', error);
    res.status(500).json({ error: 'Failed to fetch patient' });
  }
});

// POST /api/patients - Create a new patient
router.post('/', async (req, res) => {
  try {
    // Validate input
    const { error, value } = patientSchema.validate(req.body);
    if (error) {
      return res.status(400).json({ 
        error: 'Validation failed', 
        details: error.details 
      });
    }
    
    await database.connect();
    
    // Check if patient_id already exists
    const existing = await database.get(
      'SELECT id FROM patients WHERE patient_id = ?',
      [value.patient_id]
    );
    
    if (existing) {
      return res.status(409).json({ error: 'Patient ID already exists' });
    }
    
    // Calculate BMI if height and weight are provided
    if (value.height && value.weight && !value.bmi) {
      value.bmi = (value.weight / Math.pow(value.height / 100, 2)).toFixed(2);
    }
    
    // Insert patient
    const sql = `
      INSERT INTO patients (
        patient_id, name, gender, age, height, weight, bmi, waist, hip, neck,
        thermal_24h, thermal_25h, voice_25h, has_icas, icas_grade
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `;
    
    const result = await database.run(sql, [
      value.patient_id, value.name, value.gender, value.age,
      value.height, value.weight, value.bmi, value.waist,
      value.hip, value.neck, value.thermal_24h ? 1 : 0,
      value.thermal_25h ? 1 : 0, value.voice_25h ? 1 : 0,
      value.has_icas ? 1 : 0, value.icas_grade || '无'
    ]);
    
    // Fetch the created patient
    const newPatient = await database.get(
      'SELECT * FROM patients WHERE id = ?',
      [result.id]
    );
    
    res.status(201).json(newPatient);
    
  } catch (error) {
    console.error('Error creating patient:', error);
    res.status(500).json({ error: 'Failed to create patient' });
  }
});

// PUT /api/patients/:id - Update a patient
router.put('/:id', async (req, res) => {
  try {
    // Validate input
    const { error, value } = patientSchema.validate(req.body);
    if (error) {
      return res.status(400).json({
        error: 'Validation failed',
        details: error.details
      });
    }

    await database.connect();

    // Check if patient exists
    const existing = await database.get(
      'SELECT id FROM patients WHERE patient_id = ?',
      [req.params.id]
    );

    if (!existing) {
      return res.status(404).json({ error: 'Patient not found' });
    }

    // Calculate BMI if height and weight are provided
    if (value.height && value.weight && !value.bmi) {
      value.bmi = (value.weight / Math.pow(value.height / 100, 2)).toFixed(2);
    }

    // Update patient
    const sql = `
      UPDATE patients SET
        name = ?, gender = ?, age = ?, height = ?, weight = ?, bmi = ?,
        waist = ?, hip = ?, neck = ?, thermal_24h = ?, thermal_25h = ?,
        voice_25h = ?, has_icas = ?, icas_grade = ?
      WHERE patient_id = ?
    `;

    await database.run(sql, [
      value.name, value.gender, value.age, value.height, value.weight,
      value.bmi, value.waist, value.hip, value.neck,
      value.thermal_24h ? 1 : 0, value.thermal_25h ? 1 : 0,
      value.voice_25h ? 1 : 0, value.has_icas ? 1 : 0,
      value.icas_grade || '无', req.params.id
    ]);

    // Fetch the updated patient
    const updatedPatient = await database.get(
      'SELECT * FROM patients WHERE patient_id = ?',
      [req.params.id]
    );

    res.json(updatedPatient);

  } catch (error) {
    console.error('Error updating patient:', error);
    res.status(500).json({ error: 'Failed to update patient' });
  }
});

// DELETE /api/patients/:id - Delete a patient
router.delete('/:id', async (req, res) => {
  try {
    await database.connect();

    // Check if patient exists
    const existing = await database.get(
      'SELECT id FROM patients WHERE patient_id = ?',
      [req.params.id]
    );

    if (!existing) {
      return res.status(404).json({ error: 'Patient not found' });
    }

    // Delete associated records first (due to foreign key constraints)
    await database.run('DELETE FROM predictions WHERE patient_id = ?', [req.params.id]);
    await database.run('DELETE FROM files WHERE patient_id = ?', [req.params.id]);

    // Delete patient
    await database.run('DELETE FROM patients WHERE patient_id = ?', [req.params.id]);

    res.json({ message: 'Patient deleted successfully' });

  } catch (error) {
    console.error('Error deleting patient:', error);
    res.status(500).json({ error: 'Failed to delete patient' });
  }
});

module.exports = router;
