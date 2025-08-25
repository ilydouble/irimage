const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs-extra');
const { v4: uuidv4 } = require('uuid');
const database = require('../config/database');

const router = express.Router();

// File type configurations
const fileTypeConfig = {
  thermal_24h: {
    allowedExtensions: ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
    maxSize: 10 * 1024 * 1024, // 10MB
    directory: 'thermal_24h'
  },
  thermal_25h: {
    allowedExtensions: ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
    maxSize: 10 * 1024 * 1024, // 10MB
    directory: 'thermal_25h'
  },
  voice_25h: {
    allowedExtensions: ['.wav', '.mp3', '.m4a', '.aac', '.flac'],
    maxSize: 50 * 1024 * 1024, // 50MB
    directory: 'voice_25h'
  }
};

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const fileType = req.body.file_type || req.query.file_type;
    const config = fileTypeConfig[fileType];
    
    if (!config) {
      return cb(new Error('Invalid file type'));
    }
    
    const uploadDir = path.join(__dirname, '..', 'uploads', config.directory);
    fs.ensureDirSync(uploadDir);
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const patientId = req.body.patient_id || req.query.patient_id;
    const fileType = req.body.file_type || req.query.file_type;
    const ext = path.extname(file.originalname);
    const timestamp = Date.now();
    const uniqueId = uuidv4().slice(0, 8);
    
    const filename = `${patientId}_${fileType}_${timestamp}_${uniqueId}${ext}`;
    cb(null, filename);
  }
});

// File filter function
const fileFilter = (req, file, cb) => {
  const fileType = req.body.file_type || req.query.file_type;
  const config = fileTypeConfig[fileType];
  
  if (!config) {
    return cb(new Error('Invalid file type'), false);
  }
  
  const ext = path.extname(file.originalname).toLowerCase();
  if (!config.allowedExtensions.includes(ext)) {
    return cb(new Error(`Invalid file extension. Allowed: ${config.allowedExtensions.join(', ')}`), false);
  }
  
  cb(null, true);
};

// Configure multer
const upload = multer({
  storage: storage,
  fileFilter: fileFilter,
  limits: {
    fileSize: 50 * 1024 * 1024 // 50MB max (will be checked per file type)
  }
});

// GET /api/files - Get all files with optional filtering
router.get('/', async (req, res) => {
  try {
    await database.connect();
    
    const { 
      patient_id, 
      file_type, 
      page = 1, 
      limit = 10 
    } = req.query;
    
    let sql = `
      SELECT f.*, p.name as patient_name 
      FROM files f 
      LEFT JOIN patients p ON f.patient_id = p.patient_id 
      WHERE 1=1
    `;
    const params = [];
    
    // Add filters
    if (patient_id) {
      sql += ' AND f.patient_id = ?';
      params.push(patient_id);
    }
    
    if (file_type) {
      sql += ' AND f.file_type = ?';
      params.push(file_type);
    }
    
    // Add pagination
    const offset = (page - 1) * limit;
    sql += ' ORDER BY f.upload_time DESC LIMIT ? OFFSET ?';
    params.push(parseInt(limit), offset);
    
    const files = await database.all(sql, params);
    
    // Get total count
    let countSql = 'SELECT COUNT(*) as total FROM files WHERE 1=1';
    const countParams = [];
    
    if (patient_id) {
      countSql += ' AND patient_id = ?';
      countParams.push(patient_id);
    }
    
    if (file_type) {
      countSql += ' AND file_type = ?';
      countParams.push(file_type);
    }
    
    const countResult = await database.get(countSql, countParams);
    
    res.json({
      files,
      pagination: {
        page: parseInt(page),
        limit: parseInt(limit),
        total: countResult.total,
        pages: Math.ceil(countResult.total / limit)
      }
    });
    
  } catch (error) {
    console.error('Error fetching files:', error);
    res.status(500).json({ error: 'Failed to fetch files' });
  }
});

// GET /api/files/:id - Get a specific file record
router.get('/:id', async (req, res) => {
  try {
    await database.connect();
    
    const file = await database.get(`
      SELECT f.*, p.name as patient_name 
      FROM files f 
      LEFT JOIN patients p ON f.patient_id = p.patient_id 
      WHERE f.id = ?
    `, [req.params.id]);
    
    if (!file) {
      return res.status(404).json({ error: 'File not found' });
    }
    
    res.json(file);
    
  } catch (error) {
    console.error('Error fetching file:', error);
    res.status(500).json({ error: 'Failed to fetch file' });
  }
});

// POST /api/files/upload - Upload a file
router.post('/upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }
    
    const { patient_id, file_type } = req.body;
    
    if (!patient_id || !file_type) {
      // Clean up uploaded file
      fs.removeSync(req.file.path);
      return res.status(400).json({ 
        error: 'patient_id and file_type are required' 
      });
    }
    
    // Validate file type
    if (!fileTypeConfig[file_type]) {
      fs.removeSync(req.file.path);
      return res.status(400).json({ error: 'Invalid file_type' });
    }
    
    // Check file size against type-specific limits
    const config = fileTypeConfig[file_type];
    if (req.file.size > config.maxSize) {
      fs.removeSync(req.file.path);
      return res.status(400).json({ 
        error: `File too large. Maximum size for ${file_type}: ${config.maxSize / 1024 / 1024}MB` 
      });
    }
    
    await database.connect();
    
    // Check if patient exists
    const patient = await database.get(
      'SELECT id FROM patients WHERE patient_id = ?',
      [patient_id]
    );
    
    if (!patient) {
      fs.removeSync(req.file.path);
      return res.status(404).json({ error: 'Patient not found' });
    }
    
    // Save file record to database
    const relativePath = `/uploads/${config.directory}/${req.file.filename}`;
    
    const sql = `
      INSERT INTO files (patient_id, file_type, file_name, file_path)
      VALUES (?, ?, ?, ?)
    `;
    
    const result = await database.run(sql, [
      patient_id,
      file_type,
      req.file.originalname,
      relativePath
    ]);
    
    // Update patient's file flags
    const updatePatientSql = `UPDATE patients SET ${file_type} = 1 WHERE patient_id = ?`;
    await database.run(updatePatientSql, [patient_id]);
    
    // Fetch the created file record
    const newFile = await database.get(`
      SELECT f.*, p.name as patient_name 
      FROM files f 
      LEFT JOIN patients p ON f.patient_id = p.patient_id 
      WHERE f.id = ?
    `, [result.id]);
    
    res.status(201).json({
      ...newFile,
      file_size: req.file.size,
      upload_success: true
    });
    
  } catch (error) {
    // Clean up uploaded file on error
    if (req.file) {
      fs.removeSync(req.file.path);
    }
    
    console.error('Error uploading file:', error);
    res.status(500).json({ error: 'Failed to upload file' });
  }
});

// DELETE /api/files/:id - Delete a file
router.delete('/:id', async (req, res) => {
  try {
    await database.connect();
    
    // Get file record
    const file = await database.get(
      'SELECT * FROM files WHERE id = ?',
      [req.params.id]
    );
    
    if (!file) {
      return res.status(404).json({ error: 'File not found' });
    }
    
    // Delete physical file
    const fullPath = path.join(__dirname, '..', file.file_path);
    if (fs.existsSync(fullPath)) {
      fs.removeSync(fullPath);
    }
    
    // Delete database record
    await database.run('DELETE FROM files WHERE id = ?', [req.params.id]);
    
    // Update patient's file flag if no more files of this type
    const remainingFiles = await database.get(
      'SELECT COUNT(*) as count FROM files WHERE patient_id = ? AND file_type = ?',
      [file.patient_id, file.file_type]
    );
    
    if (remainingFiles.count === 0) {
      const updatePatientSql = `UPDATE patients SET ${file.file_type} = 0 WHERE patient_id = ?`;
      await database.run(updatePatientSql, [file.patient_id]);
    }
    
    res.json({ message: 'File deleted successfully' });
    
  } catch (error) {
    console.error('Error deleting file:', error);
    res.status(500).json({ error: 'Failed to delete file' });
  }
});

// GET /api/files/download/:id - Download a file
router.get('/download/:id', async (req, res) => {
  try {
    await database.connect();
    
    const file = await database.get(
      'SELECT * FROM files WHERE id = ?',
      [req.params.id]
    );
    
    if (!file) {
      return res.status(404).json({ error: 'File not found' });
    }
    
    const fullPath = path.join(__dirname, '..', file.file_path);
    
    if (!fs.existsSync(fullPath)) {
      return res.status(404).json({ error: 'Physical file not found' });
    }
    
    res.download(fullPath, file.file_name);
    
  } catch (error) {
    console.error('Error downloading file:', error);
    res.status(500).json({ error: 'Failed to download file' });
  }
});

module.exports = router;
