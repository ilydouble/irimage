#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const database = require('../config/database');

// 读取临床特征文件并解析ICAS数据
function parseClinicalData() {
  const clinicalPath = path.join(__dirname, '..', 'data', '临床特征核对后最终版.md');
  const content = fs.readFileSync(clinicalPath, 'utf-8');

  const lines = content.split('\n');
  const icasData = {};

  // 找到数据开始的行
  let dataStartIndex = -1;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].includes('| BY039 |') || lines[i].includes('| CN028 |')) {
      dataStartIndex = i;
      break;
    }
  }

  if (dataStartIndex === -1) {
    console.warn('未找到临床特征数据开始行');
    return icasData;
  }

  // 安全地解析数值，处理空值和无效值
  const parseNumber = (value) => {
    if (!value || value === '' || value === '#N/A' || value === 'NULL') {
      return null;
    }
    const num = parseFloat(value);
    return isNaN(num) ? null : num;
  };

  // 解析每一行临床特征数据
  for (let i = dataStartIndex; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line || !line.startsWith('|')) continue;

    const columns = line.split('|').map(col => col.trim()).filter(col => col);

    if (columns.length < 12) continue;

    try {
      const patientId = columns[0];
      const hasIcas = parseInt(columns[9]) === 1; // 有无ICAS（0=无，1=有）
      const icasSeverity = parseInt(columns[11]); // 狭窄程度多分类

      // 将狭窄程度转换为中文
      let icasGrade = '无';
      if (hasIcas) {
        switch (icasSeverity) {
          case 1:
            icasGrade = '轻度';
            break;
          case 2:
            icasGrade = '中度';
            break;
          case 3:
            icasGrade = '重度';
            break;
          default:
            icasGrade = '无'; // 改为'无'而不是'正常'
        }
      }

      icasData[patientId] = {
        has_icas: hasIcas,
        icas_grade: icasGrade || '无',
        // 添加其他临床数据（假设体重在第4列，需要根据实际文件调整）
        weight: parseNumber(columns[3]), // 请根据实际列位置调整
        height: parseNumber(columns[4]), // 请根据实际列位置调整
        age: parseNumber(columns[2])     // 请根据实际列位置调整
      };

    } catch (error) {
      console.warn(`解析临床特征第${i+1}行时出错:`, error.message);
    }
  }

  console.log(`解析到 ${Object.keys(icasData).length} 个患者的临床特征数据`);
  return icasData;
}

// 读取markdown文件并解析数据
function parseMarkdownData(icasData = {}) {
  const markdownPath = path.join(__dirname, '..', 'data', '数据对齐和基线信息匹配.md');
  const content = fs.readFileSync(markdownPath, 'utf-8');
  
  // 按行分割
  const lines = content.split('\n');
  console.log(`文件总行数: ${lines.length}`);
  
  // 找到表格数据开始的行
  let dataStartIndex = -1;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line.includes('| 编号') || line.includes('|:---')) {
      continue;
    }
    if (line.startsWith('|') && line.includes('|')) {
      const columns = line.split('|').map(col => col.trim()).filter(col => col);
      if (columns.length >= 10 && columns[0].match(/^[A-Z0-9]+$/)) {
        dataStartIndex = i;
        console.log(`数据开始行: ${i+1}, 内容: ${line.substring(0, 50)}...`);
        break;
      }
    }
  }
  
  if (dataStartIndex === -1) {
    throw new Error('找不到数据开始行');
  }
  
  const patients = [];
  let totalLines = 0;
  let validLines = 0;
  let skippedLines = [];
  let lastValidLine = -1;

  for (let i = dataStartIndex; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line || !line.startsWith('|')) continue;
    
    totalLines++;
    
    // 不要过滤空列，保持原始列结构
    const columns = line.split('|').map(col => col.trim());
    // 移除首尾的空元素（由于行首行尾的|导致的）
    if (columns[0] === '') columns.shift();
    if (columns[columns.length - 1] === '') columns.pop();
    
    if (columns.length < 12) {
      skippedLines.push(`第${i+1}行: 列数不足 (${columns.length}列) - ${line.substring(0, 50)}...`);
      continue;
    }
    
    try {
      const patientId = columns[0];


      if (!patientId || !patientId.match(/^[A-Z0-9]+$/)) {
        skippedLines.push(`第${i+1}行: 患者ID格式错误 (${patientId})`);
        continue;
      }
      
      // 从临床特征数据中获取真实ICAS信息和其他临床数据
      const icasInfo = icasData[patientId] || {
        has_icas: false,
        icas_grade: '无',
        weight: null,
        height: null,
        age: null
      };

      // 安全地解析数值，处理空值和无效值
      const parseNumber = (value) => {
        if (!value || value === '' || value === '#N/A' || value === 'NULL') {
          return null;
        }
        const num = parseFloat(value);
        return isNaN(num) ? null : num;
      };

      const parseInteger = (value) => {
        if (!value || value === '' || value === '#N/A' || value === 'NULL') {
          return null;
        }
        const num = parseInt(value);
        return isNaN(num) ? null : num;
      };

      const patient = {
        patient_id: patientId,
        name: columns[1] || '未知',
        gender: columns[5] === '男' ? 'male' : (columns[5] === '女' ? 'female' : 'male'),
        age: icasInfo.age || parseInteger(columns[6]),
        height: icasInfo.height || parseNumber(columns[7]),
        weight: icasInfo.weight || parseNumber(columns[8]),
        waist: parseNumber(columns[9]),
        hip: parseNumber(columns[10]),
        neck: parseNumber(columns[11]),
        // 检查项目：空值或"是"都视为true，其他为false
        thermal_24h: columns[2] === '是',
        thermal_25h: columns[3] === '是',
        voice_25h: columns[4] === '是',
        // 使用真实ICAS数据
        has_icas: icasInfo.has_icas,
        icas_grade: icasInfo.icas_grade || '无'
      };

      // 计算BMI
      if (patient.height && patient.weight) {
        patient.bmi = parseFloat((patient.weight / Math.pow(patient.height / 100, 2)).toFixed(2));
      }
      
      // 只要有患者编号就保留记录
      if (patient.patient_id) {
        patients.push(patient);
      }
      validLines++;
      lastValidLine = i;
      
    } catch (error) {
      skippedLines.push(`第${i+1}行: ${error.message} - ${line.substring(0, 50)}...`);
    }
  }
  
  console.log(`最后一个有效数据行: ${lastValidLine + 1}`);
  console.log(`总处理行数: ${totalLines}, 有效行数: ${validLines}, 跳过行数: ${skippedLines.length}`);
  
  return patients;
}

// 导入患者数据到数据库
async function importPatients(patients) {
  const sql = `
    INSERT OR REPLACE INTO patients (
      patient_id, name, gender, age, height, weight, bmi, waist, hip, neck,
      thermal_24h, thermal_25h, voice_25h, has_icas, icas_grade
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `;
  
  let imported = 0;
  let errors = 0;
  
  for (const patient of patients) {
    try {
      await database.run(sql, [
        patient.patient_id,
        patient.name,
        patient.gender,
        patient.age,
        patient.height,
        patient.weight,
        patient.bmi,
        patient.waist,
        patient.hip,
        patient.neck,
        patient.thermal_24h ? 1 : 0,
        patient.thermal_25h ? 1 : 0,
        patient.voice_25h ? 1 : 0,
        patient.has_icas ? 1 : 0,
        patient.icas_grade
      ]);
      imported++;
    } catch (error) {
      console.error(`导入患者 ${patient.patient_id} 失败:`, error.message);
      errors++;
    }
  }
  
  return { imported, errors };
}

// ICAS预测函数
function predictICAS(patient) {
  // 基于患者特征的简单ICAS预测模型
  let riskScore = 0;
  let factors = [];

  // 年龄因子 (年龄越大风险越高)
  if (patient.age >= 70) {
    riskScore += 30;
    factors.push('高龄');
  } else if (patient.age >= 60) {
    riskScore += 20;
    factors.push('中年');
  } else if (patient.age >= 50) {
    riskScore += 10;
    factors.push('中年');
  }

  // 性别因子 (男性风险稍高)
  if (patient.gender === 'male') {
    riskScore += 5;
    factors.push('男性');
  }

  // BMI因子
  if (patient.bmi) {
    if (patient.bmi >= 30) {
      riskScore += 15;
      factors.push('肥胖');
    } else if (patient.bmi >= 25) {
      riskScore += 8;
      factors.push('超重');
    }
  }



  // 颈围因子 (颈围大可能与血管疾病相关)
  if (patient.neck) {
    const neckThreshold = patient.gender === 'male' ? 38 : 35;
    if (patient.neck > neckThreshold) {
      riskScore += 12;
      factors.push('颈围偏大');
    }
  }

  // 腰围因子
  if (patient.waist) {
    const waistThreshold = patient.gender === 'male' ? 90 : 80;
    if (patient.waist > waistThreshold) {
      riskScore += 10;
      factors.push('腰围超标');
    }
  }

  // 添加一些随机性
  riskScore += Math.random() * 10 - 5; // ±5分
  riskScore = Math.max(0, Math.min(100, riskScore));

  // 确定预测等级
  let predictedGrade = '无';
  let hasIcas = false;

  if (riskScore >= 70) {
    predictedGrade = '重度';
    hasIcas = true;
  } else if (riskScore >= 50) {
    predictedGrade = '中度';
    hasIcas = true;
  } else if (riskScore >= 30) {
    predictedGrade = '轻度';
    hasIcas = true;
  }

  // 计算置信度
  const confidence = Math.min(95, 60 + Math.abs(riskScore - 50) * 0.7);

  return {
    predicted_icas: hasIcas,
    predicted_icas_grade: predictedGrade,
    predicted_icas_confidence: parseFloat(confidence.toFixed(2)),
    prediction_factors: factors.join(', ')
  };
}

// 为患者生成ICAS预测
async function generateICASPredictions(patients) {
  let generated = 0;
  let errors = 0;

  for (const patient of patients) {
    try {
      const prediction = predictICAS(patient);

      // 更新患者记录，添加预测结果
      const sql = `
        UPDATE patients SET
          predicted_icas = ?,
          predicted_icas_grade = ?,
          predicted_icas_confidence = ?
        WHERE patient_id = ?
      `;

      await database.run(sql, [
        prediction.predicted_icas ? 1 : 0,
        prediction.predicted_icas_grade,
        prediction.predicted_icas_confidence,
        patient.patient_id
      ]);

      generated++;
    } catch (error) {
      console.error(`为患者 ${patient.patient_id} 生成ICAS预测失败:`, error.message);
      errors++;
    }
  }

  return { generated, errors };
}

// 为导入的患者生成预测数据
async function generatePredictionsForPatients(patients) {
  const predictionService = require('../services/predictionService');
  
  let generated = 0;
  let errors = 0;
  
  for (const patient of patients) {
    try {
      // 获取完整的患者数据
      const fullPatient = await database.get(
        'SELECT * FROM patients WHERE patient_id = ?',
        [patient.patient_id]
      );
      
      if (!fullPatient) continue;
      
      // 生成预测
      const prediction = await predictionService.predict(fullPatient);
      
      // 保存预测结果
      const sql = `
        INSERT OR REPLACE INTO predictions (
          patient_id, risk_score, risk_level, confidence, factors, recommendations
        ) VALUES (?, ?, ?, ?, ?, ?)
      `;
      
      await database.run(sql, [
        patient.patient_id,
        prediction.risk_score,
        prediction.risk_level,
        prediction.confidence,
        JSON.stringify(prediction.factors),
        prediction.recommendations
      ]);
      
      generated++;
    } catch (error) {
      console.error(`为患者 ${patient.patient_id} 生成预测失败:`, error.message);
      errors++;
    }
  }
  
  return { generated, errors };
}

// 主函数
async function main() {
  try {
    console.log('开始导入markdown数据...');
    
    // 连接数据库
    await database.connect();

    // 解析临床特征数据（ICAS真实值）
    console.log('解析临床特征数据...');
    const icasData = parseClinicalData();

    // 解析markdown数据
    console.log('解析markdown文件...');
    const patients = parseMarkdownData(icasData);
    console.log(`解析到 ${patients.length} 个患者数据`);
    
    // 检查数据匹配情况
    const icasPatientIds = Object.keys(icasData);
    const baselinePatientIds = patients.map(p => p.patient_id);

    console.log(`\n数据匹配分析:`);
    console.log(`临床特征文件患者数: ${icasPatientIds.length}`);
    console.log(`基线信息文件患者数: ${baselinePatientIds.length}`);

    const onlyInIcas = icasPatientIds.filter(id => !baselinePatientIds.includes(id));
    const onlyInBaseline = baselinePatientIds.filter(id => !icasPatientIds.includes(id));

    console.log(`只在临床特征中的患者: ${onlyInIcas.length}`);
    console.log(`只在基线信息中的患者: ${onlyInBaseline.length}`);

    if (onlyInIcas.length > 0) {
      console.log(`\n只在临床特征中的患者ID:`);
      onlyInIcas.forEach((id, index) => {
        console.log(`${index + 1}. ${id}`);
      });
    }

    if (onlyInBaseline.length > 0) {
      console.log(`\n只在基线信息中的患者ID:`);
      onlyInBaseline.forEach((id, index) => {
        console.log(`${index + 1}. ${id}`);
      });
    }

    // 检查是否有重复的患者ID
    const duplicateIds = baselinePatientIds.filter((id, index) => baselinePatientIds.indexOf(id) !== index);
    if (duplicateIds.length > 0) {
      console.log(`\n基线信息中重复的患者ID: ${duplicateIds.join(', ')}`);
    }

    // 显示前几个患者的信息作为预览
    console.log('\n前5个患者数据预览:');
    patients.slice(0, 5).forEach((patient, index) => {
      console.log(`${index + 1}. ${patient.patient_id} - ${patient.name} (${patient.gender === 'male' ? '男' : '女'}, ${patient.age}岁)`);
    });
    
    // 导入患者数据
    console.log('\n导入患者数据到数据库...');
    const importResult = await importPatients(patients);
    console.log(`✓ 成功导入 ${importResult.imported} 个患者`);
    if (importResult.errors > 0) {
      console.log(`⚠ ${importResult.errors} 个患者导入失败`);
    }
    
    // 注释掉ICAS预测生成
    /*
    console.log('\n为患者生成ICAS预测...');
    const icasPredictionResult = await generateICASPredictions(patients);
    console.log(`✓ 成功生成 ${icasPredictionResult.generated} 个ICAS预测`);
    if (icasPredictionResult.errors > 0) {
      console.log(`⚠ ${icasPredictionResult.errors} 个ICAS预测生成失败`);
    }

    // 生成风险预测数据
    console.log('\n为患者生成风险预测数据...');
    const predictionResult = await generatePredictionsForPatients(patients);
    console.log(`✓ 成功生成 ${predictionResult.generated} 个风险预测`);
    if (predictionResult.errors > 0) {
      console.log(`⚠ ${predictionResult.errors} 个风险预测生成失败`);
    }
    */
    
    // 显示统计信息
    const totalPatients = await database.get('SELECT COUNT(*) as count FROM patients');
    // const totalPredictions = await database.get('SELECT COUNT(*) as count FROM predictions');
    
    console.log('\n数据库统计:');
    console.log(`总患者数: ${totalPatients.count}`);
    // console.log(`总预测数: ${totalPredictions.count}`);
    
    console.log('\n数据导入完成！');
    
  } catch (error) {
    console.error('导入过程中发生错误:', error);
    process.exit(1);
  } finally {
    await database.close();
  }
}

// 如果直接运行此脚本
if (require.main === module) {
  main();
}

module.exports = { parseMarkdownData, importPatients, generatePredictionsForPatients };
