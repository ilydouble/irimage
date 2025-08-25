const axios = require('axios');

// Configuration for prediction service
const config = {
  // Set to 'mock' for development, 'api' for production
  mode: process.env.PREDICTION_MODE || 'mock',
  
  // Real API configuration
  apiUrl: process.env.PREDICTION_API_URL || 'http://localhost:5000/predict',
  apiKey: process.env.PREDICTION_API_KEY || '',
  timeout: 30000, // 30 seconds
  
  // Mock configuration
  mockDelay: 1000 // 1 second delay to simulate real API
};

class PredictionService {
  constructor() {
    this.mode = config.mode;
    console.log(`Prediction service initialized in ${this.mode} mode`);
  }

  // Switch between mock and real API mode
  setMode(mode) {
    if (!['mock', 'api'].includes(mode)) {
      throw new Error('Invalid mode. Use "mock" or "api"');
    }
    this.mode = mode;
    console.log(`Prediction service switched to ${mode} mode`);
  }

  // Main prediction method
  async predict(patientData) {
    if (this.mode === 'mock') {
      return await this.mockPredict(patientData);
    } else {
      return await this.apiPredict(patientData);
    }
  }

  // Mock prediction implementation
  async mockPredict(patientData) {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, config.mockDelay));

    // Extract relevant features
    const { age, gender, bmi, waist, hip, neck, has_icas, icas_grade } = patientData;

    // Simple rule-based mock prediction
    let riskScore = 0;
    let factors = {};
    let recommendations = [];

    // Age factor (0-30 points)
    if (age < 30) {
      riskScore += 5;
      factors.age = 'low';
    } else if (age < 50) {
      riskScore += 15;
      factors.age = 'medium';
    } else {
      riskScore += 25;
      factors.age = 'high';
      recommendations.push('年龄较大，建议定期体检');
    }

    // BMI factor (0-25 points)
    if (bmi) {
      if (bmi < 18.5) {
        riskScore += 5;
        factors.bmi = 'underweight';
        recommendations.push('体重偏低，建议增加营养');
      } else if (bmi < 24) {
        riskScore += 2;
        factors.bmi = 'normal';
      } else if (bmi < 28) {
        riskScore += 15;
        factors.bmi = 'overweight';
        recommendations.push('体重超标，建议控制饮食');
      } else {
        riskScore += 25;
        factors.bmi = 'obese';
        recommendations.push('肥胖，建议减重和运动');
      }
    }

    // Waist circumference factor (0-20 points)
    if (waist) {
      const waistThreshold = gender === 'male' ? 90 : 80;
      if (waist > waistThreshold) {
        riskScore += 20;
        factors.waist_circumference = 'high';
        recommendations.push('腰围超标，增加心血管疾病风险');
      } else {
        factors.waist_circumference = 'normal';
      }
    }

    // Neck circumference factor (0-15 points)
    if (neck) {
      const neckThreshold = gender === 'male' ? 37 : 34;
      if (neck > neckThreshold) {
        riskScore += 15;
        factors.neck_circumference = 'high';
        recommendations.push('颈围较大，可能存在睡眠呼吸暂停风险');
      } else {
        factors.neck_circumference = 'normal';
      }
    }

    // ICAS factor (0-30 points)
    if (has_icas) {
      switch (icas_grade) {
        case '轻度':
          riskScore += 15;
          factors.icas = 'mild';
          recommendations.push('轻度ICAS，需要定期监测');
          break;
        case '中度':
          riskScore += 25;
          factors.icas = 'moderate';
          recommendations.push('中度ICAS，建议专科治疗');
          break;
        case '重度':
          riskScore += 30;
          factors.icas = 'severe';
          recommendations.push('重度ICAS，需要立即治疗');
          break;
        default:
          factors.icas = 'none';
      }
    }

    // Gender factor
    factors.gender = gender;
    if (gender === 'male') {
      riskScore += 5; // Males generally have higher cardiovascular risk
    }

    // Add some randomness to make it more realistic
    riskScore += Math.random() * 10 - 5; // ±5 points
    riskScore = Math.max(0, Math.min(100, riskScore)); // Clamp to 0-100

    // Determine risk level
    let riskLevel;
    if (riskScore < 30) {
      riskLevel = 'low';
    } else if (riskScore < 70) {
      riskLevel = 'medium';
    } else {
      riskLevel = 'high';
    }

    // Calculate confidence (higher for more extreme scores)
    const confidence = Math.min(95, 60 + Math.abs(riskScore - 50));

    // Add general recommendations
    if (recommendations.length === 0) {
      recommendations.push('保持健康生活方式');
    }
    recommendations.push('定期体检，监测健康状况');

    return {
      risk_score: parseFloat(riskScore.toFixed(2)),
      risk_level: riskLevel,
      confidence: parseFloat(confidence.toFixed(2)),
      factors: factors,
      recommendations: recommendations.join('；'),
      model_version: 'mock-v1.0',
      prediction_time: new Date().toISOString()
    };
  }

  // Real API prediction implementation
  async apiPredict(patientData) {
    try {
      const response = await axios.post(config.apiUrl, {
        patient_data: patientData
      }, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${config.apiKey}`,
          'X-API-Version': '1.0'
        },
        timeout: config.timeout
      });

      // Validate response structure
      const result = response.data;
      if (!result.risk_score || !result.risk_level || !result.confidence) {
        throw new Error('Invalid response format from prediction API');
      }

      return {
        risk_score: result.risk_score,
        risk_level: result.risk_level,
        confidence: result.confidence,
        factors: result.factors || {},
        recommendations: result.recommendations || '请咨询医生',
        model_version: result.model_version || 'unknown',
        prediction_time: new Date().toISOString()
      };

    } catch (error) {
      console.error('Prediction API error:', error.message);
      
      // Fallback to mock prediction if API fails
      console.log('Falling back to mock prediction...');
      return await this.mockPredict(patientData);
    }
  }

  // Health check for the prediction service
  async healthCheck() {
    if (this.mode === 'mock') {
      return {
        status: 'healthy',
        mode: 'mock',
        message: 'Mock prediction service is running'
      };
    }

    try {
      const response = await axios.get(`${config.apiUrl}/health`, {
        timeout: 5000
      });
      
      return {
        status: 'healthy',
        mode: 'api',
        message: 'Real prediction API is accessible',
        api_status: response.data
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        mode: 'api',
        message: 'Real prediction API is not accessible',
        error: error.message
      };
    }
  }
}

// Create singleton instance
const predictionService = new PredictionService();

module.exports = predictionService;
