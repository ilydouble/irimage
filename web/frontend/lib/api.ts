// API configuration and client
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001/api';

// Types
export interface Patient {
  id: number;
  patient_id: string;
  name: string;
  gender: 'male' | 'female';
  age: number;
  height?: number;
  weight?: number;
  bmi?: number;
  waist?: number;
  hip?: number;
  neck?: number;
  thermal_24h: boolean;
  thermal_25h: boolean;
  voice_25h: boolean;
  has_icas: boolean;
  icas_grade: '无' | '轻度' | '中度' | '重度';
  created_at: string;
  updated_at: string;
  predictions?: Prediction[];
  files?: FileRecord[];
}

export interface Prediction {
  id: number;
  patient_id: string;
  predicted_icas: boolean;
  predicted_icas_grade: '无' | '轻度' | '中度' | '重度';
  predicted_icas_confidence?: number;
  factors?: string;
  recommendations?: string;
  created_at: string;
  patient_name?: string;
  gender?: string;
  age?: number;
}

export interface FileRecord {
  id: number;
  patient_id: string;
  file_type: 'thermal_24h' | 'thermal_25h' | 'voice_25h';
  file_name: string;
  file_path: string;
  upload_time: string;
  patient_name?: string;
}

export interface PaginationResponse<T> {
  data: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    pages: number;
  };
}

// API client class
class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API request failed: ${url}`, error);
      throw error;
    }
  }

  // Patient API methods
  async getPatients(params?: {
    search?: string;
    gender?: string;
    has_icas?: boolean;
    icas_grade?: string;
    age_min?: number;
    age_max?: number;
    thermal_24h?: boolean;
    thermal_25h?: boolean;
    voice_25h?: boolean;
    page?: number;
    limit?: number;
    sortBy?: string;
    sortOrder?: 'asc' | 'desc';
  }): Promise<{ patients: Patient[]; pagination: any }> {
    const searchParams = new URLSearchParams();
    
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          searchParams.append(key, value.toString());
        }
      });
    }

    const query = searchParams.toString();
    return this.request(`/patients${query ? `?${query}` : ''}`);
  }

  async getPatient(patientId: string): Promise<Patient> {
    return this.request(`/patients/${patientId}`);
  }

  async createPatient(patient: Omit<Patient, 'id' | 'created_at' | 'updated_at'>): Promise<Patient> {
    return this.request('/patients', {
      method: 'POST',
      body: JSON.stringify(patient),
    });
  }

  async updatePatient(patientId: string, patient: Partial<Patient>): Promise<Patient> {
    return this.request(`/patients/${patientId}`, {
      method: 'PUT',
      body: JSON.stringify(patient),
    });
  }

  async deletePatient(patientId: string): Promise<{ message: string }> {
    return this.request(`/patients/${patientId}`, {
      method: 'DELETE',
    });
  }

  // Prediction API methods
  async getPredictions(params?: {
    patient_id?: string;
    risk_level?: string;
    page?: number;
    limit?: number;
  }): Promise<{ predictions: Prediction[]; pagination: any }> {
    const searchParams = new URLSearchParams();
    
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          searchParams.append(key, value.toString());
        }
      });
    }

    const query = searchParams.toString();
    return this.request(`/predictions${query ? `?${query}` : ''}`);
  }

  async getPrediction(predictionId: number): Promise<Prediction> {
    return this.request(`/predictions/${predictionId}`);
  }

  async createPrediction(patientId: string): Promise<Prediction> {
    return this.request('/predictions', {
      method: 'POST',
      body: JSON.stringify({ patient_id: patientId }),
    });
  }

  async deletePrediction(predictionId: number): Promise<{ message: string }> {
    return this.request(`/predictions/${predictionId}`, {
      method: 'DELETE',
    });
  }

  async batchCreatePredictions(patientIds: string[]): Promise<{
    success: any[];
    errors: any[];
    summary: { total: number; successful: number; failed: number };
  }> {
    return this.request('/predictions/batch', {
      method: 'POST',
      body: JSON.stringify({ patient_ids: patientIds }),
    });
  }

  async getPredictionServiceHealth(): Promise<{
    status: string;
    mode: string;
    message: string;
  }> {
    return this.request('/predictions/service/health');
  }

  async setPredictionServiceMode(mode: 'mock' | 'api'): Promise<{
    message: string;
    mode: string;
  }> {
    return this.request('/predictions/service/mode', {
      method: 'POST',
      body: JSON.stringify({ mode }),
    });
  }

  // File API methods
  async getFiles(params?: {
    patient_id?: string;
    file_type?: string;
    page?: number;
    limit?: number;
  }): Promise<{ files: FileRecord[]; pagination: any }> {
    const searchParams = new URLSearchParams();
    
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          searchParams.append(key, value.toString());
        }
      });
    }

    const query = searchParams.toString();
    return this.request(`/files${query ? `?${query}` : ''}`);
  }

  async getFile(fileId: number): Promise<FileRecord> {
    return this.request(`/files/${fileId}`);
  }

  async uploadFile(
    file: File,
    patientId: string,
    fileType: 'thermal_24h' | 'thermal_25h' | 'voice_25h'
  ): Promise<FileRecord> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('patient_id', patientId);
    formData.append('file_type', fileType);

    const response = await fetch(`${this.baseUrl}/files/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || `Upload failed: ${response.statusText}`);
    }

    return await response.json();
  }

  async deleteFile(fileId: number): Promise<{ message: string }> {
    return this.request(`/files/${fileId}`, {
      method: 'DELETE',
    });
  }

  getFileDownloadUrl(fileId: number): string {
    return `${this.baseUrl}/files/download/${fileId}`;
  }

  // Analytics API methods
  async getAnalyticsOverview(): Promise<{
    overview: {
      total_patients: number;
      total_predictions: number;
      total_files: number;
      recent_predictions: number;
      recent_uploads: number;
    };
    distributions: {
      risk_levels: Array<{ risk_level: string; count: number }>;
      genders: Array<{ gender: string; count: number }>;
      icas_grades: Array<{ icas_grade: string; count: number }>;
      age_groups: Array<{ age_group: string; count: number }>;
    };
  }> {
    return this.request('/analytics/overview');
  }

  async getAnalyticsPredictions(period: string = '30'): Promise<{
    predictions_over_time: Array<{ date: string; count: number; avg_risk_score: number }>;
    risk_score_distribution: Array<{ score_range: string; count: number }>;
    confidence_stats: {
      avg_confidence: number;
      min_confidence: number;
      max_confidence: number;
      total_predictions: number;
    };
    risk_trends: Array<{ date: string; risk_level: string; count: number }>;
  }> {
    return this.request(`/analytics/predictions?period=${period}`);
  }

  async getAnalyticsPatients(): Promise<any> {
    return this.request('/analytics/patients');
  }

  async getAnalyticsFiles(): Promise<any> {
    return this.request('/analytics/files');
  }

  // Health check
  async healthCheck(): Promise<{
    status: string;
    timestamp: string;
    version: string;
  }> {
    return this.request('/health');
  }
}

// Create and export API client instance
export const apiClient = new ApiClient(API_BASE_URL);

// Export default
export default apiClient;
