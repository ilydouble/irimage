
'use client';

import { useState } from 'react';

interface PredictionFormProps {
  onPredict: (data: any) => void;
  isLoading: boolean;
}

export default function PredictionForm({ onPredict, isLoading }: PredictionFormProps) {
  const [formData, setFormData] = useState({
    patientId: '',
    name: '',
    gender: '',
    age: '',
    height: '',
    weight: '',
    waist: '',
    hip: '',
    neck: '',
    thermal24: '',
    thermal25: '',
    voice25: '',
    bloodPressure: '',
    cholesterol: '',
    diabetes: '',
    smoking: '',
    exercise: ''
  });

  const [errors, setErrors] = useState<{ [key: string]: string }>({});
  const [showThermal24Upload, setShowThermal24Upload] = useState(false);
  const [showThermal25Upload, setShowThermal25Upload] = useState(false);
  const [showVoiceUpload, setShowVoiceUpload] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState({
    thermal24: null as File | null,
    thermal25: null as File | null,
    voice25: null as File | null
  });

  const handleInputChange = (field: string, value: string) => {
    setFormData({ ...formData, [field]: value });
    if (errors[field]) {
      setErrors({ ...errors, [field]: '' });
    }

    // 检查是否需要弹出上传界面
    if (field === 'thermal24' && value === 'yes') {
      setShowThermal24Upload(true);
    } else if (field === 'thermal24' && value === 'no') {
      setShowThermal24Upload(false);
      setUploadedFiles(prev => ({ ...prev, thermal24: null }));
    }

    if (field === 'thermal25' && value === 'yes') {
      setShowThermal25Upload(true);
    } else if (field === 'thermal25' && value === 'no') {
      setShowThermal25Upload(false);
      setUploadedFiles(prev => ({ ...prev, thermal25: null }));
    }

    if (field === 'voice25' && value === 'yes') {
      setShowVoiceUpload(true);
    } else if (field === 'voice25' && value === 'no') {
      setShowVoiceUpload(false);
      setUploadedFiles(prev => ({ ...prev, voice25: null }));
    }
  };

  const handleFileUpload = (type: 'thermal24' | 'thermal25' | 'voice25', file: File) => {
    setUploadedFiles(prev => ({ ...prev, [type]: file }));
  };

  const validateForm = () => {
    const newErrors: { [key: string]: string } = {};

    if (!formData.patientId.trim()) newErrors.patientId = '请输入患者编号';
    if (!formData.name.trim()) newErrors.name = '请输入患者姓名';
    if (!formData.gender) newErrors.gender = '请选择性别';
    if (!formData.age || parseInt(formData.age) < 1 || parseInt(formData.age) > 120) {
      newErrors.age = '请输入有效年龄（1-120）';
    }
    if (!formData.height || parseInt(formData.height) < 100 || parseInt(formData.height) > 250) {
      newErrors.height = '请输入有效身高（100-250cm）';
    }
    if (!formData.weight || parseInt(formData.weight) < 30 || parseInt(formData.weight) > 200) {
      newErrors.weight = '请输入有效体重（30-200kg）';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (validateForm()) {
      onPredict({ ...formData, uploadedFiles });
    }
  };

  const getBMI = () => {
    if (formData.height && formData.weight) {
      const heightInM = parseInt(formData.height) / 100;
      const bmi = parseInt(formData.weight) / (heightInM * heightInM);
      return bmi.toFixed(1);
    }
    return '';
  };

  // 文件上传组件
  const FileUploadModal = ({ 
    isOpen, 
    onClose, 
    type, 
    title, 
    acceptTypes,
    onFileSelect 
  }: {
    isOpen: boolean;
    onClose: () => void;
    type: 'thermal24' | 'thermal25' | 'voice25';
    title: string;
    acceptTypes: string;
    onFileSelect: (file: File) => void;
  }) => {
    const [dragOver, setDragOver] = useState(false);
    const currentFile = uploadedFiles[type];

    const handleDrop = (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        onFileSelect(files[0]);
      }
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files.length > 0) {
        onFileSelect(e.target.files[0]);
      }
    };

    const isAudio = type === 'voice25';

    if (!isOpen) return null;

    return (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
        <div className="bg-white rounded-xl p-6 max-w-md w-full mx-4">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
            <button
              onClick={onClose}
              className="w-8 h-8 flex items-center justify-center text-gray-400 hover:text-gray-600 cursor-pointer"
            >
              <i className="ri-close-line text-xl"></i>
            </button>
          </div>

          {currentFile ? (
            <div className="text-center">
              <div className="mb-4">
                <div className="w-16 h-16 flex items-center justify-center bg-green-100 rounded-full mx-auto mb-2">
                  <i className={`ri-${isAudio ? 'mic' : 'image'}-line text-2xl text-green-600`}></i>
                </div>
                <p className="text-sm font-medium text-gray-900">{currentFile.name}</p>
                <p className="text-xs text-gray-500">
                  {(currentFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
              <div className="flex space-x-3">
                <button
                  onClick={() => {
                    setUploadedFiles(prev => ({ ...prev, [type]: null }));
                  }}
                  className="flex-1 bg-gray-100 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-200 text-sm cursor-pointer whitespace-nowrap"
                >
                  重新选择
                </button>
                <button
                  onClick={onClose}
                  className="flex-1 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 text-sm cursor-pointer whitespace-nowrap"
                >
                  确认上传
                </button>
              </div>
            </div>
          ) : (
            <div>
              <div
                className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                  dragOver ? 'border-blue-400 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
                }`}
                onDrop={handleDrop}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onClick={() => document.getElementById(`file-input-${type}`)?.click()}
              >
                <div className="w-12 h-12 flex items-center justify-center bg-gray-100 rounded-full mx-auto mb-4">
                  <i className={`ri-${isAudio ? 'mic' : 'image'}-line text-xl text-gray-600`}></i>
                </div>
                <p className="text-sm text-gray-600 mb-2">
                  点击选择文件或拖拽文件到此处
                </p>
                <p className="text-xs text-gray-500">
                  {isAudio ? '支持 MP3、WAV、M4A 格式' : '支持 JPG、PNG、JPEG 格式'}
                </p>
              </div>

              <input
                id={`file-input-${type}`}
                type="file"
                accept={acceptTypes}
                onChange={handleFileChange}
                className="hidden"
              />

              <div className="mt-4 flex justify-end">
                <button
                  onClick={onClose}
                  className="bg-gray-100 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-200 text-sm cursor-pointer whitespace-nowrap"
                >
                  取消
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="bg-white p-6 rounded-xl shadow-lg border">
      <h3 className="text-lg font-semibold text-gray-900 mb-6">患者信息录入</h3>
      
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* 基本信息 */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              患者编号 <span className="text-red-500">*</span>
            </label>
            <input
              type="text"
              placeholder="如：P001"
              className={`w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm ${
                errors.patientId ? 'border-red-300' : 'border-gray-300'
              }`}
              value={formData.patientId}
              onChange={(e) => handleInputChange('patientId', e.target.value)}
            />
            {errors.patientId && <p className="mt-1 text-xs text-red-500">{errors.patientId}</p>}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              患者姓名 <span className="text-red-500">*</span>
            </label>
            <input
              type="text"
              placeholder="请输入姓名"
              className={`w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm ${
                errors.name ? 'border-red-300' : 'border-gray-300'
              }`}
              value={formData.name}
              onChange={(e) => handleInputChange('name', e.target.value)}
            />
            {errors.name && <p className="mt-1 text-xs text-red-500">{errors.name}</p>}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              性别 <span className="text-red-500">*</span>
            </label>
            <select
              className={`w-full pr-8 pl-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm ${
                errors.gender ? 'border-red-300' : 'border-gray-300'
              }`}
              value={formData.gender}
              onChange={(e) => handleInputChange('gender', e.target.value)}
            >
              <option value="">请选择</option>
              <option value="male">男</option>
              <option value="female">女</option>
            </select>
            {errors.gender && <p className="mt-1 text-xs text-red-500">{errors.gender}</p>}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              年龄 <span className="text-red-500">*</span>
            </label>
            <input
              type="number"
              placeholder="岁"
              min="1"
              max="120"
              className={`w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm ${
                errors.age ? 'border-red-300' : 'border-gray-300'
              }`}
              value={formData.age}
              onChange={(e) => handleInputChange('age', e.target.value)}
            />
            {errors.age && <p className="mt-1 text-xs text-red-500">{errors.age}</p>}
          </div>
        </div>

        {/* 体征测量 */}
        <div>
          <h4 className="text-md font-medium text-gray-900 mb-4">体征测量</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                身高 (cm) <span className="text-red-500">*</span>
              </label>
              <input
                type="number"
                placeholder="如：175"
                min="100"
                max="250"
                className={`w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm ${
                  errors.height ? 'border-red-300' : 'border-gray-300'
                }`}
                value={formData.height}
                onChange={(e) => handleInputChange('height', e.target.value)}
              />
              {errors.height && <p className="mt-1 text-xs text-red-500">{errors.height}</p>}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                体重 (kg) <span className="text-red-500">*</span>
              </label>
              <input
                type="number"
                placeholder="如：70"
                min="30"
                max="200"
                className={`w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm ${
                  errors.weight ? 'border-red-300' : 'border-gray-300'
                }`}
                value={formData.weight}
                onChange={(e) => handleInputChange('weight', e.target.value)}
              />
              {errors.weight && <p className="mt-1 text-xs text-red-500">{errors.weight}</p>}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">BMI</label>
              <input
                type="text"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg bg-gray-50 text-sm"
                value={getBMI()}
                readOnly
                placeholder="自动计算"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">腰围 (cm)</label>
              <input
                type="number"
                placeholder="如：85"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                value={formData.waist}
                onChange={(e) => handleInputChange('waist', e.target.value)}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">臀围 (cm)</label>
              <input
                type="number"
                placeholder="如：95"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                value={formData.hip}
                onChange={(e) => handleInputChange('hip', e.target.value)}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">颈围 (cm)</label>
              <input
                type="number"
                placeholder="如：38"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                value={formData.neck}
                onChange={(e) => handleInputChange('neck', e.target.value)}
              />
            </div>
          </div>
        </div>

        {/* 检查完成情况 */}
        <div>
          <h4 className="text-md font-medium text-gray-900 mb-4">检查完成情况</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">24年热成像</label>
              <div className="flex items-center space-x-2">
                <select
                  className="flex-1 pr-8 pl-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                  value={formData.thermal24}
                  onChange={(e) => handleInputChange('thermal24', e.target.value)}
                >
                  <option value="">请选择</option>
                  <option value="yes">已完成</option>
                  <option value="no">未完成</option>
                </select>
                {formData.thermal24 === 'yes' && (
                  <button
                    type="button"
                    onClick={() => setShowThermal24Upload(true)}
                    className={`px-3 py-2 text-xs rounded-lg cursor-pointer whitespace-nowrap ${
                      uploadedFiles.thermal24 ? 'bg-green-100 text-green-700' : 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                    }`}
                  >
                    {uploadedFiles.thermal24 ? (
                      <><i className="ri-check-line mr-1"></i>已上传</>
                    ) : (
                      <><i className="ri-upload-line mr-1"></i>上传</>
                    )}
                  </button>
                )}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">25年热成像</label>
              <div className="flex items-center space-x-2">
                <select
                  className="flex-1 pr-8 pl-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                  value={formData.thermal25}
                  onChange={(e) => handleInputChange('thermal25', e.target.value)}
                >
                  <option value="">请选择</option>
                  <option value="yes">已完成</option>
                  <option value="no">未完成</option>
                </select>
                {formData.thermal25 === 'yes' && (
                  <button
                    type="button"
                    onClick={() => setShowThermal25Upload(true)}
                    className={`px-3 py-2 text-xs rounded-lg cursor-pointer whitespace-nowrap ${
                      uploadedFiles.thermal25 ? 'bg-green-100 text-green-700' : 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                    }`}
                  >
                    {uploadedFiles.thermal25 ? (
                      <><i className="ri-check-line mr-1"></i>已上传</>
                    ) : (
                      <><i className="ri-upload-line mr-1"></i>上传</>
                    )}
                  </button>
                )}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">25年语音检查</label>
              <div className="flex items-center space-x-2">
                <select
                  className="flex-1 pr-8 pl-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                  value={formData.voice25}
                  onChange={(e) => handleInputChange('voice25', e.target.value)}
                >
                  <option value="">请选择</option>
                  <option value="yes">已完成</option>
                  <option value="no">未完成</option>
                </select>
                {formData.voice25 === 'yes' && (
                  <button
                    type="button"
                    onClick={() => setShowVoiceUpload(true)}
                    className={`px-3 py-2 text-xs rounded-lg cursor-pointer whitespace-nowrap ${
                      uploadedFiles.voice25 ? 'bg-green-100 text-green-700' : 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                    }`}
                  >
                    {uploadedFiles.voice25 ? (
                      <><i className="ri-check-line mr-1"></i>已上传</>
                    ) : (
                      <><i className="ri-mic-line mr-1"></i>上传</>
                    )}
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* 其他健康指标 */}
        <div>
          <h4 className="text-md font-medium text-gray-900 mb-4">其他健康指标</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">血压情况</label>
              <select
                className="w-full pr-8 pl-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                value={formData.bloodPressure}
                onChange={(e) => handleInputChange('bloodPressure', e.target.value)}
              >
                <option value="">请选择</option>
                <option value="normal">正常</option>
                <option value="high">偏高</option>
                <option value="low">偏低</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">胆固醇水平</label>
              <select
                className="w-full pr-8 pl-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                value={formData.cholesterol}
                onChange={(e) => handleInputChange('cholesterol', e.target.value)}
              >
                <option value="">请选择</option>
                <option value="normal">正常</option>
                <option value="high">偏高</option>
                <option value="low">偏低</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">糖尿病史</label>
              <select
                className="w-full pr-8 pl-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                value={formData.diabetes}
                onChange={(e) => handleInputChange('diabetes', e.target.value)}
              >
                <option value="">请选择</option>
                <option value="no">无</option>
                <option value="type1">1型糖尿病</option>
                <option value="type2">2型糖尿病</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">吸烟情况</label>
              <select
                className="w-full pr-8 pl-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                value={formData.smoking}
                onChange={(e) => handleInputChange('smoking', e.target.value)}
              >
                <option value="">请选择</option>
                <option value="never">从不吸烟</option>
                <option value="former">已戒烟</option>
                <option value="current">目前吸烟</option>
              </select>
            </div>
          </div>
        </div>

        {/* 提交按钮 */}
        <div className="flex justify-center pt-6">
          <button
            type="submit"
            disabled={isLoading}
            className={`px-8 py-3 rounded-lg font-medium text-white whitespace-nowrap cursor-pointer ${
              isLoading 
                ? 'bg-gray-400 cursor-not-allowed' 
                : 'bg-blue-600 hover:bg-blue-700'
            }`}
          >
            {isLoading ? (
              <span className="flex items-center">
                <i className="ri-loader-4-line animate-spin mr-2"></i>
                预测中...
              </span>
            ) : (
              '开始预测'
            )}
          </button>
        </div>
      </form>

      {/* 文件上传弹窗 */}
      <FileUploadModal
        isOpen={showThermal24Upload}
        onClose={() => setShowThermal24Upload(false)}
        type="thermal24"
        title="上传24年热成像图片"
        acceptTypes="image/*"
        onFileSelect={(file) => handleFileUpload('thermal24', file)}
      />

      <FileUploadModal
        isOpen={showThermal25Upload}
        onClose={() => setShowThermal25Upload(false)}
        type="thermal25"
        title="上传25年热成像图片"
        acceptTypes="image/*"
        onFileSelect={(file) => handleFileUpload('thermal25', file)}
      />

      <FileUploadModal
        isOpen={showVoiceUpload}
        onClose={() => setShowVoiceUpload(false)}
        type="voice25"
        title="上传25年语音文件"
        acceptTypes="audio/*"
        onFileSelect={(file) => handleFileUpload('voice25', file)}
      />
    </div>
  );
}
