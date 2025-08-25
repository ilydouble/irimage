'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import PredictionForm from './PredictionForm';
import PredictionResults from './PredictionResults';
import { apiClient, Prediction } from '../../lib/api';

export default function ModelPrediction() {
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [recentPredictions, setRecentPredictions] = useState<Prediction[]>([]);
  const [loadingPredictions, setLoadingPredictions] = useState(true);

  // 获取最近的预测记录
  const fetchRecentPredictions = async () => {
    try {
      setLoadingPredictions(true);
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001/api'}/predictions?limit=10`);
      const data = await response.json();
      setRecentPredictions(data.predictions || []);
    } catch (error) {
      console.error('获取预测记录失败:', error);
    } finally {
      setLoadingPredictions(false);
    }
  };

  useEffect(() => {
    fetchRecentPredictions();
  }, []);

  // 格式化ICAS等级显示
  const getICASBadge = (hasIcas: boolean, grade: string) => {
    if (!hasIcas || grade === '无') {
      return (
        <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-green-100 text-green-800">
          无风险
        </span>
      );
    }

    const colorMap = {
      '轻度': 'bg-yellow-100 text-yellow-800',
      '中度': 'bg-orange-100 text-orange-800',
      '重度': 'bg-red-100 text-red-800'
    };

    return (
      <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${colorMap[grade as keyof typeof colorMap] || 'bg-gray-100 text-gray-800'}`}>
        {grade}
      </span>
    );
  };

  // 格式化时间显示
  const formatDateTime = (dateString: string) => {
    return new Date(dateString).toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const handlePredict = async (patientData: any) => {
    setIsLoading(true);
    
    // 模拟预测过程
    setTimeout(() => {
      const mockResult = {
        riskScore: Math.random() * 100,
        riskLevel: Math.random() > 0.7 ? 'high' : Math.random() > 0.4 ? 'medium' : 'low',
        confidence: 85 + Math.random() * 15,
        factors: [
          { name: '年龄因子', impact: Math.random() * 30 + 10, positive: Math.random() > 0.5 },
          { name: 'BMI指数', impact: Math.random() * 25 + 5, positive: Math.random() > 0.5 },
          { name: '颈围比例', impact: Math.random() * 20 + 5, positive: Math.random() > 0.5 },
          { name: '热成像特征', impact: Math.random() * 35 + 10, positive: Math.random() > 0.5 },
          { name: '语音特征', impact: Math.random() * 30 + 5, positive: Math.random() > 0.5 }
        ],
        recommendations: [
          '建议定期进行颈动脉超声检查',
          '保持健康的生活方式，控制体重',
          '定期监测血压和血脂水平',
          '如有异常症状应及时就医'
        ]
      };
      
      setPredictionResult(mockResult);
      setIsLoading(false);
    }, 2000);
  };

  const resetPrediction = () => {
    setPredictionResult(null);
  };

  return (
    <div className="min-h-screen bg-white">
      {/* 导航 */}
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <Link href="/" className="text-xl font-bold text-gray-900 font-['Pacifico'] cursor-pointer">PatientCare</Link>
            </div>
            <div className="flex items-center space-x-8">
              <Link href="/analytics" className="text-gray-700 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium whitespace-nowrap cursor-pointer">
                统计分析
              </Link>
              <Link href="/search" className="text-gray-700 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium whitespace-nowrap cursor-pointer">
                患者检索
              </Link>
              <Link href="/prediction" className="text-blue-600 px-3 py-2 rounded-md text-sm font-medium whitespace-nowrap cursor-pointer">
                模型预测
              </Link>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* 页面标题 */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">ICAS风险预测模型</h1>
          <p className="text-gray-600">基于人工智能算法的颅内动脉粥样硬化性狭窄风险评估</p>
        </div>

        {!predictionResult ? (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* 预测表单 */}
            <div className="lg:col-span-2">
              <PredictionForm 
                onPredict={handlePredict}
                isLoading={isLoading}
              />
            </div>

            {/* 模型信息 */}
            <div className="space-y-6">
              <div className="bg-white p-6 rounded-xl shadow-lg border">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">模型信息</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">模型版本</span>
                    <span className="text-sm font-medium">v2.1.0</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">训练样本</span>
                    <span className="text-sm font-medium">15,247</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">准确率</span>
                    <span className="text-sm font-medium text-green-600">94.2%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">最后更新</span>
                    <span className="text-sm font-medium">2024-12-20</span>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-br from-blue-50 to-indigo-50 p-6 rounded-xl border">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">预测说明</h3>
                <div className="space-y-2 text-sm text-gray-600">
                  <p>• 本模型基于多维度生理指标进行风险评估</p>
                  <p>• 整合热成像和语音特征数据</p>
                  <p>• 提供个性化的健康建议</p>
                  <p>• 预测结果仅供临床参考</p>
                </div>
              </div>

              <div className="bg-white p-6 rounded-xl shadow-lg border">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">风险等级说明</h3>
                <div className="space-y-3">
                  <div className="flex items-center">
                    <div className="w-3 h-3 bg-green-500 rounded-full mr-3"></div>
                    <div>
                      <div className="text-sm font-medium">低风险 (0-30分)</div>
                      <div className="text-xs text-gray-500">建议定期体检</div>
                    </div>
                  </div>
                  <div className="flex items-center">
                    <div className="w-3 h-3 bg-yellow-500 rounded-full mr-3"></div>
                    <div>
                      <div className="text-sm font-medium">中风险 (31-70分)</div>
                      <div className="text-xs text-gray-500">需要密切关注</div>
                    </div>
                  </div>
                  <div className="flex items-center">
                    <div className="w-3 h-3 bg-red-500 rounded-full mr-3"></div>
                    <div>
                      <div className="text-sm font-medium">高风险 (71-100分)</div>
                      <div className="text-xs text-gray-500">建议及时就医</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <PredictionResults 
            result={predictionResult}
            onReset={resetPrediction}
          />
        )}

        {/* 历史预测记录 */}
        {!predictionResult && (
          <div className="mt-12 bg-white rounded-xl shadow-lg border">
            <div className="p-6 border-b">
              <h3 className="text-lg font-semibold text-gray-900">最近预测记录</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">时间</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">患者编号</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">患者姓名</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">ICAS预测</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">预测等级</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">置信度</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {loadingPredictions ? (
                    <tr>
                      <td colSpan={6} className="px-6 py-4 text-center text-sm text-gray-500">
                        加载中...
                      </td>
                    </tr>
                  ) : recentPredictions.length === 0 ? (
                    <tr>
                      <td colSpan={6} className="px-6 py-4 text-center text-sm text-gray-500">
                        暂无预测记录
                      </td>
                    </tr>
                  ) : (
                    recentPredictions.map((prediction) => (
                      <tr key={prediction.id} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {formatDateTime(prediction.created_at)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {prediction.patient_id}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {prediction.patient_name || '-'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {prediction.predicted_icas ? '是' : '否'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          {getICASBadge(prediction.predicted_icas, prediction.predicted_icas_grade)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {prediction.predicted_icas_confidence ? `${prediction.predicted_icas_confidence}%` : '-'}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}