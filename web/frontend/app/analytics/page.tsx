'use client';

import Link from 'next/link';
import { useState, useEffect } from 'react';
import AccuracyChart from './AccuracyChart';
import DemographicsChart from './DemographicsChart';
import BMIDistributionChart from './BMIDistributionChart';
import ICASDistributionChart from './ICASDistributionChart';

interface OverviewData {
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
}

interface PatientsData {
  bmi_distribution: Array<{ bmi_category: string; count: number }>;
  file_completion: {
    thermal_24h_rate: string;
    thermal_25h_rate: string;
    voice_25h_rate: string;
    thermal_24h_count: number;
    thermal_25h_count: number;
    voice_25h_count: number;
    total_patients: number;
  };
  icas_correlation: Array<{
    gender: string;
    age_group: string;
    icas_count: number;
    total_count: number;
    icas_percentage: number;
  }>;
  physical_stats: {
    avg_height: number;
    avg_weight: number;
    avg_bmi: number;
    avg_waist: number;
    avg_hip: number;
    avg_neck: number;
  };
}

export default function Analytics() {
  const [overviewData, setOverviewData] = useState<OverviewData | null>(null);
  const [patientsData, setPatientsData] = useState<PatientsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAnalyticsData = async () => {
      try {
        setLoading(true);

        // 并行获取所有统计数据
        const [overviewResponse, patientsResponse] = await Promise.all([
          fetch('http://localhost:3001/api/analytics/overview'),
          fetch('http://localhost:3001/api/analytics/patients')
        ]);

        if (!overviewResponse.ok || !patientsResponse.ok) {
          throw new Error('Failed to fetch analytics data');
        }

        const overview = await overviewResponse.json();
        const patients = await patientsResponse.json();

        setOverviewData(overview);
        setPatientsData(patients);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchAnalyticsData();
  }, []);

  // 生成统计卡片数据
  const getStatsCards = () => {
    if (!overviewData || !patientsData) return [];

    const { overview } = overviewData;
    const { file_completion } = patientsData;

    return [
      {
        label: '总患者数',
        value: overview.total_patients.toLocaleString(),
        change: `+${overview.recent_predictions}`,
        color: 'blue'
      },
      {
        label: '24年热成像完成率',
        value: file_completion.thermal_24h_rate,
        change: `${file_completion.thermal_24h_count}/${file_completion.total_patients}`,
        color: 'green'
      },
      {
        label: '25年热成像完成率',
        value: file_completion.thermal_25h_rate,
        change: `${file_completion.thermal_25h_count}/${file_completion.total_patients}`,
        color: 'purple'
      },
      {
        label: '语音检查完成率',
        value: file_completion.voice_25h_rate,
        change: `${file_completion.voice_25h_count}/${file_completion.total_patients}`,
        color: 'orange'
      },
    ];
  };

  const accuracyStats = [
    { model: 'ICAS风险预测', accuracy: '94.2%', precision: '91.8%', recall: '96.5%' },
    { model: '热成像分析', accuracy: '89.7%', precision: '87.3%', recall: '92.1%' },
    { model: '语音识别', accuracy: '92.4%', precision: '90.6%', recall: '94.2%' },
    { model: '综合评估', accuracy: '96.1%', precision: '94.7%', recall: '97.8%' },
  ];

  if (loading) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">加载统计数据中...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <div className="text-center">
          <div className="text-red-600 text-xl mb-4">加载失败</div>
          <p className="text-gray-600">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            重新加载
          </button>
        </div>
      </div>
    );
  }

  const stats = getStatsCards();

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
              <Link href="/analytics" className="text-blue-600 px-3 py-2 rounded-md text-sm font-medium whitespace-nowrap cursor-pointer">
                统计分析
              </Link>
              <Link href="/search" className="text-gray-700 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium whitespace-nowrap cursor-pointer">
                患者检索
              </Link>
              <Link href="/prediction" className="text-gray-700 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium whitespace-nowrap cursor-pointer">
                模型预测
              </Link>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* 页面标题 */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">患者统计与预测准确率分析</h1>
          <p className="text-gray-600">实时监控患者数据统计和模型预测性能评估</p>
        </div>

        {/* 统计卡片 */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          {stats.map((stat, index) => (
            <div key={index} className="bg-white p-6 rounded-xl shadow-lg border">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 mb-1">{stat.label}</p>
                  <p className="text-2xl font-bold text-gray-900">{stat.value}</p>
                  <p className={`text-sm font-medium text-${stat.color}-600`}>{stat.change}</p>
                </div>
                <div className={`w-12 h-12 flex items-center justify-center bg-${stat.color}-100 rounded-lg`}>
                  <i className={`ri-arrow-up-line text-xl text-${stat.color}-600`}></i>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* BMI分布 */}
          <div className="bg-white p-6 rounded-xl shadow-lg border">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">BMI分布</h3>
            <BMIDistributionChart data={patientsData?.bmi_distribution || []} />
          </div>

          {/* ICAS分布 */}
          <div className="bg-white p-6 rounded-xl shadow-lg border">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">ICAS分布</h3>
            <ICASDistributionChart data={overviewData?.distributions.icas_grades || []} />
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* 年龄性别分布 */}
          <div className="bg-white p-6 rounded-xl shadow-lg border">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">年龄性别分布</h3>
            <DemographicsChart data={overviewData?.distributions} />
          </div>

          {/* 预测准确率分析 */}
          <div className="bg-white p-6 rounded-xl shadow-lg border">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">模型预测准确率</h3>
            <AccuracyChart />
          </div>
        </div>

        {/* 模型性能详细表 */}
        <div className="bg-white rounded-xl shadow-lg border">
          <div className="p-6 border-b">
            <h3 className="text-lg font-semibold text-gray-900">模型性能指标</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">模型名称</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">准确率</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">精确率</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">召回率</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">状态</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {accuracyStats.map((item, index) => (
                  <tr key={index} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {item.model}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-green-100 text-green-800">
                        {item.accuracy}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{item.precision}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{item.recall}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-blue-100 text-blue-800">
                        运行中
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
