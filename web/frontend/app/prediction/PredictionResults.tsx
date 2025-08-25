'use client';

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

interface PredictionResultsProps {
  result: {
    riskScore: number;
    riskLevel: string;
    confidence: number;
    factors: Array<{
      name: string;
      impact: number;
      positive: boolean;
    }>;
    recommendations: string[];
  };
  onReset: () => void;
}

export default function PredictionResults({ result, onReset }: PredictionResultsProps) {
  const getRiskLevelInfo = (level: string) => {
    switch (level) {
      case 'low':
        return {
          text: '低风险',
          color: 'text-green-600',
          bgColor: 'bg-green-100',
          description: '患者目前处于低风险状态，建议保持健康的生活方式'
        };
      case 'medium':
        return {
          text: '中风险',
          color: 'text-yellow-600',
          bgColor: 'bg-yellow-100',
          description: '患者存在一定风险，建议定期检查并注意生活方式调整'
        };
      case 'high':
        return {
          text: '高风险',
          color: 'text-red-600',
          bgColor: 'bg-red-100',
          description: '患者处于高风险状态，建议尽快就医进行详细检查'
        };
      default:
        return {
          text: '未知',
          color: 'text-gray-600',
          bgColor: 'bg-gray-100',
          description: '无法确定风险等级'
        };
    }
  };

  const riskInfo = getRiskLevelInfo(result.riskLevel);

  const factorChartData = result.factors.map(factor => ({
    name: factor.name,
    影响程度: factor.impact,
    类型: factor.positive ? '保护因子' : '危险因子'
  }));

  const riskDistributionData = [
    { name: '当前风险', value: result.riskScore, color: '#ef4444' },
    { name: '安全区间', value: 100 - result.riskScore, color: '#10b981' }
  ];

  const downloadReport = () => {
    console.log('下载预测报告');
  };

  const saveToDatabase = () => {
    console.log('保存到数据库');
  };

  return (
    <div className="space-y-8">
      {/* 结果概览 */}
      <div className="bg-white p-8 rounded-xl shadow-lg border">
        <div className="flex justify-between items-start mb-6">
          <h2 className="text-2xl font-bold text-gray-900">ICAS风险预测结果</h2>
          <button
            onClick={onReset}
            className="text-gray-500 hover:text-gray-700 cursor-pointer"
          >
            <i className="ri-close-line text-xl"></i>
          </button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* 风险评分 */}
          <div className="text-center">
            <div className="relative w-32 h-32 mx-auto mb-4">
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <div className="text-3xl font-bold text-gray-900">
                    {result.riskScore.toFixed(1)}
                  </div>
                  <div className="text-sm text-gray-500">风险评分</div>
                </div>
              </div>
              <svg className="w-32 h-32 transform -rotate-90" viewBox="0 0 100 100">
                <circle
                  cx="50"
                  cy="50"
                  r="40"
                  stroke="#e5e7eb"
                  strokeWidth="8"
                  fill="none"
                />
                <circle
                  cx="50"
                  cy="50"
                  r="40"
                  stroke={result.riskLevel === 'high' ? '#ef4444' : result.riskLevel === 'medium' ? '#f59e0b' : '#10b981'}
                  strokeWidth="8"
                  fill="none"
                  strokeDasharray={`${(result.riskScore / 100) * 251.2} 251.2`}
                  strokeLinecap="round"
                />
              </svg>
            </div>
          </div>

          {/* 风险等级 */}
          <div className="text-center">
            <div className={`inline-flex px-4 py-2 rounded-full text-lg font-semibold ${riskInfo.bgColor} ${riskInfo.color} mb-4`}>
              {riskInfo.text}
            </div>
            <p className="text-gray-600 text-sm">{riskInfo.description}</p>
          </div>

          {/* 置信度 */}
          <div className="text-center">
            <div className="text-3xl font-bold text-blue-600 mb-2">
              {result.confidence.toFixed(1)}%
            </div>
            <div className="text-sm text-gray-500 mb-2">预测置信度</div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-600 h-2 rounded-full" 
                style={{ width: `${result.confidence}%` }}
              ></div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* 风险因子分析 */}
        <div className="bg-white p-6 rounded-xl shadow-lg border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">风险因子分析</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={factorChartData} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 12 }} />
                <YAxis 
                  type="category" 
                  dataKey="name" 
                  tick={{ fontSize: 12 }}
                  width={80}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'white', 
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                    fontSize: '12px'
                  }}
                />
                <Bar
                  dataKey="影响程度"
                  fill="#8884d8"
                  radius={[0, 4, 4, 0]}
                >
                  {factorChartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.类型 === '保护因子' ? '#10b981' : '#ef4444'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          
          {/* 因子详情 */}
          <div className="mt-4 space-y-2">
            {result.factors.map((factor, index) => (
              <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                <div className="flex items-center">
                  <div className={`w-3 h-3 rounded-full mr-3 ${factor.positive ? 'bg-green-500' : 'bg-red-500'}`}></div>
                  <span className="text-sm text-gray-700">{factor.name}</span>
                </div>
                <div className="flex items-center">
                  <span className="text-sm font-medium text-gray-900 mr-2">{factor.impact.toFixed(1)}</span>
                  <span className={`text-xs px-2 py-1 rounded-full ${
                    factor.positive ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                  }`}>
                    {factor.positive ? '保护' : '危险'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 风险分布 */}
        <div className="bg-white p-6 rounded-xl shadow-lg border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">风险分布</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={riskDistributionData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={120}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {riskDistributionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(value) => [`${typeof value === 'number' ? value.toFixed(1) : value}%`, '']}
                  contentStyle={{
                    backgroundColor: 'white',
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                    fontSize: '12px'
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* 风险等级说明 */}
          <div className="mt-4 space-y-2">
            <div className="flex items-center">
              <div className="w-3 h-3 bg-green-500 rounded-full mr-3"></div>
              <span className="text-sm text-gray-700">低风险 (0-30分)</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-yellow-500 rounded-full mr-3"></div>
              <span className="text-sm text-gray-700">中风险 (31-70分)</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-red-500 rounded-full mr-3"></div>
              <span className="text-sm text-gray-700">高风险 (71-100分)</span>
            </div>
          </div>
        </div>
      </div>

      {/* 健康建议 */}
      <div className="bg-white p-6 rounded-xl shadow-lg border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          <i className="ri-lightbulb-line text-yellow-500 mr-2"></i>
          健康建议
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {result.recommendations.map((recommendation, index) => (
            <div key={index} className="flex items-start p-4 bg-blue-50 rounded-lg">
              <div className="w-6 h-6 flex items-center justify-center bg-blue-100 rounded-full mt-0.5 mr-3 flex-shrink-0">
                <i className="ri-check-line text-xs text-blue-600"></i>
              </div>
              <p className="text-sm text-gray-700">{recommendation}</p>
            </div>
          ))}
        </div>
      </div>

      {/* 操作按钮 */}
      <div className="flex justify-center space-x-4">
        <button
          onClick={downloadReport}
          className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 font-medium whitespace-nowrap cursor-pointer"
        >
          <i className="ri-download-line mr-2"></i>
          下载报告
        </button>
        <button
          onClick={saveToDatabase}
          className="bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700 font-medium whitespace-nowrap cursor-pointer"
        >
          <i className="ri-save-line mr-2"></i>
          保存记录
        </button>
        <button
          onClick={onReset}
          className="bg-gray-100 text-gray-700 px-6 py-3 rounded-lg hover:bg-gray-200 font-medium whitespace-nowrap cursor-pointer"
        >
          <i className="ri-refresh-line mr-2"></i>
          重新预测
        </button>
      </div>
    </div>
  );
}