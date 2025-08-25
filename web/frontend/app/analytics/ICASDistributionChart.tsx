'use client';

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface ICASData {
  icas_grade: string;
  count: number;
}

interface ICASDistributionChartProps {
  data: ICASData[];
}

const ICAS_COLORS = {
  '无': '#10b981',      // 绿色 - 正常
  '轻度': '#f59e0b',    // 橙色 - 轻度
  '中度': '#ef4444',    // 红色 - 中度
  '重度': '#dc2626'     // 深红色 - 重度
};

const ICAS_LABELS = {
  '无': '正常 (0)',
  '轻度': '轻度 (1)',
  '中度': '中度 (2)',
  '重度': '重度 (3)'
};

export default function ICASDistributionChart({ data }: ICASDistributionChartProps) {
  // 转换数据格式并添加颜色和标签
  const chartData = data.map(item => ({
    name: ICAS_LABELS[item.icas_grade as keyof typeof ICAS_LABELS] || item.icas_grade,
    value: item.count,
    color: ICAS_COLORS[item.icas_grade as keyof typeof ICAS_COLORS] || '#6b7280',
    grade: item.icas_grade
  }));

  // 按照严重程度排序
  const sortedData = chartData.sort((a, b) => {
    const order = ['无', '轻度', '中度', '重度'];
    return order.indexOf(a.grade) - order.indexOf(b.grade);
  });

  const total = chartData.reduce((sum, item) => sum + item.value, 0);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0];
      return (
        <div className="bg-white p-3 rounded-lg shadow-lg border text-sm">
          <p className="font-medium">{label}</p>
          <p className="text-gray-600">
            患者数: <span className="font-semibold">{data.value}</span>
          </p>
          <p className="text-gray-600">
            占比: <span className="font-semibold">{total > 0 ? ((data.value / total) * 100).toFixed(1) : 0}%</span>
          </p>
        </div>
      );
    }
    return null;
  };

  if (!data || data.length === 0) {
    return (
      <div className="h-80 flex items-center justify-center">
        <p className="text-gray-500">暂无ICAS数据</p>
      </div>
    );
  }

  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={sortedData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="name" 
            tick={{ fontSize: 12 }}
            axisLine={{ stroke: '#e5e7eb' }}
          />
          <YAxis 
            tick={{ fontSize: 12 }}
            axisLine={{ stroke: '#e5e7eb' }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Bar
            dataKey="value"
            radius={[4, 4, 0, 0]}
          >
            {sortedData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
