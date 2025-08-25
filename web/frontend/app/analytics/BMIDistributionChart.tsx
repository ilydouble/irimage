'use client';

import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

interface BMIData {
  bmi_category: string;
  count: number;
}

interface BMIDistributionChartProps {
  data: BMIData[];
}

const BMI_COLORS = {
  'Underweight': '#ef4444',  // 红色 - 偏瘦
  'Normal': '#10b981',       // 绿色 - 正常
  'Overweight': '#f59e0b',   // 橙色 - 超重
  'Obese': '#dc2626',        // 深红色 - 肥胖
  'Unknown': '#6b7280'       // 灰色 - 未知
};

const BMI_LABELS = {
  'Underweight': '偏瘦',
  'Normal': '正常',
  'Overweight': '超重',
  'Obese': '肥胖',
  'Unknown': '未知'
};

export default function BMIDistributionChart({ data }: BMIDistributionChartProps) {
  // 转换数据格式并添加颜色
  const chartData = data.map(item => ({
    name: BMI_LABELS[item.bmi_category as keyof typeof BMI_LABELS] || item.bmi_category,
    value: item.count,
    color: BMI_COLORS[item.bmi_category as keyof typeof BMI_COLORS] || '#6b7280'
  }));

  const total = chartData.reduce((sum, item) => sum + item.value, 0);

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0];
      return (
        <div className="bg-white p-3 rounded-lg shadow-lg border text-sm">
          <p className="font-medium">{data.name}</p>
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
        <p className="text-gray-500">暂无BMI数据</p>
      </div>
    );
  }

  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={chartData}
            cx="50%"
            cy="50%"
            innerRadius={60}
            outerRadius={120}
            paddingAngle={2}
            dataKey="value"
          >
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Pie>
          <Tooltip content={<CustomTooltip />} />
          <Legend 
            verticalAlign="bottom" 
            height={36}
            iconType="circle"
            wrapperStyle={{ fontSize: '12px' }}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
