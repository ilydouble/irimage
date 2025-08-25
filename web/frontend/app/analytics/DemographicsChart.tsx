'use client';

import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

interface DistributionData {
  genders: Array<{ gender: string; count: number }>;
  age_groups: Array<{ age_group: string; count: number }>;
}

interface DemographicsChartProps {
  data?: DistributionData;
}

const AGE_COLORS = ['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b', '#ef4444'];
const GENDER_COLORS = { 'male': '#3b82f6', 'female': '#ec4899' };

const GENDER_LABELS = {
  'male': '男性',
  'female': '女性'
};

export default function DemographicsChart({ data }: DemographicsChartProps) {
  // 转换年龄数据
  const ageData = data?.age_groups.map((item, index) => ({
    name: item.age_group,
    value: item.count,
    color: AGE_COLORS[index % AGE_COLORS.length]
  })) || [];

  // 转换性别数据
  const genderData = data?.genders.map(item => ({
    name: GENDER_LABELS[item.gender as keyof typeof GENDER_LABELS] || item.gender,
    value: item.count,
    color: GENDER_COLORS[item.gender as keyof typeof GENDER_COLORS] || '#6b7280'
  })) || [];

  const totalAge = ageData.reduce((sum, item) => sum + item.value, 0);
  const totalGender = genderData.reduce((sum, item) => sum + item.value, 0);

  const CustomTooltip = ({ active, payload, chartType }: any) => {
    if (active && payload && payload.length) {
      const total = chartType === 'age' ? totalAge : totalGender;
      return (
        <div className="bg-white p-3 rounded-lg shadow-lg border text-sm">
          <p className="font-medium">{payload[0].name}</p>
          <p className="text-gray-600">
            患者数: <span className="font-semibold">{payload[0].value}</span>
          </p>
          <p className="text-gray-600">
            占比: <span className="font-semibold">{total > 0 ? ((payload[0].value / total) * 100).toFixed(1) : 0}%</span>
          </p>
        </div>
      );
    }
    return null;
  };

  if (!data) {
    return (
      <div className="h-80 flex items-center justify-center">
        <p className="text-gray-500">暂无人口统计数据</p>
      </div>
    );
  }

  return (
    <div className="h-80">
      <div className="grid grid-cols-2 h-full">
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-2 text-center">年龄分布</h4>
          <ResponsiveContainer width="100%" height="90%">
            <PieChart>
              <Pie
                data={ageData}
                cx="50%"
                cy="50%"
                innerRadius={40}
                outerRadius={80}
                paddingAngle={2}
                dataKey="value"
              >
                {ageData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip content={(props) => <CustomTooltip {...props} chartType="age" />} />
              <Legend
                verticalAlign="bottom"
                height={36}
                iconType="circle"
                wrapperStyle={{ fontSize: '10px' }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-2 text-center">性别分布</h4>
          <ResponsiveContainer width="100%" height="90%">
            <PieChart>
              <Pie
                data={genderData}
                cx="50%"
                cy="50%"
                innerRadius={40}
                outerRadius={80}
                paddingAngle={2}
                dataKey="value"
              >
                {genderData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip content={(props) => <CustomTooltip {...props} chartType="gender" />} />
              <Legend
                verticalAlign="bottom"
                height={36}
                iconType="circle"
                wrapperStyle={{ fontSize: '10px' }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}