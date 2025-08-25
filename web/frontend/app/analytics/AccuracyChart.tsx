'use client';

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const data = [
  { 
    name: 'ICAS风险预测',
    准确率: 94.2,
    精确率: 91.8,
    召回率: 96.5
  },
  { 
    name: '热成像分析',
    准确率: 89.7,
    精确率: 87.3,
    召回率: 92.1
  },
  { 
    name: '语音识别',
    准确率: 92.4,
    精确率: 90.6,
    召回率: 94.2
  },
  { 
    name: '综合评估',
    准确率: 96.1,
    精确率: 94.7,
    召回率: 97.8
  }
];

export default function AccuracyChart() {
  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="name" 
            tick={{ fontSize: 12 }}
            axisLine={{ stroke: '#e5e7eb' }}
          />
          <YAxis 
            tick={{ fontSize: 12 }}
            axisLine={{ stroke: '#e5e7eb' }}
            domain={[80, 100]}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: 'white', 
              border: '1px solid #e5e7eb',
              borderRadius: '8px',
              fontSize: '12px'
            }}
            formatter={(value) => [`${value}%`, '']}
          />
          <Legend />
          <Bar dataKey="准确率" fill="#3b82f6" radius={[2, 2, 0, 0]} />
          <Bar dataKey="精确率" fill="#10b981" radius={[2, 2, 0, 0]} />
          <Bar dataKey="召回率" fill="#8b5cf6" radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}