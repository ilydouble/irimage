'use client';

interface SearchFiltersProps {
  filters: {
    gender: string;
    ageRange: string;
    thermal24: string;
    thermal25: string;
    voice25: string;
    icas: string;
    icasGrade: string;
    predictedIcas: string;
    predictedIcasGrade: string;
    bmiRange: string;
  };
  setFilters: (filters: any) => void;
  onClear: () => void;
}

export default function SearchFilters({ filters, setFilters, onClear }: SearchFiltersProps) {
  const handleFilterChange = (key: string, value: string) => {
    setFilters({ ...filters, [key]: value });
  };

  return (
    <div className="border-t pt-6">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-medium text-gray-900">高级筛选条件</h3>
        <button
          type="button"
          onClick={onClear}
          className="text-sm text-gray-500 hover:text-gray-700 cursor-pointer"
        >
          <i className="ri-refresh-line mr-1"></i>
          清空筛选
        </button>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {/* 性别 */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">性别</label>
          <div className="relative">
            <select
              className="w-full pr-8 pl-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 text-sm"
              value={filters.gender}
              onChange={(e) => handleFilterChange('gender', e.target.value)}
            >
              <option value="">全部</option>
              <option value="男">男</option>
              <option value="女">女</option>
            </select>
          </div>
        </div>

        {/* 年龄范围 */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">年龄范围</label>
          <div className="relative">
            <select
              className="w-full pr-8 pl-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 text-sm"
              value={filters.ageRange}
              onChange={(e) => handleFilterChange('ageRange', e.target.value)}
            >
              <option value="">全部年龄</option>
              <option value="30以下">30以下</option>
              <option value="30-39">30-39岁</option>
              <option value="40-49">40-49岁</option>
              <option value="50-59">50-59岁</option>
              <option value="60以上">60以上</option>
            </select>
          </div>
        </div>

        {/* 24年热成像 */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">24年热成像</label>
          <div className="relative">
            <select
              className="w-full pr-8 pl-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 text-sm"
              value={filters.thermal24}
              onChange={(e) => handleFilterChange('thermal24', e.target.value)}
            >
              <option value="">全部</option>
              <option value="yes">已完成</option>
              <option value="no">未完成</option>
            </select>
          </div>
        </div>

        {/* 25年热成像 */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">25年热成像</label>
          <div className="relative">
            <select
              className="w-full pr-8 pl-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 text-sm"
              value={filters.thermal25}
              onChange={(e) => handleFilterChange('thermal25', e.target.value)}
            >
              <option value="">全部</option>
              <option value="yes">已完成</option>
              <option value="no">未完成</option>
            </select>
          </div>
        </div>

        {/* 25年语音 */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">25年语音检查</label>
          <div className="relative">
            <select
              className="w-full pr-8 pl-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 text-sm"
              value={filters.voice25}
              onChange={(e) => handleFilterChange('voice25', e.target.value)}
            >
              <option value="">全部</option>
              <option value="yes">已完成</option>
              <option value="no">未完成</option>
            </select>
          </div>
        </div>

        {/* 真实ICAS */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">真实ICAS状态</label>
          <div className="relative">
            <select
              className="w-full pr-8 pl-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 text-sm"
              value={filters.icas}
              onChange={(e) => handleFilterChange('icas', e.target.value)}
            >
              <option value="">全部</option>
              <option value="是">有ICAS</option>
              <option value="否">无ICAS</option>
            </select>
          </div>
        </div>

        {/* 真实ICAS分级 */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">真实ICAS分级</label>
          <div className="relative">
            <select
              className="w-full pr-8 pl-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 text-sm"
              value={filters.icasGrade}
              onChange={(e) => handleFilterChange('icasGrade', e.target.value)}
            >
              <option value="">全部分级</option>
              <option value="轻度">轻度</option>
              <option value="中度">中度</option>
              <option value="重度">重度</option>
            </select>
          </div>
        </div>

        {/* 预测ICAS */}
        {/* <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">预测ICAS状态</label>
          <div className="relative">
            <select
              className="w-full pr-8 pl-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 text-sm"
              value={filters.predictedIcas}
              onChange={(e) => handleFilterChange('predictedIcas', e.target.value)}
            >
              <option value="">全部</option>
              <option value="是">有ICAS</option>
              <option value="否">无ICAS</option>
            </select>
          </div>
        </div> */}

        {/* 预测ICAS分级
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">预测ICAS分级</label>
          <div className="relative">
            <select
              className="w-full pr-8 pl-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 text-sm"
              value={filters.predictedIcasGrade}
              onChange={(e) => handleFilterChange('predictedIcasGrade', e.target.value)}
            >
              <option value="">全部分级</option>
              <option value="轻度">轻度</option>
              <option value="中度">中度</option>
              <option value="重度">重度</option>
            </select>
          </div>
        </div> */}

        {/* BMI范围 */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">BMI范围</label>
          <div className="relative">
            <select
              className="w-full pr-8 pl-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 text-sm"
              value={filters.bmiRange}
              onChange={(e) => handleFilterChange('bmiRange', e.target.value)}
            >
              <option value="">全部BMI</option>
              <option value="underweight">偏瘦(&lt;18.5)</option>
              <option value="normal">正常(18.5-24.9)</option>
              <option value="overweight">超重(25-29.9)</option>
              <option value="obese">肥胖(≥30)</option>
            </select>
          </div>
        </div>
      </div>
    </div>
  );
}