'use client';

import { useState } from 'react';
import Link from 'next/link';
import PatientList from './PatientList';
import SearchFilters from './SearchFilters';

export default function PatientSearch() {
  const [searchQuery, setSearchQuery] = useState('');
  const [filters, setFilters] = useState({
    gender: '',
    ageRange: '',
    thermal24: '',
    thermal25: '',
    voice25: '',
    icas: '',
    icasGrade: '',
    predictedIcas: '',
    predictedIcasGrade: '',
    bmiRange: ''
  });
  const [showAdvanced, setShowAdvanced] = useState(false);



  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    console.log('搜索患者:', { searchQuery, filters });
  };

  const clearFilters = () => {
    setSearchQuery('');
    setFilters({
      gender: '',
      ageRange: '',
      thermal24: '',
      thermal25: '',
      voice25: '',
      icas: '',
      icasGrade: '',
      predictedIcas: '',
      predictedIcasGrade: '',
      bmiRange: ''
    });
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
              <Link href="/search" className="text-blue-600 px-3 py-2 rounded-md text-sm font-medium whitespace-nowrap cursor-pointer">
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
          <h1 className="text-3xl font-bold text-gray-900 mb-2">患者检索查询</h1>
          <p className="text-gray-600">快速搜索和筛选患者信息，支持多条件组合查询</p>


        </div>

        {/* 搜索区域 */}
        <div className="bg-white p-6 rounded-xl shadow-lg border mb-8">
          <form onSubmit={handleSearch}>
            {/* 主搜索框 */}
            <div className="flex gap-4 mb-6">
              <div className="flex-1 relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <i className="ri-search-line text-gray-400 text-sm"></i>
                </div>
                <input
                  type="text"
                  placeholder="输入患者编号、姓名进行搜索..."
                  className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
              <button
                type="submit"
                className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 font-medium whitespace-nowrap cursor-pointer"
              >
                搜索
              </button>
              <button
                type="button"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  setShowAdvanced(!showAdvanced);
                }}
                className="bg-gray-100 text-gray-700 px-6 py-3 rounded-lg hover:bg-gray-200 font-medium whitespace-nowrap cursor-pointer"
              >
                <i className={`ri-filter-line mr-2 ${showAdvanced ? 'text-blue-600' : ''}`}></i>
                高级筛选
              </button>
            </div>

            {/* 高级筛选 */}
            {showAdvanced && (
              <SearchFilters 
                filters={filters} 
                setFilters={setFilters}
                onClear={clearFilters}
              />
            )}
          </form>
        </div>

        {/* 患者列表 */}
        <PatientList searchQuery={searchQuery} filters={filters} />
      </div>
    </div>
  );
}