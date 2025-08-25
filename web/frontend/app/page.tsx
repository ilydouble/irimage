'use client';

import Link from 'next/link';

export default function Home() {
  return (
    <div className="min-h-screen bg-white">
      {/* 顶部导航 */}
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <h1 className="text-xl font-bold text-gray-900 font-['Pacifico']">PatientCare</h1>
            </div>
            <div className="flex items-center space-x-8">
              <Link href="/analytics" className="text-gray-700 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium whitespace-nowrap cursor-pointer">
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

      {/* 主要内容 */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Hero区域 */}
        <div className="text-center mb-16">
          <h1 className="text-4xl font-bold text-gray-900 mb-6">
            患者数据管理系统
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            基于人工智能的医疗数据分析平台，提供患者统计分析、智能检索和疾病预测功能
          </p>
          <div className="flex justify-center space-x-4">
            <Link href="/analytics" className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 font-medium whitespace-nowrap cursor-pointer">
              开始分析
            </Link>
            <Link href="/search" className="bg-gray-100 text-gray-700 px-6 py-3 rounded-lg hover:bg-gray-200 font-medium whitespace-nowrap cursor-pointer">
              搜索患者
            </Link>
          </div>
        </div>

        {/* 功能卡片 */}
        <div className="grid md:grid-cols-3 gap-8 mb-16">
          <div className="bg-white p-8 rounded-xl shadow-lg border">
            <div className="w-12 h-12 flex items-center justify-center bg-blue-100 rounded-lg mb-6">
              <i className="ri-bar-chart-line text-2xl text-blue-600"></i>
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-4">统计分析</h3>
            <p className="text-gray-600 mb-6">
              患者数据统计分析和预测准确率评估，提供详细的数据洞察和可视化报告
            </p>
            <Link href="/analytics" className="text-blue-600 hover:text-blue-700 font-medium whitespace-nowrap cursor-pointer">
              查看详情 →
            </Link>
          </div>

          <div className="bg-white p-8 rounded-xl shadow-lg border">
            <div className="w-12 h-12 flex items-center justify-center bg-green-100 rounded-lg mb-6">
              <i className="ri-search-line text-2xl text-green-600"></i>
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-4">患者检索</h3>
            <p className="text-gray-600 mb-6">
              快速检索和查询患者信息，支持多条件筛选和高级搜索功能
            </p>
            <Link href="/search" className="text-green-600 hover:text-green-700 font-medium whitespace-nowrap cursor-pointer">
              立即搜索 →
            </Link>
          </div>

          <div className="bg-white p-8 rounded-xl shadow-lg border">
            <div className="w-12 h-12 flex items-center justify-center bg-purple-100 rounded-lg mb-6">
              <i className="ri-brain-line text-2xl text-purple-600"></i>
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-4">模型预测</h3>
            <p className="text-gray-600 mb-6">
              基于机器学习模型的疾病预测和风险评估，辅助医疗决策制定
            </p>
            <Link href="/prediction" className="text-purple-600 hover:text-purple-700 font-medium whitespace-nowrap cursor-pointer">
              开始预测 →
            </Link>
          </div>
        </div>

        {/* 数据概览 */}
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-8 rounded-xl mb-16">
          <h2 className="text-2xl font-bold text-gray-900 mb-8 text-center">系统数据概览</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-600 mb-2">1,247</div>
              <div className="text-gray-600">总患者数</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-green-600 mb-2">892</div>
              <div className="text-gray-600">完成检查</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-purple-600 mb-2">94.2%</div>
              <div className="text-gray-600">预测准确率</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-orange-600 mb-2">156</div>
              <div className="text-gray-600">高风险患者</div>
            </div>
          </div>
        </div>

        {/* 系统特性 */}
        <div className="grid md:grid-cols-2 gap-12 items-center mb-16">
          <div>
            <h2 className="text-3xl font-bold text-gray-900 mb-6">智能化医疗数据分析</h2>
            <div className="space-y-4">
              <div className="flex items-start">
                <div className="w-6 h-6 flex items-center justify-center bg-blue-100 rounded-full mt-1 mr-4">
                  <i className="ri-check-line text-sm text-blue-600"></i>
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900">多维度数据分析</h4>
                  <p className="text-gray-600">整合热成像、语音、体征等多种检查数据</p>
                </div>
              </div>
              <div className="flex items-start">
                <div className="w-6 h-6 flex items-center justify-center bg-green-100 rounded-full mt-1 mr-4">
                  <i className="ri-check-line text-sm text-green-600"></i>
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900">实时预测评估</h4>
                  <p className="text-gray-600">基于AI模型的ICAS风险实时评估</p>
                </div>
              </div>
              <div className="flex items-start">
                <div className="w-6 h-6 flex items-center justify-center bg-purple-100 rounded-full mt-1 mr-4">
                  <i className="ri-check-line text-sm text-purple-600"></i>
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900">数据可视化</h4>
                  <p className="text-gray-600">直观的图表展示和统计分析结果</p>
                </div>
              </div>
            </div>
          </div>
          <div className="relative">
            <img 
              src="https://readdy.ai/api/search-image?query=Modern%20medical%20data%20analysis%20dashboard%20with%20charts%2C%20graphs%2C%20and%20patient%20information%20displayed%20on%20computer%20screens%20in%20a%20clean%20healthcare%20environment%2C%20professional%20medical%20technology%20interface%2C%20blue%20and%20white%20color%20scheme%2C%20minimalist%20design&width=600&height=400&seq=1&orientation=landscape"
              alt="医疗数据分析界面"
              className="w-full h-80 object-cover object-top rounded-xl shadow-lg"
            />
          </div>
        </div>
      </main>

      {/* 底部 */}
      <footer className="bg-gray-50 mt-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="text-center">
            <h3 className="text-lg font-semibold text-gray-900 mb-2 font-['Pacifico']">PatientCare</h3>
            <p className="text-gray-600">智能医疗数据管理系统</p>
          </div>
        </div>
      </footer>
    </div>
  );
}