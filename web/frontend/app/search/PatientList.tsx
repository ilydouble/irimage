'use client';

import { useState, useEffect } from 'react';
import { apiClient, Patient } from '../../lib/api';

interface PatientListProps {
  searchQuery: string;
  filters: any;
}



export default function PatientList({ searchQuery, filters }: PatientListProps) {

  const [selectedPatients, setSelectedPatients] = useState<string[]>([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [showPatientDetail, setShowPatientDetail] = useState<string | null>(null);
  const [patients, setPatients] = useState<Patient[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pagination, setPagination] = useState({
    page: 1,
    limit: 10,
    total: 0,
    pages: 0
  });
  const [sortBy, setSortBy] = useState<string>('created_at');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  const itemsPerPage = 10;



  // 获取患者数据
  const fetchPatients = async () => {
    try {
      setLoading(true);
      setError(null);



      // 构建查询参数
      const params: any = {
        page: currentPage,
        limit: itemsPerPage
      };

      if (searchQuery.trim()) {
        params.search = searchQuery.trim();
      }

      if (filters.gender) {
        params.gender = filters.gender === '男' ? 'male' : 'female';
      }

      if (filters.icas) {
        params.has_icas = filters.icas === '是';
      }

      if (filters.icasGrade && filters.icasGrade !== '全部') {
        params.icas_grade = filters.icasGrade;
      }

      // 处理检查项目筛选
      if (filters.thermal24) {
        params.thermal_24h = filters.thermal24 === 'yes';
      }

      if (filters.thermal25) {
        params.thermal_25h = filters.thermal25 === 'yes';
      }

      if (filters.voice25) {
        params.voice_25h = filters.voice25 === 'yes';
      }

      // 处理BMI范围筛选
      if (filters.bmiRange) {
        switch (filters.bmiRange) {
          case 'underweight':
            params.bmi_max = 18.4;
            break;
          case 'normal':
            params.bmi_min = 18.5;
            params.bmi_max = 24.9;
            break;
          case 'overweight':
            params.bmi_min = 25;
            params.bmi_max = 29.9;
            break;
          case 'obese':
            params.bmi_min = 30;
            break;
        }
      }

      // 处理年龄范围
      if (filters.ageRange) {
        const ageRanges: { [key: string]: { min?: number; max?: number } } = {
          '30以下': { max: 29 },
          '30-39': { min: 30, max: 39 },
          '40-49': { min: 40, max: 49 },
          '50-59': { min: 50, max: 59 },
          '60以上': { min: 60 }
        };

        const range = ageRanges[filters.ageRange];
        if (range) {
          if (range.min !== undefined) params.age_min = range.min;
          if (range.max !== undefined) params.age_max = range.max;
        }
      }

      // 添加排序参数
      params.sortBy = sortBy;
      params.sortOrder = sortOrder;

      const response = await apiClient.getPatients(params);
      setPatients(response.patients);
      setPagination(response.pagination);
    } catch (err) {
      setError(err instanceof Error ? err.message : '获取患者数据失败');
      console.error('Error fetching patients:', err);
    } finally {
      setLoading(false);
    }
  };

  // 当搜索条件、页码或排序变化时重新获取数据
  useEffect(() => {
    fetchPatients();
  }, [searchQuery, filters, currentPage, sortBy, sortOrder]);

  // 重置页码当搜索条件变化时
  useEffect(() => {
    setCurrentPage(1);
  }, [searchQuery, filters]);

  // 处理排序
  const handleSort = (field: string) => {
    if (sortBy === field) {
      // 如果点击的是当前排序字段，切换排序方向
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      // 如果点击的是新字段，设置为该字段并默认降序
      setSortBy(field);
      setSortOrder('desc');
    }
    setCurrentPage(1); // 重置到第一页
  };

  // 获取排序图标
  const getSortIcon = (field: string) => {
    if (sortBy !== field) {
      return (
        <span className="text-gray-400 ml-1">
          <svg className="w-3 h-3 inline" fill="currentColor" viewBox="0 0 20 20">
            <path d="M5 12l5-5 5 5H5z"/>
          </svg>
        </span>
      );
    }
    return (
      <span className="text-blue-600 ml-1">
        {sortOrder === 'asc' ? (
          <svg className="w-3 h-3 inline" fill="currentColor" viewBox="0 0 20 20">
            <path d="M5 12l5-5 5 5H5z"/>
          </svg>
        ) : (
          <svg className="w-3 h-3 inline" fill="currentColor" viewBox="0 0 20 20">
            <path d="M15 8l-5 5-5-5h10z"/>
          </svg>
        )}
      </span>
    );
  };

  const handleSelectAll = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.checked) {
      setSelectedPatients(patients.map(p => p.patient_id));
    } else {
      setSelectedPatients([]);
    }
  };

  const handleSelectPatient = (patientId: string) => {
    if (selectedPatients.includes(patientId)) {
      setSelectedPatients(selectedPatients.filter(id => id !== patientId));
    } else {
      setSelectedPatients([...selectedPatients, patientId]);
    }
  };

  const getBMI = (weight?: number, height?: number) => {
    if (!weight || !height) return '-';
    const heightInM = height / 100;
    return (weight / (heightInM * heightInM)).toFixed(1);
  };

  const getStatusBadge = (hasData: boolean) => {
    return hasData ? (
      <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-green-100 text-green-800">
        已完成
      </span>
    ) : (
      <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-gray-100 text-gray-800">
        未完成
      </span>
    );
  };

  const getICASBadge = (hasICAS: boolean, grade: string) => {
    if (!hasICAS || grade === '无') {
      return <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-green-100 text-green-800">正常</span>;
    }

    const colorMap: { [key: string]: string } = {
      '轻度': 'bg-yellow-100 text-yellow-800',
      '中度': 'bg-orange-100 text-orange-800',
      '重度': 'bg-red-100 text-red-800'
    };

    return (
      <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${colorMap[grade] || 'bg-gray-100 text-gray-800'}`}>
        {grade}
      </span>
    );
  };

  if (loading) {
    return (
      <div className="bg-white rounded-xl shadow-lg border p-8 text-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
        <p className="mt-4 text-gray-600">加载中...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-xl shadow-lg border p-8 text-center">
        <div className="text-red-600 mb-4">
          <i className="ri-error-warning-line text-2xl"></i>
        </div>
        <p className="text-red-600 mb-4">{error}</p>
        <button
          onClick={fetchPatients}
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
        >
          重试
        </button>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow-lg border">
      {/* 表格头部操作 */}
      <div className="px-6 py-4 border-b flex justify-between items-center">
        <div className="flex items-center space-x-4">
          <span className="text-sm text-gray-600">
            已选择 {selectedPatients.length} 个患者
          </span>
          {selectedPatients.length > 0 && (
            <div className="flex space-x-2">
              <button className="text-sm bg-blue-50 text-blue-600 px-3 py-1 rounded hover:bg-blue-100 cursor-pointer whitespace-nowrap">
                批量导出
              </button>
              <button className="text-sm bg-gray-50 text-gray-600 px-3 py-1 rounded hover:bg-gray-100 cursor-pointer whitespace-nowrap">
                批量分析
              </button>
            </div>
          )}
        </div>
        <div className="flex items-center space-x-2 text-sm text-gray-500">
          每页显示:
          <select className="pr-8 pl-2 py-1 border border-gray-300 rounded">
            <option>10</option>
            <option>25</option>
            <option>50</option>
          </select>
        </div>
      </div>

      {/* 表格 */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left">
                <input
                  type="checkbox"
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  onChange={handleSelectAll}
                  checked={selectedPatients.length === patients.length && patients.length > 0}
                />
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100 select-none"
                onClick={() => handleSort('patient_id')}
              >
                <div className="flex items-center">
                  编号
                  {getSortIcon('patient_id')}
                </div>
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100 select-none"
                onClick={() => handleSort('name')}
              >
                <div className="flex items-center">
                  姓名
                  {getSortIcon('name')}
                </div>
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100 select-none"
                onClick={() => handleSort('gender')}
              >
                <div className="flex items-center">
                  性别
                  {getSortIcon('gender')}
                </div>
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100 select-none"
                onClick={() => handleSort('age')}
              >
                <div className="flex items-center">
                  年龄
                  {getSortIcon('age')}
                </div>
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100 select-none"
                onClick={() => handleSort('bmi')}
              >
                <div className="flex items-center">
                  BMI
                  {getSortIcon('bmi')}
                </div>
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">24年热成像</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">25年热成像</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">25年语音</th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100 select-none"
                onClick={() => handleSort('has_icas')}
              >
                <div className="flex items-center">
                  真实ICAS
                  {getSortIcon('has_icas')}
                </div>
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">预测ICAS</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">操作</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {patients.map((patient) => (
              <tr key={patient.patient_id} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap">
                  <input
                    type="checkbox"
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    checked={selectedPatients.includes(patient.patient_id)}
                    onChange={() => handleSelectPatient(patient.patient_id)}
                  />
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                  {patient.patient_id}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {patient.name}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {patient.gender === 'male' ? '男' : '女'}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {patient.age}岁
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {getBMI(patient.weight, patient.height)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  {getStatusBadge(patient.thermal_24h)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  {getStatusBadge(patient.thermal_25h)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  {getStatusBadge(patient.voice_25h)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  {getICASBadge(patient.has_icas, patient.icas_grade)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex flex-col space-y-1">
                    {patient.predictions && patient.predictions.length > 0 ? (
                      <>
                        {getICASBadge(patient.predictions[0].predicted_icas, patient.predictions[0].predicted_icas_grade)}
                        {patient.predictions[0].predicted_icas_confidence && (
                          <span className="text-xs text-gray-500">
                            置信度: {patient.predictions[0].predicted_icas_confidence}%
                          </span>
                        )}
                      </>
                    ) : (
                      <span className="text-xs text-gray-400">暂无预测</span>
                    )}
                  </div>
                </td>

                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                  <button
                    onClick={() => setShowPatientDetail(patient.patient_id)}
                    className="text-blue-600 hover:text-blue-900 mr-3 cursor-pointer whitespace-nowrap"
                  >
                    详情
                  </button>
                  <button className="text-gray-600 hover:text-gray-900 cursor-pointer whitespace-nowrap">
                    编辑
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* 分页 */}
      <div className="px-6 py-4 border-t flex items-center justify-between">
        <div className="text-sm text-gray-700">
          <div>显示第 {((pagination.page - 1) * pagination.limit) + 1}-{Math.min(pagination.page * pagination.limit, pagination.total)} 条，共 {pagination.total} 条记录</div>
          <div className="text-xs text-gray-500 mt-1">
            按 {sortBy === 'patient_id' ? '编号' :
                sortBy === 'name' ? '姓名' :
                sortBy === 'gender' ? '性别' :
                sortBy === 'age' ? '年龄' :
                sortBy === 'bmi' ? 'BMI' :
                sortBy === 'has_icas' ? '真实ICAS' :
                '创建时间'} {sortOrder === 'asc' ? '升序' : '降序'}排列
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
            disabled={currentPage === 1}
            className="px-3 py-1 text-sm border border-gray-300 rounded hover:bg-gray-50 cursor-pointer whitespace-nowrap disabled:opacity-50 disabled:cursor-not-allowed"
          >
            上一页
          </button>

          {/* 页码按钮 */}
          {Array.from({ length: Math.min(5, pagination.pages) }, (_, i) => {
            const pageNum = Math.max(1, Math.min(pagination.pages - 4, currentPage - 2)) + i;
            if (pageNum > pagination.pages) return null;

            return (
              <button
                key={pageNum}
                onClick={() => setCurrentPage(pageNum)}
                className={`px-3 py-1 text-sm border rounded cursor-pointer whitespace-nowrap ${
                  pageNum === currentPage
                    ? 'bg-blue-600 text-white border-blue-600'
                    : 'border-gray-300 hover:bg-gray-50'
                }`}
              >
                {pageNum}
              </button>
            );
          })}

          {pagination.pages > 5 && currentPage < pagination.pages - 2 && (
            <>
              <span className="px-3 py-1 text-sm text-gray-500">...</span>
              <button
                onClick={() => setCurrentPage(pagination.pages)}
                className="px-3 py-1 text-sm border border-gray-300 rounded hover:bg-gray-50 cursor-pointer whitespace-nowrap"
              >
                {pagination.pages}
              </button>
            </>
          )}

          <button
            onClick={() => setCurrentPage(Math.min(pagination.pages, currentPage + 1))}
            disabled={currentPage === pagination.pages}
            className="px-3 py-1 text-sm border border-gray-300 rounded hover:bg-gray-50 cursor-pointer whitespace-nowrap disabled:opacity-50 disabled:cursor-not-allowed"
          >
            下一页
          </button>
        </div>
      </div>

      {/* 患者详情弹窗 */}
      {showPatientDetail && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-gray-900">患者详细信息</h3>
              <button
                onClick={() => setShowPatientDetail(null)}
                className="w-8 h-8 flex items-center justify-center text-gray-400 hover:text-gray-600 cursor-pointer"
              >
                <i className="ri-close-line text-xl"></i>
              </button>
            </div>
            {(() => {
              const patient = patients.find(p => p.patient_id === showPatientDetail);
              if (!patient) return null;

              return (
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">患者编号</label>
                    <p className="text-sm text-gray-900 mt-1">{patient.patient_id}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">姓名</label>
                    <p className="text-sm text-gray-900 mt-1">{patient.name}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">性别</label>
                    <p className="text-sm text-gray-900 mt-1">{patient.gender === 'male' ? '男' : '女'}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">年龄</label>
                    <p className="text-sm text-gray-900 mt-1">{patient.age}岁</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">身高</label>
                    <p className="text-sm text-gray-900 mt-1">{patient.height ? `${patient.height}cm` : '-'}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">体重</label>
                    <p className="text-sm text-gray-900 mt-1">{patient.weight ? `${patient.weight}kg` : '-'}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">腰围</label>
                    <p className="text-sm text-gray-900 mt-1">{patient.waist ? `${patient.waist}cm` : '-'}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">臀围</label>
                    <p className="text-sm text-gray-900 mt-1">{patient.hip ? `${patient.hip}cm` : '-'}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">颈围</label>
                    <p className="text-sm text-gray-900 mt-1">{patient.neck ? `${patient.neck}cm` : '-'}</p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700">BMI</label>
                    <p className="text-sm text-gray-900 mt-1">{getBMI(patient.weight, patient.height)}</p>
                  </div>
                  <div className="col-span-2">
                    <label className="block text-sm font-medium text-gray-700 mb-2">检查完成情况</label>
                    <div className="grid grid-cols-3 gap-4">
                      <div className="text-center">
                        <p className="text-xs text-gray-600 mb-1">24年热成像</p>
                        {getStatusBadge(patient.thermal_24h)}
                      </div>
                      <div className="text-center">
                        <p className="text-xs text-gray-600 mb-1">25年热成像</p>
                        {getStatusBadge(patient.thermal_25h)}
                      </div>
                      <div className="text-center">
                        <p className="text-xs text-gray-600 mb-1">25年语音</p>
                        {getStatusBadge(patient.voice_25h)}
                      </div>
                    </div>
                  </div>
                  <div className="col-span-2">
                    <label className="block text-sm font-medium text-gray-700 mb-2">ICAS诊断</label>
                    <div className="flex items-center space-x-4">
                      <span className="text-sm text-gray-600">状态:</span>
                      {getICASBadge(patient.has_icas, patient.icas_grade)}
                    </div>
                  </div>
                </div>
              );
            })()}
          </div>
        </div>
      )}
    </div>
  );
}