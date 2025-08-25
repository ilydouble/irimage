# IR Image Patient Care System

基于SQLite的患者管理和预测系统，支持热成像和语音数据分析。

## 系统架构

- **后端**: Node.js + Express + SQLite
- **前端**: Next.js + React + TypeScript + Tailwind CSS
- **数据库**: SQLite (简化版设计)
- **预测模型**: Mock接口 (可切换到真实模型)

## 功能特性

### 患者管理
- ✅ 患者信息的增删改查
- ✅ 支持多条件搜索和筛选
- ✅ 分页显示
- ✅ 患者详情查看

### 预测功能
- ✅ Mock预测模型 (基于规则的风险评估)
- ✅ 可切换的预测服务 (Mock/API模式)
- ✅ 批量预测
- ✅ 预测结果管理

### 文件管理
- ✅ 文件上传 (热成像、语音文件)
- ✅ 文件类型验证
- ✅ 文件下载
- ✅ 文件记录管理

### 数据分析
- ✅ 系统概览统计
- ✅ 预测分析
- ✅ 患者分析
- ✅ 文件分析
- ✅ 数据导出

## 快速开始

### 1. 安装依赖

```bash
# 安装后端依赖
npm install

# 安装前端依赖
cd frontend
npm install
cd ..
```

### 2. 初始化数据库

```bash
# 创建数据库表结构
npm run init-db

# 添加示例数据
npm run seed-db
```

### 3. 启动服务

```bash
# 启动后端服务 (端口 3001)
npm start

# 或者开发模式
npm run dev
```

```bash
# 启动前端服务 (端口 3000)
cd frontend
npm run dev
```

### 4. 访问系统

- 前端界面: http://localhost:3000
- 后端API: http://localhost:3001
- API健康检查: http://localhost:3001/api/health

## API 接口

### 患者管理
- `GET /api/patients` - 获取患者列表
- `GET /api/patients/:id` - 获取患者详情
- `POST /api/patients` - 创建患者
- `PUT /api/patients/:id` - 更新患者
- `DELETE /api/patients/:id` - 删除患者

### 预测管理
- `GET /api/predictions` - 获取预测列表
- `POST /api/predictions` - 创建预测
- `POST /api/predictions/batch` - 批量预测
- `GET /api/predictions/service/health` - 预测服务健康检查
- `POST /api/predictions/service/mode` - 切换预测模式

### 文件管理
- `GET /api/files` - 获取文件列表
- `POST /api/files/upload` - 上传文件
- `GET /api/files/download/:id` - 下载文件
- `DELETE /api/files/:id` - 删除文件

### 数据分析
- `GET /api/analytics/overview` - 系统概览
- `GET /api/analytics/predictions` - 预测分析
- `GET /api/analytics/patients` - 患者分析
- `GET /api/analytics/files` - 文件分析

## 数据库管理

### 数据库操作命令

```bash
# 备份数据库
npm run db-manager backup

# 恢复数据库
npm run db-manager restore <backup_file>

# 导出数据
npm run db-manager export [type] [output_file]
# 例: npm run db-manager export patients patients.json

# 导入数据
npm run db-manager import <json_file>

# 重置数据库 (删除所有数据)
npm run db-manager reset --confirm

# 查看数据库状态
npm run db-manager status

# 查看帮助
npm run db-manager help
```

### 数据库结构

#### 患者表 (patients)
- 基本信息: 编号、姓名、性别、年龄
- 体征数据: 身高、体重、BMI、腰围、臀围、颈围
- 检查状态: 24年热成像、25年热成像、25年语音
- ICAS信息: 诊断状态、分级

#### 预测表 (predictions)
- 风险评分、风险等级、置信度
- 影响因子、建议

#### 文件表 (files)
- 文件类型、文件名、文件路径
- 上传时间

## 预测模型

### Mock模式 (默认)
- 基于规则的风险评估算法
- 考虑年龄、BMI、腰围、颈围、ICAS状态等因素
- 返回风险评分 (0-100)、风险等级 (low/medium/high)、置信度

### API模式
- 可对接真实的机器学习模型
- 支持HTTP API调用
- 自动降级到Mock模式 (当API不可用时)

### 切换预测模式

```bash
# 切换到Mock模式
curl -X POST http://localhost:3001/api/predictions/service/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "mock"}'

# 切换到API模式
curl -X POST http://localhost:3001/api/predictions/service/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "api"}'

# 检查服务状态
curl http://localhost:3001/api/predictions/service/health
```

## 环境变量

创建 `.env` 文件配置环境变量:

```env
# 预测服务配置
PREDICTION_MODE=mock
PREDICTION_API_URL=http://localhost:5000/predict
PREDICTION_API_KEY=your_api_key

# 前端API配置
NEXT_PUBLIC_API_URL=http://localhost:3001/api
```

## 文件上传

支持的文件类型:
- **热成像**: .jpg, .jpeg, .png, .bmp, .tiff (最大10MB)
- **语音**: .wav, .mp3, .m4a, .aac, .flac (最大50MB)

文件存储在 `uploads/` 目录下，按类型分类:
- `uploads/thermal_24h/` - 24年热成像
- `uploads/thermal_25h/` - 25年热成像  
- `uploads/voice_25h/` - 25年语音

## 开发说明

### 项目结构
```
├── config/           # 数据库配置
├── routes/           # API路由
├── services/         # 业务服务
├── scripts/          # 数据库脚本
├── uploads/          # 文件上传目录
├── database/         # SQLite数据库文件
├── frontend/         # Next.js前端
│   ├── app/         # 页面组件
│   ├── lib/         # API客户端
│   └── ...
└── ...
```

### 添加新功能
1. 后端: 在 `routes/` 中添加新的路由
2. 前端: 在 `frontend/app/` 中添加新页面
3. API: 在 `frontend/lib/api.ts` 中添加API方法

## 示例数据

系统包含5个示例患者数据:
- 101EC - 张广磊 (男, 42岁, 高风险)
- 045EC - 徐淑燕 (女, 43岁, 中风险)  
- 051FE - 赵波 (男, 43岁, 高风险)
- 067CK - 李明 (男, 38岁, 有ICAS轻度)
- 089FE - 王丽 (女, 35岁, 低风险)

## 故障排除

### 常见问题

1. **数据库文件不存在**
   ```bash
   npm run init-db
   ```

2. **前端无法连接后端**
   - 检查后端服务是否启动 (端口3001)
   - 检查CORS配置

3. **文件上传失败**
   - 检查文件大小和类型
   - 确保uploads目录存在且可写

4. **预测服务不可用**
   - 检查预测服务模式
   - 查看服务健康状态

## 许可证

MIT License
