# BidCheck API - 文档查重API服务

基于深度学习和大模型的智能文档查重系统，支持文档内容的重复检测和相似性分析。

## 🌟 功能特性

- **智能查重**: 基于大模型的语义相似性检测，支持重复内容识别
- **分块处理**: 自动文档分割，支持长文档的细粒度分析
- **聚类分析**: 使用HDBSCAN算法进行文档聚类，提高检测效率
- **并行处理**: 采用LangChain RunnableParallel实现高性能并行分析
- **RESTful API**: 标准化的API接口，易于集成
- **Docker支持**: 容器化部署，支持快速部署和扩展
- **实时监控**: 集成LangSmith链路追踪，提供详细的执行分析

## 🏗️ 项目架构

```text
src/
├── api/                    # API层
│   ├── app.py             # FastAPI应用定义
│   └── service.py         # 核心业务服务
├── core/                  # 核心处理模块
│   ├── document_processor.py    # 文档处理器
│   └── clustering_manager.py    # 聚类管理器
├── detectors/             # 检测器模块
│   └── llm_duplicate_detector.py # 大模型重复检测
├── models/                # 数据模型
│   ├── api_models.py      # API数据模型
│   └── data_models.py     # 内部数据模型
├── validators/            # 验证器模块
│   └── validation_manager.py    # 验证管理器
├── config/               # 配置管理
│   └── config.py         # 系统配置
└── utils/                # 工具类
```

## 🚀 快速开始

### 环境要求

- Python 3.11+
- Docker 20.10+
- Docker Compose 2.0+

### 1. 克隆项目

```bash
git clone https://github.com/zhihh/bidcheck-api.git
cd bidcheck-api
```

### 2. 环境配置

#### 通过 config.py 配置

修改 `src/config/config.py` 文件中的默认配置：

```python

        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
        os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        
        # 模型配置
        os.environ["LLM_MODEL_NAME"] = os.getenv("LLM_MODEL_NAME", "qwen-turbo")
        os.environ["EMBEDDING_MODEL_NAME"] = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4")
```

或手动设置环境变量。

#### 通过`.env`文件配置

创建 `.env` 文件并配置以下环境变量：

```bash
# OpenAI API配置
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 模型配置
LLM_MODEL_NAME=qwen-turbo
EMBEDDING_MODEL_NAME=text-embedding-v4

```

### 3. 部署方式

#### 方式一：Docker Compose（推荐）

```bash
# 快速部署
./scripts/deploy.sh

# 或者手动部署
docker compose up -d
```

#### 方式二：本地开发

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python main.py
```

### 4. 验证部署

访问以下端点验证服务状态：

- 服务状态: <http://localhost:8000/>
- 健康检查: <http://localhost:8000/health>
- API文档: <http://localhost:8000/docs>

## 📖 API使用指南

### 文档查重接口

**端点**: `POST /api/v2/analyze`

**请求格式**:

```json
[
  {
    "documentId": 1,
    "page": 1,
    "content": "文档内容..."
  },
  {
    "documentId": 2,
    "page": 1,
    "content": "另一个文档内容..."
  }
]
```

**响应格式**:

```json
{
  "success": true,
  "message": "分析完成，发现 2 对重复内容",
  "data": [
    {
      "documentId1": 1,
      "page1": 1,
      "chunkId1": 0,
      "content1": "重复内容片段1",
      "documentId2": 2,
      "page2": 1,
      "chunkId2": 0,
      "content2": "重复内容片段2",
      "reason": "内容在语义上高度相似",
      "score": 0.95
    }
  ],
  "total_count": 2,
  "processing_time": 3.45
}
```

### 使用示例

```python
import requests
import json

# 准备测试数据
documents = [
    {
        "documentId": 1,
        "page": 1,
        "content": "人工智能是计算机科学的一个分支..."
    },
    {
        "documentId": 2,
        "page": 1,
        "content": "AI是计算机科学的重要领域..."
    }
]

# 发送请求
response = requests.post(
    "http://localhost:8000/api/v2/analyze",
    json=documents,
    headers={"Content-Type": "application/json"}
)

# 处理响应
result = response.json()
print(f"发现 {result['total_count']} 对重复内容")
```

## 🔧 配置说明

### 模型配置

系统支持多种大模型，通过环境变量配置：

- `LLM_MODEL_NAME`: 用于重复检测的大模型（默认: qwen-turbo）
- `EMBEDDING_MODEL_NAME`: 用于向量化的嵌入模型（默认: text-embedding-v4）

### 性能调优

- **文档分割**: 可在 `document_processor.py` 中调整 `chunk_size` 和 `chunk_overlap`
- **聚类参数**: 可在 `clustering_manager.py` 中调整 HDBSCAN 参数
- **并发控制**: 系统自动管理并发，避免资源冲突

## 🧪 测试

### 运行测试

```bash
# 运行复杂数据测试
python test/test_complex_data.py

# 使用Jupyter进行交互式测试
jupyter notebook notebook/test.ipynb
```

### 测试数据

项目包含多种测试场景：

- 学术论文重复检测
- 技术文档相似性分析
- 多语言内容检测
- 大规模文档处理

## 📊 监控与日志

### LangSmith追踪

系统集成LangSmith链路追踪，提供：

- 详细的执行时间分析
- 模型调用链路可视化
- 性能瓶颈识别

### 日志管理

- 应用日志: `logs/bidcheck-api.log`
- 访问日志: 通过Docker Compose查看
- 错误追踪: 集成到响应中

## 🐳 容器化部署

### 构建镜像

```bash
# 构建生产镜像
docker build -t zhihh/bidcheck-api:latest .

# 推送到仓库（可选）
docker push zhihh/bidcheck-api:latest
```

### 生产部署

```bash
# 生产环境部署
./scripts/deploy.sh --prod

# 开发环境部署
./scripts/deploy.sh --dev
```

## 🔒 安全说明

- API密钥通过环境变量配置，避免硬编码
- 支持CORS配置，可根据需要限制访问源
- 容器化运行，与主机环境隔离
- 支持HTTPS部署（需配置反向代理）

## 📈 性能指标

- **处理速度**: 单文档分析通常在2-5秒内完成
- **并发支持**: 支持多请求并发处理
- **内存使用**: 优化的内存管理，支持大文档处理
- **准确率**: 基于大模型的语义分析，准确率>90%

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

## 🆘 技术支持

如有问题或建议，请通过以下方式联系：

- 创建 Issue
- 发送邮件至项目维护者
- 查看项目文档和示例

## 🔄 版本历史

- **v2.0.0**: 模块化重构，支持并行处理
- **v1.0.0**: 初始版本，基础查重功能

---

⭐ 如果这个项目对您有帮助，请给个Star支持！
