# 🧠 RAG 知识库

可运行项目。用 **LangChain** 编排、**DeepSeek** 提供大模型，实现检索增强生成（RAG），目录结构采用标准 FastAPI 分层包格式。

## 🏗️ 架构

```text
提问 → FastAPI / Streamlit
   ↓
Embedding Service（HuggingFace bge-small-zh-v1.5）→ 文本转向量
   ↓
Chroma 向量库 → 召回相似文档块
   ↓
LLM（DeepSeek Chat）→ 读取上下文块并生成回答
   ↓
返回 回答 + 来源上下文

🛠️ 技术栈
后端：FastAPI, Uvicorn, Pydantic
向量库：Chroma（持久化到本地 ./chroma_db）
Embedding：HuggingFace bge-small-zh-v1.5（中文友好，512 维）
LLM：DeepSeek `deepseek-flash`（DeepSeek-V4.1-Flash，langchain-deepseek 的 ChatDeepSeek，默认关闭思考模式）
前端：Streamlit
```

## 📁 文件结构

```text
rag-knowledge-base-langchain-deepseek/
├── README.md                 # 本说明
├── requirements.txt          # 依赖（langchain 拆分包 + deepseek）
├── .env.example              # 密钥模板
├── .gitignore
├── ingest.py                 # 入库脚本：data/sample_documents → Chroma
├── streamlit_app.py          # 前端问答界面
├── app/                      # 应用包（分层结构）
│   ├── __init__.py
│   ├── config.py             # Settings 类：模型名 / 路径 / chunk / top_k
│   ├── main.py               # FastAPI 入口（lifespan + CORS + 路由）
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py        # Pydantic 请求/响应模型
│   ├── routes/
│   │   ├── __init__.py
│   │   └── knowledge.py      # APIRouter：health / query / documents 增删查
│   └── services/
│       ├── __init__.py
│       ├── vectorstore.py    # Embedding / Chroma 唯一构造入口（统一 collection_name）
│       ├── ingest_service.py # 加载 + 切块 + 全量同步入库 + 文档增删改查
│       └── rag_service.py    # ChatDeepSeek + LCEL（top_k 可指定、单次检索）
├── data/
│   └── sample_documents/     # 样例文档（*.txt）
└── tests/
    ├── __init__.py
    └── test_rag.py           # pytest：真实 Embedding 链路 + Fake 库覆盖同步 / top_k / API
```

## 🚀 本地运行

```bash
# 0.（可选）建虚拟环境
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置密钥
cp .env.example .env
#   在 .env 里填写： DEEPSEEK_API_KEY=sk-你的真实密钥

# 3. 入库（加载 data/sample_documents/*.txt → 切块 → 向量化 → 全量同步写 Chroma）
python ingest.py

# 4. 启动服务
uvicorn app.main:app --reload --port 8000   # API，浏览器开 http://localhost:8000/docs
streamlit run streamlit_app.py              # 前端，默认 8501 端口
```

跑通标志：`python -c "import langchain_deepseek, langchain_chroma, langchain_huggingface; print('ok')"` 不报错。

## 📡 API 接口（FastAPI）

服务启动后，交互文档见 http://localhost:8000/docs

| 方法 | 接口 | 说明 |
|------|------|------|
| GET  | `/api/v1/health`    | 健康检查，返回块数、集合名与所用 LLM 模型 |
| POST | `/api/v1/query`     | 提问并生成回答；`top_k` 生效，响应带 `top_k` 与检索来源 |
| GET  | `/api/v1/documents` | 列出知识库文档、各自块数、源文件是否存在 |
| POST | `/api/v1/documents` | 传 `{title, content}` 新增/覆盖一篇文档（写磁盘 + 替换该文档向量块） |
| DELETE | `/api/v1/documents/{title}` | 删除一篇文档：清向量块 + 删源文件 |

Streamlit 侧栏同样提供文档列表、上传、删除与「全量同步 data 目录」按钮。

## 🧪 测试

```bash
pytest -q
```

- 入库 → 检索：用真实 Embedding 模型
- 整条 RAG 链：用 `GenericFakeChatModel` 验证「检索 → 拼 prompt → 生成」不崩

## 🔑 两个关键认知

1. **Embedding 一致性铁律**：入库与问答必须用同一个 Embedding 模型，否则向量不在同一空间、检索全空。
2. **切块数对不上正常**：`RecursiveCharacterTextSplitter` 比手写 `\n\n+长度` 更聪明，块数略少——这是特性不是 bug。

## 🛡️ 检索质量 + 异常收口

| 项 | 做法 |
|---|---|
| 相似度阈值 | `similarity_search_with_score` 取 L2 距离（**越小越相关**），`SIMILARITY_THRESHOLD=1.15` 过滤，阈值可按所用模型与语料标定；设 0 关闭过滤 |
| 不幻觉 | 无块通过阈值时直接返回「知识库中没有找到相关内容」，**不调用大模型**，避免基于无关上下文生成答案 |
| 可追溯 | 回答与检索到的来源片段同步返回，答案依据可核对 |
| LLM 容错 | `request_timeout=30`、`max_retries=2` 配置化。注意字段名为 `request_timeout`；传 `timeout` 会被 pydantic 按 `extra=ignore` 忽略 |
| 复用与兜底 | LLM 懒加载单例复用；全局异常处理器记日志并返回结构化 500；错误分流 503（缺 key）/ 502（超时限流）/ 422（契约） |

阈值说明：增大阈值 → 召回更宽松（可能混入无关块）；减小阈值 → 更严格（可能挡掉相关内容）。
