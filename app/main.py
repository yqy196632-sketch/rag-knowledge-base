"""
FastAPI Application Entry Point.

FastAPI 生命周期事件：
- startup：服务启动时运行一次（构建 embedding + 加载 Chroma 向量库）
- shutdown：服务停止时运行（清理）
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_swagger_ui_oauth2_redirect_html
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.WARNING,  # 压住 sentence_transformers / httpx 的 INFO 噪音
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logging.getLogger("app").setLevel(logging.INFO)
logger = logging.getLogger("rag")
logger.setLevel(logging.INFO)

from app.config import settings
from app.routes.knowledge import router as knowledge_router
from app.services import vectorstore
from app.services.rag_service import rag_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时构建 embedding 与向量库（只做一次）。"""
    print("=" * 50)
    print("Starting RAG Knowledge Base API")
    print("=" * 50)

    print(f"\n[1/2] Loading embedding model ({settings.EMBEDDING_MODEL})...")
    emb = vectorstore.build_embeddings()

    print(f"[2/2] Connecting to ChromaDB (collection={settings.COLLECTION_NAME})...")
    # 统一入口：collection_name 只在这里定义，入库端与问答端必然一致
    vectordb = vectorstore.set_vectordb(vectorstore.create_chroma(emb))
    rag_service.vectordb = vectordb

    if vectordb._collection.count() == 0:
        print("[warn] 向量库为空，请先运行 python ingest.py 或在 /api/v1/documents 上传文档")

    print("\n" + "=" * 50)
    print("API is ready! Visit http://localhost:8000/docs")
    print("=" * 50 + "\n")

    yield  # Application runs here

    print("\nShutting down...")


# Create FastAPI app
app = FastAPI(
    title="RAG Knowledge Base API",
    description=(
        "基于 LangChain 编排、DeepSeek 提供大模型的检索增强生成（RAG）知识库。\n\n"
        "## 功能\n"
        "- 文档自动切块、向量化并存入 Chroma（全量同步，不残留旧块）\n"
        "- 使用 bge-small-zh-v1.5 中文 Embedding\n"
        "- POST /api/v1/query 问答接口，top_k 可指定\n"
        "- GET / POST / DELETE /api/v1/documents 文档管理接口\n\n"
        "## 使用流程\n"
        "1. 运行 `python ingest.py` 入库（需先有 data/sample_documents/*.txt）\n"
        "2. 启动服务 `uvicorn app.main:app --reload`\n"
        "3. 调用 POST /api/v1/query 提问\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,  # 关闭默认 /docs（其静态资源依赖被墙的 jsdelivr），改用下方自定义国内 CDN 版
)

# Add CORS middleware（允许前端调用 API）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请改为你的前端地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(knowledge_router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """兜底：任何未捕获异常都记日志并返回结构化 500，不把堆栈暴露给调用方。"""
    logger.exception("未处理异常: %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "服务内部错误，请查看服务端日志",
            "error_type": type(exc).__name__,
        },
    )


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "RAG Knowledge Base API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


# 自定义 /docs：Swagger UI 的 JS/CSS 改走国内 BootCDN，避免默认 jsdelivr 在大陆白屏
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        swagger_js_url="https://cdn.bootcdn.net/ajax/libs/swagger-ui/5.17.14/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.bootcdn.net/ajax/libs/swagger-ui/5.17.14/swagger-ui.css",
        oauth2_redirect_url="/docs/oauth2-redirect",
    )


# OAuth2 重定向页（Swagger UI 自带、自包含 HTML，不依赖外部 CDN）
@app.get("/docs/oauth2-redirect", include_in_schema=False)
async def custom_swagger_ui_oauth2_redirect():
    return get_swagger_ui_oauth2_redirect_html()
