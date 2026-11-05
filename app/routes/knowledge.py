"""
API Routes for the Knowledge Base.
"""
import logging

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

from app.config import settings
from app.models.schemas import (
    DocumentCreate,
    DocumentListResponse,
    DocumentMutationResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceItem,
)
from app.services import vectorstore
from app.services.ingest_service import ingest_service, validate_title
from app.services.rag_service import rag_service

router = APIRouter(
    prefix="/api/v1",
    tags=["knowledge-base"],
)


def _get_vectordb():
    """统一从 vectorstore 单例取库，保证 collection_name 一致。"""
    return vectorstore.get_vectordb()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """检查服务是否运行，并返回向量库块数、集合名与所用 LLM 模型。"""
    db = _get_vectordb()
    return HealthResponse(
        status="healthy",
        llm_model=settings.DEEPSEEK_MODEL,
        collection_count=db._collection.count(),
        collection=settings.COLLECTION_NAME,
        similarity_threshold=settings.SIMILARITY_THRESHOLD,
    )


@router.post("/query", response_model=QueryResponse)
def query_knowledge_base(request: QueryRequest):
    """基于知识库回答问题；返回来源编号/距离分数、生效阈值与是否跳过了大模型。"""
    db = _get_vectordb()
    try:
        # LLM 由 rag_service 懒加载并复用；无相关块时连 LLM 都不会调用
        result = rag_service.ask(request.question, vectordb=db, top_k=request.top_k)
    except RuntimeError as exc:
        # 缺 DEEPSEEK_API_KEY 一类的配置问题，属于服务未就绪
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        # 超时 / 限流 / 网络错误：可读的 502，不把堆栈丢给调用方
        logger.exception("RAG 生成失败: question=%r", request.question)
        raise HTTPException(
            status_code=502,
            detail=f"生成回答失败：{type(exc).__name__}: {exc}",
        ) from exc

    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        sources=[SourceItem(**s) for s in result["sources"]],
        top_k=result["top_k"],
        threshold=result["threshold"],
        skipped_llm=result["skipped_llm"],
    )


# ---------------- 文档管理 ----------------

@router.get("/documents", response_model=DocumentListResponse)
def list_documents():
    """列出知识库里的文档及各自块数。"""
    db = _get_vectordb()
    docs = ingest_service.list_documents(vectordb=db)
    return DocumentListResponse(
        documents=docs,
        total_chunks=sum(d["chunks"] for d in docs),
        collection=settings.COLLECTION_NAME,
    )


@router.post("/documents", response_model=DocumentMutationResponse)
def upsert_document(payload: DocumentCreate):
    """新增或覆盖一篇文档（写入 data 目录 + 替换该 title 的向量块）。"""
    db = _get_vectordb()
    try:
        result = ingest_service.add_document(payload.title, payload.content, vectordb=db)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        logger.exception("文档写入磁盘失败: title=%r", payload.title)
        raise HTTPException(status_code=500, detail=f"文件写入失败：{exc}") from exc
    except Exception as exc:
        logger.exception("文档入库失败: title=%r", payload.title)
        raise HTTPException(status_code=500, detail=f"写入失败：{type(exc).__name__}: {exc}") from exc

    return DocumentMutationResponse(
        action="upsert",
        title=result["title"],
        total_chunks=db._collection.count(),
        detail=f"已写入 {result['chunks']} 个块，源文件 {result['path']}",
    )


@router.delete("/documents/{title}", response_model=DocumentMutationResponse)
def delete_document(title: str):
    """删除一篇文档：清向量块 + 删源文件。"""
    try:
        safe_title = validate_title(title)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    db = _get_vectordb()
    result = ingest_service.remove_document(safe_title, vectordb=db)
    return DocumentMutationResponse(
        action="delete",
        title=safe_title,
        total_chunks=db._collection.count(),
        detail=(f"删除 {result['deleted_chunks']} 个块；"
                + ("源文件已删除" if result["file_removed"] else "源文件不存在")),
    )
