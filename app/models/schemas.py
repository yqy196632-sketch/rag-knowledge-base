"""
Pydantic models for request and response validation.
FastAPI 用这些模型做：1) 入参校验  2) 自动生成文档  3) 序列化响应。
"""
import re
from typing import Dict, Any, List

from pydantic import BaseModel, Field, field_validator


# ============ REQUEST MODELS ============

class QueryRequest(BaseModel):
    """请求体：提问。"""
    question: str = Field(..., min_length=1, description="要问的问题")
    top_k: int = Field(default=3, ge=1, le=10, description="检索返回的上下文块数量")


class DocumentCreate(BaseModel):
    """新增/覆盖一篇文档（title 同时是磁盘文件名，禁止路径穿越）。"""
    title: str = Field(..., min_length=1, max_length=80, description="文档标题，不含扩展名")
    content: str = Field(..., min_length=1, description="文档正文")

    @field_validator("title")
    @classmethod
    def _safe_title(cls, v: str) -> str:
        v = v.strip()
        if ".." in v or "/" in v or "\\" in v or not re.match(
            r"^[\w\u4e00-\u9fff][\w\u4e00-\u9fff.\- ]{0,79}$", v
        ):
            raise ValueError("title 只能包含中文/字母/数字/下划线/连字符/空格/点")
        return v


# ============ RESPONSE MODELS ============

class SourceItem(BaseModel):
    """单条检索来源（index 对应答案里的 [n] 引用，score 为 L2 距离，越小越相关）。"""
    content: str
    metadata: Dict[str, Any]
    index: int = 0
    score: float = 0.0


class QueryResponse(BaseModel):
    """问答响应。"""
    question: str
    answer: str
    sources: List[SourceItem]
    top_k: int = Field(default=3, description="本次实际检索的块数")
    threshold: float = Field(default=0.0, description="本次生效的距离阈值")
    skipped_llm: bool = Field(default=False, description="是否因无相关块而跳过大模型")


class HealthResponse(BaseModel):
    """健康检查响应。"""
    status: str
    llm_model: str
    collection_count: int
    collection: str = Field(default="", description="当前生效的集合名")
    similarity_threshold: float = Field(default=0.0, description="当前生效的距离阈值")


class DocumentItem(BaseModel):
    """知识库里的一篇文档。"""
    title: str
    chunks: int
    file_exists: bool


class DocumentListResponse(BaseModel):
    """文档列表响应。"""
    documents: List[DocumentItem]
    total_chunks: int
    collection: str


class DocumentMutationResponse(BaseModel):
    """新增/删除文档的结果。"""
    action: str
    title: str
    total_chunks: int
    detail: str = ""
