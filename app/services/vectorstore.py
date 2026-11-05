"""
Vector Store - Embedding 与 Chroma 的唯一构造入口。
"""
from typing import Optional

from app.config import settings

# 进程内共享的向量库客户端（由 lifespan 或测试注入）
_client = None


def build_embeddings():
    """构建中文 Embedding；入库与问答必须共用同一模型。"""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},  # 归一化，距离比较才公平
    )


def create_chroma(embeddings=None, persist_directory: Optional[str] = None):
    """创建/连接 Chroma，统一传 collection_name。"""
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=settings.COLLECTION_NAME,
        embedding_function=embeddings or build_embeddings(),
        persist_directory=persist_directory or settings.CHROMA_PERSIST_DIR,
    )


def get_vectordb(embeddings=None):
    """取全局单例；未初始化时按配置懒加载（可复用调用方已构建好的 embedding）。"""
    global _client
    if _client is None:
        _client = create_chroma(embeddings)
    return _client


def set_vectordb(client):
    """由 lifespan / 测试显式注入。"""
    global _client
    _client = client
    return _client


def reset_vectordb():
    """丢弃单例（测试隔离用）。"""
    global _client
    _client = None


def has_client() -> bool:
    return _client is not None


def count() -> int:
    """当前集合的向量块数量（Chroma 私有属性，集中在此便于以后替换）。"""
    return get_vectordb()._collection.count()
