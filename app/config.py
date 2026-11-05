"""
Configuration settings for the RAG application.
集中管理 Embedding 模型、向量库、LLM、切分与检索参数。

入库与问答必须使用同一个 EMBEDDING_MODEL，否则向量不在同一空间、检索全空。
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    # ---------- Embedding（中文优先） ----------
    # bge-small-zh-v1.5：384 维、约 90MB，中文检索质量明显强于 all-MiniLM-L6-v2
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

    # ---------- 向量库 ----------
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "rag_knowledge_base")

    # ---------- LLM（DeepSeek） ----------
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")

    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    # 思考模式开关：disabled = 直接作答。RAG 是「读上下文后回答」，长思考收益小、
    # 延迟与 token 成本却明显。
    DEEPSEEK_THINKING: str = os.getenv("DEEPSEEK_THINKING", "disabled")

    # ---------- 切分参数 ----------
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))       # 每块约 500 字符
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))  # 相邻块重叠 50，防关键信息被切断

    # ---------- 检索数量与质量 ----------
    TOP_K: int = int(os.getenv("TOP_K", "3"))

    # Chroma 默认返回 L2 距离（越小越相关，归一化向量下范围 0~2）。
    # 设 0 或负数表示关闭阈值过滤。
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "1.15"))

    # ---------- LLM 网络容错 ----------
    LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "30"))       # 单次请求超时（秒）
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "2"))    # 失败自动重试次数


# Singleton instance
settings = Settings()
