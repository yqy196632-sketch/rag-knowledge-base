"""
原则：单元测试不真调 DeepSeek。
    - 入库 → 检索：保留真实 Embedding 的端到端用例（验证切分/向量化/Chroma 链路）
    - 其余用例用 FakeEmbeddings + FakeLLM：快、不下载模型、不花钱
    - API 用例用 FastAPI TestClient，向量库落在 tmp_path，不碰项目里的 chroma_db / data
    - 阈值逻辑用「固定分数的假向量库」精确验证，不受随机向量影响
"""
import itertools
import os
from pathlib import Path

# 仅防止启动即因缺少 key 报错；真正调用 LLM 的地方一律用 FakeLLM 替代
os.environ.setdefault("DEEPSEEK_API_KEY", "sk-fake-for-test")

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from app.config import settings
from app.services import vectorstore
from app.services.ingest_service import ingest_service
from app.services.rag_service import PROMPT, NO_CONTEXT_ANSWER, rag_service


class _FakeScoredDB:
    """只为验证阈值逻辑：返回固定的 (Document, L2 距离) 列表。"""

    def __init__(self, pairs):
        self.pairs = pairs

    def similarity_search_with_score(self, query, k=3):
        return self.pairs[:k]


@pytest.fixture(autouse=True)
def _isolate_services():
    """每个用例前后都清掉全局单例与 LLM 缓存，避免测试之间串状态。"""
    vectorstore.reset_vectordb()
    rag_service.clear_llm_cache()
    rag_service.llm = None
    rag_service.vectordb = None
    yield
    vectorstore.reset_vectordb()
    rag_service.clear_llm_cache()
    rag_service.llm = None
    rag_service.vectordb = None


@pytest.fixture
def fake_db(tmp_path, monkeypatch):
    """把配置指向临时目录，并注入一个 FakeEmbeddings 的向量库单例。"""
    monkeypatch.setattr(settings, "CHROMA_PERSIST_DIR", str(tmp_path / "db"))
    monkeypatch.setattr(settings, "SIMILARITY_THRESHOLD", 0.0)  # 通用用例不过滤
    monkeypatch.setattr(ingest_service, "data_dir", str(tmp_path / "docs"))
    (tmp_path / "docs").mkdir()
    monkeypatch.setattr(vectorstore, "build_embeddings", lambda: FakeEmbeddings(size=512))
    return vectorstore.set_vectordb(vectorstore.create_chroma(FakeEmbeddings(size=512)))


# ---------------- 真实 Embedding 链路 ----------------

def test_ingest_and_retrieve(tmp_path):
    from langchain_chroma import Chroma

    chunks = ingest_service.load_and_split("data/sample_documents")
    assert len(chunks) > 0, "切分后应有 chunk"

    emb = ingest_service.build_embeddings()
    db = Chroma.from_documents(chunks, emb, persist_directory=str(tmp_path / "db"))
    hits = db.similarity_search("Python", k=2)
    assert len(hits) == 2, "应检索到 2 条"


def test_rag_chain_runs(tmp_path):
    from langchain_chroma import Chroma

    emb = ingest_service.build_embeddings()
    db = Chroma.from_documents(
        ingest_service.load_and_split("data/sample_documents"),
        emb,
        persist_directory=str(tmp_path / "db"),
    )
    fake = GenericFakeChatModel(messages=iter(["这是测试回答"]))
    chain = rag_service.build_rag_chain(db, fake)
    assert "测试回答" in chain.invoke("随便问")


# ---------------- 全量同步不残留旧块 ----------------

def test_sync_removes_stale_chunks(tmp_path, fake_db, monkeypatch):
    monkeypatch.setattr(settings, "CHUNK_SIZE", 200)
    monkeypatch.setattr(settings, "CHUNK_OVERLAP", 0)
    src = tmp_path / "docs" / "a.txt"

    src.write_text("旧版本内容。" * 40, encoding="utf-8")
    first = ingest_service.load_and_split(str(tmp_path / "docs"))
    ingest_service.build_vectordb(first, sync=True)
    assert fake_db._collection.count() == len(first) > 1

    src.write_text("全新版本内容。" * 40, encoding="utf-8")
    second = ingest_service.load_and_split(str(tmp_path / "docs"))
    ingest_service.build_vectordb(second, sync=True)
    assert fake_db._collection.count() == len(second), "编辑文档后不应留下旧向量块"

    src.unlink()
    ingest_service.build_vectordb([], sync=True)
    assert fake_db._collection.count() == 0, "源文件删除后库中不应残留块"


# ---------------- top_k / 缺 key ----------------

def test_ask_respects_top_k(fake_db):
    chunks = ingest_service.load_and_split("data/sample_documents")
    ingest_service.build_vectordb(chunks, sync=True)

    fake = GenericFakeChatModel(messages=iter(["回答一", "回答二"]))
    r1 = rag_service.ask("Python", vectordb=fake_db, llm=fake, top_k=1)
    r2 = rag_service.ask("Python", vectordb=fake_db, llm=fake, top_k=4)
    assert r1["top_k"] == 1 and len(r1["sources"]) == 1
    assert r2["top_k"] == 4 and len(r2["sources"]) == 4


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.setattr(settings, "DEEPSEEK_API_KEY", "")
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        rag_service.build_llm()


# ---------------- 检索质量：阈值 / 分数 / 引用编号 ----------------

def test_threshold_filters_and_skips_llm(monkeypatch):
    monkeypatch.setattr(settings, "SIMILARITY_THRESHOLD", 1.15)
    good = Document(page_content="相关块", metadata={"title": "a"})
    bad = Document(page_content="无关块", metadata={"title": "b"})
    fake = GenericFakeChatModel(messages=iter(["不该出现的第二次回答"]))

    r = rag_service.ask("问题", vectordb=_FakeScoredDB([(good, 0.7), (bad, 1.4)]),
                        llm=fake, top_k=2)
    assert r["threshold"] == 1.15 and not r["skipped_llm"]
    assert len(r["sources"]) == 1
    assert r["sources"][0]["index"] == 1 and r["sources"][0]["score"] == 0.7
    assert r["answer"]  # 只调用一次 LLM，fake 消息未被耗尽

    # 全部超阈值：不调用 LLM，直接给「不知道」
    r2 = rag_service.ask("无关问题", vectordb=_FakeScoredDB([(bad, 1.4)]),
                         llm=fake, top_k=2)
    assert r2["skipped_llm"] is True and r2["sources"] == []
    assert r2["answer"] == NO_CONTEXT_ANSWER


def test_context_numbering_and_prompt_requires_citation():
    docs = [Document(page_content="第一段", metadata={}),
            Document(page_content="第二段", metadata={})]
    assert rag_service._format_docs(docs) == "[1] 第一段\n\n[2] 第二段"
    text = PROMPT.format(context="[1] xxx", question="q")
    assert "标注所依据的来源" in text and "[1]" in text


# ---------------- 异常收口与 LLM 复用 ----------------

def test_llm_timeout_retries_and_cache(monkeypatch):
    monkeypatch.setattr(settings, "DEEPSEEK_API_KEY", "sk-fake")
    llm = rag_service.build_llm()
    assert llm.request_timeout == settings.LLM_TIMEOUT
    assert llm.max_retries == settings.LLM_MAX_RETRIES
    assert not hasattr(llm, "timeout"), "timeout 不是有效字段，必须用 request_timeout"

    rag_service.clear_llm_cache()
    assert rag_service.get_llm() is rag_service.get_llm(), "LLM 应复用同一实例"


def test_global_exception_handler(fake_db, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("故意抛错")

    monkeypatch.setattr(ingest_service, "list_documents", boom)
    vectorstore.reset_vectordb()
    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/api/v1/documents")
        assert r.status_code == 500
        assert r.json()["error_type"] == "RuntimeError"
        assert "内部错误" in r.json()["detail"]


# ---------------- title 冲突 ----------------

def test_title_unique_for_subdirectories(tmp_path):
    root = tmp_path / "docs"
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_text("根目录内容。" * 30, encoding="utf-8")
    (root / "sub" / "a.txt").write_text("子目录内容。" * 30, encoding="utf-8")
    chunks = ingest_service.load_and_split(str(root))
    assert {c.metadata["title"] for c in chunks} == {"a", "sub__a"}


# ---------------- API 契约与文档管理 ----------------

def test_api_query_topk_and_documents(fake_db, monkeypatch):
    monkeypatch.setattr(
        rag_service, "build_llm",
        lambda: GenericFakeChatModel(messages=itertools.repeat("假回答")),
    )
    vectorstore.reset_vectordb()  # 让 lifespan 成为本用例唯一的客户端

    from app.main import app

    with TestClient(app) as c:
        health = c.get("/api/v1/health")
        assert health.status_code == 200
        assert health.json()["collection"] == settings.COLLECTION_NAME
        assert health.json()["similarity_threshold"] == 0.0

        chunks = ingest_service.load_and_split("data/sample_documents")
        ingest_service.build_vectordb(chunks, sync=True)

        q1 = c.post("/api/v1/query", json={"question": "What is Python?", "top_k": 1})
        q4 = c.post("/api/v1/query", json={"question": "What is Python?", "top_k": 4})
        assert q1.status_code == 200 and q1.json()["top_k"] == 1
        assert len(q1.json()["sources"]) == 1
        assert q4.json()["top_k"] == 4 and len(q4.json()["sources"]) == 4
        assert q4.json()["threshold"] == 0.0
        assert all("score" in s and "index" in s for s in q4.json()["sources"])

        listing = c.get("/api/v1/documents").json()
        assert listing["total_chunks"] == len(chunks)
        assert listing["collection"] == settings.COLLECTION_NAME

        r = c.post("/api/v1/documents", json={
            "title": "api测试文档",
            "content": "这是通过接口写入的测试文档。\n\n" * 10 + "内容足够长以便切出多个块。",
        })
        assert r.status_code == 200 and r.json()["action"] == "upsert"
        assert c.get("/api/v1/documents").json()["total_chunks"] > len(chunks)

        r2 = c.post("/api/v1/documents", json={"title": "api测试文档", "content": "改写了内容。"})
        assert r2.status_code == 200
        assert c.get("/api/v1/documents").json()["total_chunks"] == len(chunks) + 1

        bad = c.post("/api/v1/documents", json={"title": "../evil", "content": "x"})
        assert bad.status_code in (400, 422)

        d = c.delete("/api/v1/documents/api测试文档")
        assert d.status_code == 200 and d.json()["total_chunks"] == len(chunks)
        assert not (Path(ingest_service.data_dir) / "api测试文档.txt").exists()


def test_api_rejects_bad_contract(fake_db):
    vectorstore.reset_vectordb()
    from app.main import app

    with TestClient(app) as c:
        assert c.post("/api/v1/query", json={"question": ""}).status_code == 422
        assert c.post("/api/v1/query", json={"question": "x", "top_k": 11}).status_code == 422
