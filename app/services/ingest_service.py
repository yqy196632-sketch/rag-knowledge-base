"""
Ingest Service - 文档加载、切块、向量化、入库与文档管理。
"""
import hashlib
import os
import re
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.services import vectorstore

# title 同时用作磁盘文件名，必须挡住路径穿越
_TITLE_RE = re.compile(r"^[\w\u4e00-\u9fff][\w\u4e00-\u9fff.\- ]{0,79}$")


def validate_title(title: str) -> str:
    """校验文档标题：只允许中文/字母/数字/下划线/连字符/空格/点。"""
    title = (title or "").strip()
    if (
        not title
        or ".." in title
        or "/" in title
        or "\\" in title
        or not _TITLE_RE.match(title)
    ):
        raise ValueError("title 只能包含中文/字母/数字/下划线/连字符/空格/点，长度 1-80，且不能以点开头")
    return title


class IngestService:
    """文档入库服务：加载 → 切块 → 向量化 → 同步 Chroma。"""

    def __init__(self):
        self.data_dir = "data/sample_documents"

    # ---------- 加载与切分 ----------

    def _splitter(self) -> RecursiveCharacterTextSplitter:
        # 中文友好的分隔优先级：段落 > 换行 > 标点 > 空格
        return RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
        )

    @staticmethod
    def _decorate(chunks, title: str = None, data_dir: str = None) -> list:
        """
        给每块补 metadata['title']，供精确删除用。

        title 取「相对 data_dir 的路径去扩展名」并把路径分隔符换成 `__`，
        这样 `sub/a.txt` 与 `a.txt` 不会撞名。
        """
        for c in chunks:
            src = c.metadata.get("source", "")
            if title:
                c.metadata["title"] = title
                continue
            raw = os.path.splitext(os.path.basename(src))[0]
            if data_dir:
                try:
                    rel = os.path.relpath(os.path.normpath(src), os.path.normpath(data_dir))
                    if not rel.startswith(".."):
                        raw = os.path.splitext(rel)[0].replace("\\", "__").replace("/", "__")
                except ValueError:
                    pass  # 跨盘符等无法算相对路径时，回退到 basename
            c.metadata["title"] = raw
        return chunks

    def load_and_split(self, data_dir: str = None) -> list:
        """
        递归加载目录下所有 .txt 并切分为 chunks。
        """
        data_dir = data_dir or self.data_dir
        root = Path(data_dir)
        paths = sorted(root.rglob("*.txt"))
        docs = [
            Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": str(p)})
            for p in paths
        ]
        print(f"[1] 已加载 {len(docs)} 个文档"
              + ("" if root.exists() else f"（目录不存在：{root}）"))

        chunks = self._decorate(self._splitter().split_documents(docs), data_dir=data_dir)
        print(f"[2] 切分为 {len(chunks)} 个块")
        return chunks

    def build_embeddings(self):
        """保留原接口，委托给统一入口。"""
        return vectorstore.build_embeddings()

    # ---------- 向量库读写 ----------

    @staticmethod
    def stable_id(chunk) -> str:
        """基于「标题 + 内容」生成稳定 id —— 同一块每次入库得到同一 id。"""
        key = f"{chunk.metadata.get('title', '')}::{chunk.page_content}"
        return "chk_" + hashlib.md5(key.encode("utf-8")).hexdigest()

    @staticmethod
    def _titles_in_db(db) -> set:
        data = db.get(include=["metadatas"])
        return {
            m.get("title")
            for m in (data.get("metadatas") or [])
            if m and m.get("title")
        }

    @staticmethod
    def _delete_title(db, title: str) -> int:
        before = db._collection.count()
        db.delete(where={"title": title})
        return before - db._collection.count()

    def build_vectordb(self, chunks: list, embeddings=None,
                       persist_directory: str = None, sync: bool = True):
        """
        把 chunks 写入持久化 Chroma。

        sync=True（默认，供 ingest.py 全量入库）：以本次 chunks 为快照，
            先删掉库里「已不存在的文件」对应的块，再按 title 删旧、写新。
            结果：文档增删改都和磁盘一致，不会残留过期向量。
        sync=False：只替换本次涉及 title 的块，其余文档不动（供增量上传使用）。
        """
        if persist_directory:
            db = vectorstore.create_chroma(embeddings or self.build_embeddings(), persist_directory)
        else:
            db = vectorstore.get_vectordb(embeddings)

        incoming = {c.metadata.get("title") for c in chunks if c.metadata.get("title")}

        if sync:
            for title in sorted(self._titles_in_db(db) - incoming):
                self._delete_title(db, title)  # 源文件已删除

        # 内容改过后 stable_id 会变，必须先按 title 清掉旧块再写，否则新旧并存
        for title in sorted(incoming):
            self._delete_title(db, title)

        # 空目录是合法场景（文档全删了），Chroma 的 upsert 不接受空列表，需跳过
        if chunks:
            db.add_documents(documents=chunks, ids=[self.stable_id(c) for c in chunks])
        print(f"[3] 已同步向量库，共 {db._collection.count()} 条"
              f"（本次 {len(chunks)} 块，{'全量' if sync else '增量'}）")
        return db

    # ---------- 文档管理（供 API / Streamlit 复用） ----------

    def list_documents(self, vectordb=None) -> list:
        """按 title 聚合，返回 [{title, chunks, file_exists}]。"""
        db = vectordb or vectorstore.get_vectordb()
        counts = Counter(
            m["title"]
            for m in (db.get(include=["metadatas"]).get("metadatas") or [])
            if m and m.get("title")
        )
        return [
            {
                "title": title,
                "chunks": n,
                "file_exists": (Path(self.data_dir) / f"{title}.txt").exists(),
            }
            for title, n in sorted(counts.items())
        ]

    def add_document(self, title: str, content: str, vectordb=None) -> dict:
        """新增/覆盖一篇文档：写磁盘 → 切块 → 只替换该 title 的向量块。"""
        title = validate_title(title)
        if not content or not content.strip():
            raise ValueError("content 不能为空")

        data_dir = Path(self.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / f"{title}.txt"
        path.write_text(content, encoding="utf-8")

        docs = [Document(page_content=content, metadata={"source": str(path)})]
        chunks = self._decorate(self._splitter().split_documents(docs), title=title)

        db = vectordb or vectorstore.get_vectordb()
        self._delete_title(db, title)
        db.add_documents(documents=chunks, ids=[self.stable_id(c) for c in chunks])
        return {"title": title, "chunks": len(chunks), "path": str(path)}

    def remove_document(self, title: str, vectordb=None) -> dict:
        """删除一篇文档：同时清向量块与磁盘源文件，保证下次全量同步不会把它带回来。"""
        title = validate_title(title)
        db = vectordb or vectorstore.get_vectordb()
        deleted = self._delete_title(db, title)

        path = Path(self.data_dir) / f"{title}.txt"
        file_removed = path.exists()
        if file_removed:
            path.unlink()
        return {"title": title, "deleted_chunks": deleted, "file_removed": file_removed}

    def delete_by_title(self, title: str, persist_directory: str = None,
                        vectordb=None) -> int:
        """按 metadata['title'] 精确删除某篇文档的所有块。"""
        title = validate_title(title)
        if vectordb is not None:
            db = vectordb
        elif persist_directory:
            db = vectorstore.create_chroma(self.build_embeddings(), persist_directory)
        else:
            db = vectorstore.get_vectordb()
        deleted = self._delete_title(db, title)
        print(f"[delete] 按 title='{title}' 删除 {deleted} 个块")
        return deleted


# Singleton instance
ingest_service = IngestService()
