"""
RAG Service - 问答链编排。
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.config import settings
from app.services import vectorstore

NO_CONTEXT_ANSWER = (
    "知识库中没有找到与问题相关的内容，无法回答。"
    "（可补充相关文档，或调大 SIMILARITY_THRESHOLD 后重试）"
)

PROMPT = ChatPromptTemplate.from_template(
    "你是一个严谨的问答助手，只能基于下面的「上下文」回答；"
    "上下文找不到就诚实说不知道。\n"
    "回答中请用 [1]、[2] 这样的编号标注所依据的来源，不要编造未提供的编号。\n\n"
    "上下文：\n{context}\n\n"
    "问题：{question}"
)


class RAGService:
    """问答服务：构建 ChatDeepSeek 链并对 Chroma 检索结果生成答案。"""

    def __init__(self, vectordb=None, llm=None):
        self.vectordb = vectordb
        self.llm = llm
        self._llm_cache = None

    # ---------- 资源 ----------

    def _db(self, vectordb=None):
        return vectordb or self.vectordb or vectorstore.get_vectordb()

    def clear_llm_cache(self):
        """丢弃缓存的 LLM（配置变更 / 测试隔离用）。"""
        self._llm_cache = None

    def get_llm(self):
        """懒加载 + 复用 LLM 客户端；缺 key 时每次都会重新抛错，不会被缓存成 200。"""
        if self.llm is not None:
            return self.llm
        if self._llm_cache is None:
            self._llm_cache = self.build_llm()
        return self._llm_cache

    def build_llm(self):
        """构建 ChatDeepSeek；缺 key 时显式报错（而不是把错误当答案静默返回 200）。"""
        from langchain_deepseek import ChatDeepSeek

        api_key = settings.DEEPSEEK_API_KEY
        if not api_key:
            raise RuntimeError("未设置 DEEPSEEK_API_KEY，请在 .env 中填写")

        # RAG 的答案已被检索上下文限定，长思考收益小却明显增加延迟与 token 成本，默认关掉
        extra = {}
        if settings.DEEPSEEK_THINKING.strip().lower() in ("disabled", "off", "false", "0"):
            extra["extra_body"] = {"thinking": {"type": "disabled"}}

        return ChatDeepSeek(
            model=settings.DEEPSEEK_MODEL,
            temperature=0.3,
            api_key=api_key,
            request_timeout=settings.LLM_TIMEOUT,
            max_retries=settings.LLM_MAX_RETRIES,
            **extra,
        )

    # ---------- 检索与生成 ----------

    @staticmethod
    def _resolve_k(top_k=None) -> int:
        k = settings.TOP_K if top_k is None else int(top_k)
        return max(1, min(k, 50))

    def _retriever(self, vectordb, top_k=None):
        return vectordb.as_retriever(search_kwargs={"k": self._resolve_k(top_k)})

    @staticmethod
    def _format_docs(docs) -> str:
        """带来源编号的上下文，编号与响应 sources 的 index 一一对应。"""
        return "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))

    def build_rag_chain(self, vectordb=None, llm=None, top_k=None):
        """标准 LCEL 链（retriever | prompt | llm | parser），供测试与自定义编排使用。"""
        vectordb = self._db(vectordb)
        llm = llm or self.get_llm()
        retriever = self._retriever(vectordb, top_k)
        return (
            {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | PROMPT
            | llm
            | StrOutputParser()
        )

    def ask(self, question: str, vectordb=None, llm=None, top_k=None) -> dict:
        """
        一站式问答：返回 {answer, sources, top_k, threshold, skipped_llm}。

        只检索一次；距离高于阈值的块直接丢弃，全部丢弃时不调用 LLM。
        sources 每项含 index / score，前端可展示并对应答案里的 [n] 引用。
        """
        vectordb = self._db(vectordb)
        k = self._resolve_k(top_k)
        threshold = settings.SIMILARITY_THRESHOLD

        # Chroma 返回 (Document, L2 距离)，距离越小越相关
        pairs = vectordb.similarity_search_with_score(question, k=k)
        kept = [
            (doc, float(score))
            for doc, score in pairs
            if threshold <= 0 or float(score) <= threshold
        ]

        sources = [
            {
                "index": i,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": round(score, 4),
            }
            for i, (doc, score) in enumerate(kept, 1)
        ]

        if not kept:
            return {
                "answer": NO_CONTEXT_ANSWER,
                "sources": [],
                "top_k": k,
                "threshold": threshold,
                "skipped_llm": True,
            }

        chain = PROMPT | (llm or self.get_llm()) | StrOutputParser()
        answer = chain.invoke({
            "context": self._format_docs([doc for doc, _ in kept]),
            "question": question,
        })
        return {
            "answer": answer,
            "sources": sources,
            "top_k": k,
            "threshold": threshold,
            "skipped_llm": False,
        }


# Singleton instance（运行时由 main / streamlit 注入 vectordb 与 llm）
rag_service = RAGService()
