"""
运行：  streamlit run streamlit_app.py   （默认 8501 端口）
"""
from pathlib import Path

import streamlit as st

from app.config import settings
from app.services import vectorstore
from app.services.ingest_service import ingest_service
from app.services.rag_service import rag_service

st.set_page_config(page_title="RAG 问答", page_icon="🧠")
st.title("RAG 知识库问答")


# ---------- 初始化：每个会话只做一次 ----------
if "qa_ready" not in st.session_state:
    with st.spinner("加载模型与向量库（首次较慢，需下载 bge 模型约 90MB）..."):
        try:
            emb = vectorstore.build_embeddings()
            db = vectorstore.set_vectordb(vectorstore.create_chroma(emb))
            rag_service.vectordb = db
            st.session_state.llm = rag_service.build_llm()
            st.session_state.init_error = None
        except Exception as exc:
            st.session_state.init_error = f"{type(exc).__name__}: {exc}"
        st.session_state.qa_ready = True


init_error = st.session_state.get("init_error")

if init_error:
    st.error(f"初始化失败：{init_error}")
    st.info("请检查 .env 里的 DEEPSEEK_API_KEY，以及是否已运行 `python ingest.py`。")
else:
    # ---------- 问答区：只有点「提问」才真正调用 LLM ----------
    with st.form("ask_form"):
        q = st.text_input("输入你的问题", key="question")
        top_k = st.slider("检索块数 top_k", min_value=1, max_value=10,
                          value=settings.TOP_K)
        submitted = st.form_submit_button("提问")

    if submitted:
        if not q or not q.strip():
            st.session_state.last_result = None
            st.session_state.last_error = "问题不能为空"
        else:
            with st.spinner("检索 + 生成中..."):
                try:
                    st.session_state.last_result = rag_service.ask(
                        q.strip(), llm=st.session_state.llm, top_k=top_k
                    )
                    st.session_state.last_error = None
                except Exception as exc:
                    st.session_state.last_result = None
                    st.session_state.last_error = f"生成失败：{type(exc).__name__}: {exc}"

    # 普通 rerun 只渲染缓存结果，不会重新提问
    if st.session_state.get("last_error"):
        st.error(st.session_state.last_error)

    result = st.session_state.get("last_result")
    if result:
        st.subheader("回答")
        st.write(result["answer"])
        if result.get("skipped_llm"):
            st.warning("未检索到足够相关的内容，本次没有调用大模型。")
        st.caption(
            f"top_k={result.get('top_k', settings.TOP_K)} · "
            f"距离阈值 {result.get('threshold', settings.SIMILARITY_THRESHOLD)} · "
            f"相关块 {len(result.get('sources', []))} 个（score 为 L2 距离，越小越相关）"
        )

        if result.get("sources"):
            with st.expander("查看检索到的来源（编号对应回答里的 [n]）"):
                for s in result["sources"]:
                    st.markdown(
                        f"**[{s.get('index', '?')}] {s['metadata'].get('title', '未知')} "
                        f"· score={s.get('score', '-')}**"
                    )
                    st.text(s["content"][:300] + ("..." if len(s["content"]) > 300 else ""))


# ---------- 侧栏：知识库管理 ----------
with st.sidebar:
    st.header("知识库管理")
    try:
        docs = ingest_service.list_documents()
        st.caption(
            f"集合 `{settings.COLLECTION_NAME}` · {len(docs)} 篇文档 · "
            f"{sum(d['chunks'] for d in docs)} 块 · 阈值 {settings.SIMILARITY_THRESHOLD}"
        )
        st.dataframe(docs, hide_index=True)
    except Exception as exc:
        docs = []
        st.warning(f"读取知识库失败：{exc}")

    st.subheader("上传文档")
    uploaded = st.file_uploader("选择 .txt 文件", type=["txt"])
    if uploaded is not None and st.button("写入知识库"):
        try:
            title = Path(uploaded.name).stem
            res = ingest_service.add_document(title, uploaded.getvalue().decode("utf-8"))
            st.success(f"已写入 {title}（{res['chunks']} 块）")
            st.rerun()
        except Exception as exc:
            st.error(f"写入失败：{exc}")

    st.subheader("删除文档")
    titles = [d["title"] for d in docs]
    if titles:
        selected = st.selectbox("选择文档", titles)
        confirm = st.checkbox("确认删除（同时删除磁盘源文件）")
        if confirm and st.button("删除该文档"):
            try:
                res = ingest_service.remove_document(selected)
                st.success(f"已删除 {res['deleted_chunks']} 个块")
                st.rerun()
            except Exception as exc:
                st.error(f"删除失败：{exc}")

    st.subheader("全量同步")
    st.caption("以 data/sample_documents 目录为准重建索引，清掉过期块。")
    if st.button("重新同步 data 目录"):
        try:
            chunks = ingest_service.load_and_split()
            ingest_service.build_vectordb(chunks, sync=True)
            st.success("同步完成")
            st.rerun()
        except Exception as exc:
            st.error(f"同步失败：{exc}")
