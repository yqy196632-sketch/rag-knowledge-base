"""
入库脚本。

运行：  python ingest.py
作用：  把 data/sample_documents/*.txt → 加载 → 切块 → 向量化 → 写进 Chroma。
重复运行幂等：相同内容不会重复入库（stable_id 覆盖）。
"""
from app.services.ingest_service import ingest_service


def main():
    chunks = ingest_service.load_and_split()
    emb = ingest_service.build_embeddings()
    ingest_service.build_vectordb(chunks, emb)
    print("入库完成 ✅")


if __name__ == "__main__":
    main()
