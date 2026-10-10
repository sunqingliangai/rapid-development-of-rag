"""
第5课实战：ChromaDB 持久化向量数据库（2026-10 版）

在第3课多格式解析的基础上，把存储层从"内存FAISS"换成"ChromaDB持久化"。
（对照课程 rag_app_lesson5.py，chromadb 1.5.9）

与课程代码的三个关键差异（都有明确理由）：
1. ★修正距离度量Bug：课程建collection未指定距离空间，Chroma默认用L2
   （欧氏距离，越小越相似），课程却把返回的distance当"相似度"打印。
   我们显式指定cosine空间：metadata={"hnsw:space": "cosine"}，
   此时返回的distance = 1 - 余弦相似度，换算后打印才是真正的相似度。
2. ★索引与查询分离：课程每次启动都shutil.rmtree删库重建，恰好废掉了
   "持久化"这个换库的核心价值。我们做成两个子命令：
   index（建库，只需运行一次）、query（查询，直接用已有库，不再解析文档）。
3. ★元数据过滤实战：每个chunk记录来源文件名（metadata），
   查询时可指定--source只在某个文件里检索——这是向量库相对FAISS的
   核心增量能力，也是课程讲到但没实现的功能。

另外注意：检索top_k从3变成6，这是为第6课"先粗检、重排序后精选"做的铺垫。

运行方式：
    uv run --env-file .env python lesson5_chroma_rag.py index
    uv run --env-file .env python lesson5_chroma_rag.py query "问题"
    uv run --env-file .env python lesson5_chroma_rag.py query "问题" --source test.pdf
"""

import os

# 必须在导入 sentence-transformers 之前设置（原因见第2课）
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import uuid
from pathlib import Path

import chromadb

# 复用第3课的现成组件：多格式解析、Embedding加载、生成流程、分块参数
from lesson3_multiformat_rag import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    load_document,
    load_embedding_model,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------- 配置区 ----------
BASE_DIR = Path(__file__).resolve().parent
EMBEDDING_MODEL_PATH = BASE_DIR / "models" / "bge-small-zh-v1.5"
DATA_FOLDER = BASE_DIR / "data" / "lesson5"

# ChromaDB持久化存储目录（数据落盘，重启不丢；已加入.gitignore）
CHROMA_DB_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "documents"

TOP_K = 6  # 检索返回前6个（比之前的3个多——为第6课重排序预留"宽进"空间）


def get_collection():
    """
    创建/连接ChromaDB的持久化客户端和collection
    :return: 返回collection对象（索引和查询共用同一个入口）
    """
    # PersistentClient：数据自动落盘到CHROMA_DB_PATH目录，
    # 进程退出后数据仍在——这就是"持久化"，FAISS内存索引做不到
    client = chromadb.PersistentClient(str(CHROMA_DB_PATH))

    # get_or_create_collection：不存在则创建，存在则直接复用
    # ★关键：显式指定cosine距离空间。
    #   不指定时Chroma默认用L2（欧氏距离），语义检索应该用余弦相似度；
    #   cosine空间下query返回的distance = 1 - 余弦相似度
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def indexing_process(folder_path, embedding_model, collection):
    """
    索引流程：解析文件夹中所有文档 → 分块 → 向量化 → 连同元数据存入ChromaDB
    :param folder_path: 文档文件夹路径
    :param embedding_model: 预加载的Embedding模型
    :param collection: ChromaDB的collection对象
    """
    # 如果库里已有数据（重复运行index），用collection级API重建，
    # 比课程代码的shutil.rmtree删目录更规范——这是数据库的"DROP TABLE"
    if collection.count() > 0:
        print(f"库中已有{collection.count()}条数据，删除重建...")
        client = chromadb.PersistentClient(str(CHROMA_DB_PATH))
        client.delete_collection(COLLECTION_NAME)
        collection = get_collection()

    all_chunks = []
    all_ids = []
    all_metadatas = []

    # 遍历文件夹中所有文档（与第3课一致的解析流程）
    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue

        document_text = load_document(file_path)
        if not document_text.strip():
            continue
        print(f"文档 {filename}: {len(document_text)}字符")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = text_splitter.split_text(document_text)
        print(f"  分割为 {len(chunks)} 个文本块")

        # 每个chunk准备三样东西：原文、唯一ID、元数据
        for chunk in chunks:
            all_chunks.append(chunk)
            # uuid4生成全局唯一ID（向量库要求每条记录有ID，用于更新/删除）
            chunk_id = str(uuid.uuid4())
            all_ids.append(chunk_id)
            # 元数据记录来源文件名——元数据过滤的基石
            all_metadatas.append({"source": filename})

    print(f"\n共{len(all_chunks)}个文本块，开始向量化...")

    # 逐块编码为归一化向量（Chroma接受普通list，用tolist()转换）
    all_embeddings = []
    for chunk in all_chunks:
        embedding = embedding_model.encode(chunk, normalize_embeddings=True)
        all_embeddings.append(embedding.tolist())

    # 一次性入库：ids、向量、原文、元数据四元组对齐
    # （对比第2/3课：FAISS需要我们手工维护"向量下标↔原文"的对齐，向量库把这套关系连同元数据一起管理了）
    collection.add(
        ids=all_ids,
        embeddings=all_embeddings,
        documents=all_chunks,
        metadatas=all_metadatas,
    )
    print(f"索引完成，共入库 {collection.count()} 条记录，已持久化到 {CHROMA_DB_PATH}")


def retrieval_process(
    query, collection, embedding_model, top_k=TOP_K, source_filter=None
):
    """
    检索流程：查询向量化 → ChromaDB相似度检索（可选元数据过滤）→ 返回原文块
    :param query: 用户查询语句
    :param collection: ChromaDB的collection对象
    :param embedding_model: Embedding模型
    :param top_k: 返回前K个结果
    :param source_filter: 可选的来源文件名过滤条件
    :return: 返回最相似的文本块列表
    """
    # 查询向量化（与入库时同一个模型、同样归一化）
    query_embedding = embedding_model.encode(query, normalize_embeddings=True)
    query_embedding = query_embedding.tolist()

    # 元数据过滤：where条件限定只检索来源为source_filter的chunk。
    # 过滤发生在向量检索"之前或之中"（预过滤），不会先全库搜再筛——
    # 这就是向量库元数据能力与"检索后自己写Python过滤"的本质区别
    where_clause = None
    if source_filter:
        where_clause = {"source": source_filter}

    # Chroma查询：results是嵌套字典结构，
    # 每个字段的第[0]层对应第一条（也是唯一一条）查询的结果
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_clause,
    )

    doc_ids = results["ids"][0]
    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    filter_desc = f"（限定来源: {source_filter}）" if source_filter else ""
    print(f"查询语句: {query}{filter_desc}")
    print(f"最相似的前{len(documents)}个文本块:")

    retrieved_chunks = []
    for i in range(len(documents)):
        # cosine空间下 distance = 1 - 余弦相似度，换算回相似度打印
        # （★这就是课程代码缺失的一步：它直接把L2距离当相似度打印了）
        similarity = 1 - distances[i]
        chunk_text = documents[i]
        source_file = metadatas[i]["source"]
        print(
            f"\n--- 文本块{i} (相似度{similarity:.4f} | ID:{doc_ids[i][:8]}... | 来源:{source_file}) ---"
        )
        print(chunk_text)
        retrieved_chunks.append(chunk_text)

    print("\n检索完成")
    return retrieved_chunks


def main():
    # 命令行参数解析：index建库 / query查询 两种子命令
    parser = argparse.ArgumentParser(description="第5课：ChromaDB持久化RAG应用")
    parser.add_argument(
        "command", choices=["index", "query"], help="index=构建索引库, query=检索问答"
    )
    parser.add_argument("question", nargs="?", default=None, help="query模式的问题")
    parser.add_argument(
        "--source", default=None, help="可选：限定检索的来源文件名，如 test.pdf"
    )
    args = parser.parse_args()

    # 连接（或创建）持久化collection
    collection = get_collection()

    if args.command == "index":
        # ---------- 索引模式：解析文档、建库（只需运行一次） ----------
        print("索引模式：开始构建向量库\n" + "=" * 60)
        embedding_model = load_embedding_model()
        indexing_process(str(DATA_FOLDER), embedding_model, collection)
        print("\n后续可直接用query模式查询，无需重新索引。")
        return

    # ---------- 查询模式：直接使用已持久化的库 ----------
    # 持久化的价值在此体现：不加载文档、不重新解析、不重新向量化全部语料
    if collection.count() == 0:
        raise SystemExit(
            "向量库为空，请先运行: uv run --env-file .env python lesson5_chroma_rag.py index"
        )
    if not args.question:
        raise SystemExit(
            'query模式需要提供问题，例如: ... query "制造业的挑战有哪些？"'
        )
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit(
            "未检测到 DEEPSEEK_API_KEY，请在 .env 中配置后用 --env-file 加载"
        )

    print(
        f"查询模式：库中已有 {collection.count()} 条记录（持久化数据，本次未重新解析文档）\n"
        + "=" * 60
    )

    embedding_model = load_embedding_model()
    query = args.question

    # 检索（可带元数据过滤），再复用第3课的生成流程
    retrieved_chunks = retrieval_process(
        query, collection, embedding_model, source_filter=args.source
    )

    # 生成流程直接复用第3课的实现（与存储层无关，这正是分层的意义）
    from lesson3_multiformat_rag import generate_process

    generate_process(query, retrieved_chunks)

    print("\n" + "=" * 60 + "\nRAG流程结束")


if __name__ == "__main__":
    main()
