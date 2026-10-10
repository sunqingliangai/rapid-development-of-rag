"""
第6课实战：混合检索与重排序（2026-10 版）

架构（对照课程 rag_app_lesson6_1.py / 6_2.py）：
    向量检索(Chroma) ─┐
                      ├─► RRF融合排名 ─► Reranker精排 ─► Top-3 给LLM生成
    BM25关键词检索 ───┘   （课程缺失，补上）   （交叉编码器）

与课程代码的差异：
1. ★补全RRF融合：课程把向量+BM25结果直接拼接（重复块占位、分数不可比），
   我们用RRF只按"排名"融合：score(文档) = Σ 1/(k + rank)，k取60（原论文默认值）。
   以"文本内容"作为融合键——内容相同的块（跨格式重复）自动合并成一个候选，
   第5课观察到的"重复块占满Top-K"问题在这里被治好。
2. ★BM25索引只建一次：课程每次查询都重新取全库文档分词重建，我们建一次复用。
3. ★不用FlagEmbedding库，用transformers手写cross-encoder重排器：
   避开FlagEmbedding对本项目torch 2.2.2锁定链的依赖冲突；
   更重要的是看清Reranker的本质——一个"query+文档 → 相关性分数"的
   序列分类模型（输出1个logit，FlagReranker只是它的薄包装）。
   模型仍是课程同款 BAAI/bge-reranker-v2-m3（0.6B，CPU可跑；
   2026年更强的Qwen3-Reranker-4B需GPU，见本课讲解）。

运行前提：已运行 lesson5_chroma_rag.py index 建好向量库
运行方式：
    uv run --env-file .env python lesson6_hybrid_rerank.py
"""

import os

# 必须在导入 sentence-transformers 之前设置（原因见第2课）
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path

import chromadb
import jieba
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 复用第3课的组件：Embedding加载、生成流程
from lesson3_multiformat_rag import generate_process, load_embedding_model

# ---------- 配置区 ----------
BASE_DIR = Path(__file__).resolve().parent
EMBEDDING_MODEL_PATH = BASE_DIR / "models" / "bge-small-zh-v1.5"
RERANKER_MODEL_PATH = BASE_DIR / "models" / "bge-reranker-v2-m3"
CHROMA_DB_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "documents"

RECALL_TOP_K = 6   # 每路召回的数量（粗排"宽进"）
RRF_K = 60         # RRF平滑常数，原论文默认值，调小则头部排名权重更大
RERANK_CANDIDATES = 8  # 送入重排的候选数量（融合后）
FINAL_TOP_K = 3    # 最终给LLM的文本块数量（精排"严出"）


def build_bm25_index(collection):
    """
    用向量库中的全部文档构建BM25索引（只建一次，查询时复用）
    :param collection: ChromaDB的collection对象
    :return: 返回 (BM25索引, 文档列表)
    """
    # 从Chroma取全部文档（和元数据一起取，便于展示来源）
    all_data = collection.get(include=["documents", "metadatas"])
    all_docs = all_data["documents"]
    all_metadatas = all_data["metadatas"]

    # 中文分词：BM25按"词"统计词频，中文没有空格，必须先分词
    # （英文天然有空格分隔，可跳过这一步——这就是课程引入jieba的原因）
    print("BM25索引构建中（jieba分词）...")
    tokenized_corpus = []
    for doc in all_docs:
        tokens = list(jieba.cut(doc))
        tokenized_corpus.append(tokens)

    # BM25Okapi：经典的Okapi BM25实现（词频饱和+文档长度归一，见课程讲解）
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"BM25索引构建完成，共{len(all_docs)}篇文档\n")
    return bm25, all_docs, all_metadatas


def vector_retrieve(query, collection, embedding_model, top_k=RECALL_TOP_K):
    """
    向量检索路：query向量化 → Chroma相似度检索 → 返回按相似度排序的文本列表
    :return: 返回文本块列表（按相关度从高到低）
    """
    query_embedding = embedding_model.encode(query, normalize_embeddings=True)
    query_embedding = query_embedding.tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    documents = results["documents"][0]
    return documents


def bm25_retrieve(query, bm25, all_docs, top_k=RECALL_TOP_K):
    """
    BM25关键词检索路：query分词 → 逐文档打分 → 返回按得分排序的文本列表
    :return: 返回文本块列表（按相关度从高到低）
    """
    # 查询同样需要分词（与建索引时的分词方式保持一致）
    tokenized_query = list(jieba.cut(query))

    # 计算查询与每个文档的BM25得分（一个浮点数数组）
    bm25_scores = bm25.get_scores(tokenized_query)

    # 按得分从高到低排序，取前top_k个文档的下标
    ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    top_indices = ranked_indices[:top_k]

    documents = []
    for i in top_indices:
        documents.append(all_docs[i])
    return documents


def rrf_fusion(route_results):
    """
    RRF（Reciprocal Rank Fusion，递归折减融合）：
    多路检索的分数不可比（余弦0.8 ≠ BM25的12.3分），
    RRF只看排名不看分数：score(文档) = Σ 1/(RRF_K + rank)，
    在多路都排靠前的文档综合得分最高。

    ★以"文本内容"为融合键：内容完全相同的块（跨格式重复）会累加得分
    并自动合并为一个候选——既去重又强化了"多路都认可"的信号。
    :param route_results: 各路检索的文本列表，如[向量路结果, BM25路结果]
    :return: 返回按RRF得分排序的(文本, 得分)列表
    """
    # 用字典累积每个文本的RRF得分
    rrf_scores = {}
    for route in route_results:
        for rank, doc_text in enumerate(route, start=1):
            if doc_text not in rrf_scores:
                rrf_scores[doc_text] = 0.0
            # rank从1开始计；排名越靠前贡献越大（1/(60+1) > 1/(60+2)）
            rrf_scores[doc_text] += 1.0 / (RRF_K + rank)

    # 按累计得分从高到低排序
    fused = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return fused


def load_reranker():
    """
    加载cross-encoder重排模型（手写版，等价于课程的FlagReranker）
    :return: 返回 (tokenizer, model)
    """
    print("加载重排序模型(bge-reranker-v2-m3)...")
    tokenizer = AutoTokenizer.from_pretrained(str(RERANKER_MODEL_PATH))
    # AutoModelForSequenceClassification：序列分类模型。
    # bge-reranker的输出头是1个logit——把它当作"query与文档相关"的二分类分数
    # （CPU推理用默认float32，568M参数约2.3GB内存，Intel Mac可承受）
    model = AutoModelForSequenceClassification.from_pretrained(str(RERANKER_MODEL_PATH))
    model.eval()  # 推理模式（关闭dropout等训练行为）
    return tokenizer, model


def rerank(query, candidate_docs, tokenizer, model, top_k=FINAL_TOP_K):
    """
    重排序（精排）：对每个"query-文档"对逐一精细打分，按分数重排取前top_k
    :param candidate_docs: 融合后的候选文本列表
    :return: 返回(排序后的文本列表, 对应的相关性分数列表)
    """
    # 构造交叉编码器的输入对：每个候选文档与query组成一对
    input_pairs = []
    for doc_text in candidate_docs:
        input_pairs.append([query, doc_text])

    # tokenizer支持"句子对"编码：[CLS] query [SEP] 文档 [SEP]
    # query和文档在同一个序列里做全注意力交互——这就是cross-encoder
    # 比双塔embedding精度高的原因（代价是每对都要实时算，无法预计算）
    inputs = tokenizer(
        input_pairs,
        padding=True,        # 批内补齐到最长序列
        truncation=True,     # 超长截断
        max_length=512,      # 与模型训练时一致
        return_tensors="pt", # 返回PyTorch张量
    )

    # 推理阶段不需要梯度，no_grad省内存提速
    with torch.no_grad():
        logits = model(**inputs).logits

    # 每对得到1个logit，squeeze成一维列表
    raw_scores = logits.squeeze(-1).tolist()
    # 单个logit范围是(-∞,+∞)，套sigmoid映射到(0,1)便于阅读比较
    # （课程FlagReranker的normalize=True做的正是这一步）
    scores = []
    for raw_score in raw_scores:
        score = torch.sigmoid(torch.tensor(raw_score)).item()
        scores.append(score)

    # 按分数从高到低排序，取前top_k
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    reranked_docs = []
    reranked_scores = []
    for i in sorted_indices[:top_k]:
        reranked_docs.append(candidate_docs[i])
        reranked_scores.append(scores[i])
    return reranked_docs, reranked_scores


def main():
    print("第6课：混合检索 + RRF融合 + 重排序\n" + "=" * 60)
    query = "零售业数字化转型面临哪些挑战？"
    print(f"查询语句: {query}\n")

    # ---------- 准备工作：连接向量库、加载三个模型 ----------
    client = chromadb.PersistentClient(str(CHROMA_DB_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    if collection.count() == 0:
        raise SystemExit("向量库为空，请先运行: uv run python lesson5_chroma_rag.py index")

    embedding_model = load_embedding_model()
    bm25, all_docs, all_metadatas = build_bm25_index(collection)
    tokenizer, reranker_model = load_reranker()

    # ---------- 第一路：向量检索（语义匹配，粗排） ----------
    vector_docs = vector_retrieve(query, collection, embedding_model)
    print("\n【第一路：向量检索 Top-%d】" % RECALL_TOP_K)
    for rank, doc in enumerate(vector_docs, start=1):
        print(f"  {rank}. {doc[:50]}...")

    # ---------- 第二路：BM25关键词检索（精确匹配，粗排） ----------
    bm25_docs = bm25_retrieve(query, bm25, all_docs)
    print("\n【第二路：BM25关键词检索 Top-%d】" % RECALL_TOP_K)
    for rank, doc in enumerate(bm25_docs, start=1):
        print(f"  {rank}. {doc[:50]}...")

    # ---------- 融合：RRF排名融合（去重+合并多路信号） ----------
    fused = rrf_fusion([vector_docs, bm25_docs])
    fused_docs = []
    for doc_text, rrf_score in fused[:RERANK_CANDIDATES]:
        fused_docs.append(doc_text)
    print(f"\n【RRF融合后候选 Top-%d】（内容相同的块已自动合并）" % RERANK_CANDIDATES)
    for rank, (doc_text, rrf_score) in enumerate(fused[:RERANK_CANDIDATES], start=1):
        print(f"  {rank}. [RRF分 {rrf_score:.4f}] {doc_text[:46]}...")

    # ---------- 精排：Cross-Encoder重排序 ----------
    reranked_docs, reranked_scores = rerank(query, fused_docs, tokenizer, reranker_model)
    print(f"\n【重排序后最终 Top-%d】（送给LLM生成）" % FINAL_TOP_K)
    for rank, (doc_text, score) in enumerate(zip(reranked_docs, reranked_scores), start=1):
        print(f"\n--- 第{rank}名 (相关性 {score:.4f}) ---")
        print(doc_text)

    # ---------- 生成 ----------
    print("\n" + "=" * 60)
    generate_process(query, reranked_docs)
    print("\n" + "=" * 60 + "\nRAG流程结束")


if __name__ == "__main__":
    main()
