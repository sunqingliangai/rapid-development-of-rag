"""
第4课实战：分块策略对比实验（2026-10 版）

实验设计（控制变量法）：
    固定：同一份文档、同一批测试问题、同一个Embedding模型、同一个FAISS检索器
    变化：只有分块方式不同
    指标：Top-1相似度、Top-3相似度均值（越高代表检索越"命中"）

三个实验：
    实验1 参数扫描：chunk_size(256/512/1024) × chunk_overlap(0/128/256) 的组合
    实验2 策略对比：递归分块 vs 文档特定分块(Markdown标题) vs 简版语义分块
    实验3 加餐：Parent-Child分块（小块检索、大块生成，2026年工程主流做法）

重要说明：
    相似度得分是"代理指标"——它衡量检索命中率，不直接等于答案质量；
    答案质量的正式评估方法（RAGAS等）是第8课的主题。
    本实验全程离线运行（只用本地Embedding模型，不调用任何API）。

运行方式：
    uv run python lesson4_chunking_experiments.py
"""

import os

# 必须在导入 sentence-transformers 之前设置（原因见第2课）
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re

import faiss
import numpy as np
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# 复用第3课的现成组件：文档加载函数、Embedding模型加载函数、数据目录
from lesson3_multiformat_rag import DATA_FOLDER, load_document, load_embedding_model

# ---------- 实验配置 ----------
# 测试问题集：每个问题都对应文档中确定存在的内容（用于公平比较各策略的命中率）
EVAL_QUERIES = [
    "制造业数字化转型面临哪些挑战？",
    "零售业数字化转型面临哪些挑战？",
    "金融业数字化转型面临哪些挑战？",
    "数字化转型对客户体验有什么影响？",
    "制造业案例中的公司是什么背景？",
]


def build_index(chunks, embedding_model):
    """
    把文本块列表构建成FAISS索引（流程与第2/3课一致）
    :param chunks: 文本块列表
    :param embedding_model: 预加载的Embedding模型
    :return: 返回FAISS向量索引
    """
    embeddings = []
    for chunk in chunks:
        embedding = embedding_model.encode(chunk, normalize_embeddings=True)
        embeddings.append(embedding)
    embeddings_np = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings_np.shape[1])
    index.add(embeddings_np)
    return index


def evaluate_retrieval(chunks, embedding_model, queries, top_k=3):
    """
    评估一组文本块的检索效果：对每个测试问题做Top-K检索，统计相似度得分
    :param chunks: 文本块列表
    :param embedding_model: Embedding模型
    :param queries: 测试问题列表
    :param top_k: 检索返回的前K个结果
    :return: 返回 (Top-1相似度平均值, Top-3相似度平均值)
    """
    index = build_index(chunks, embedding_model)

    top1_scores = []
    top3_scores = []
    for query in queries:
        query_embedding = embedding_model.encode(query, normalize_embeddings=True)
        query_embedding = np.array([query_embedding])
        scores, indices = index.search(query_embedding, top_k)
        # 记录Top-1得分（最相关块的相似度）
        top1_scores.append(scores[0][0])
        # 记录Top-3得分均值（前3块的相似度平均水平）
        top3_scores.append(float(np.mean(scores[0])))

    avg_top1 = sum(top1_scores) / len(top1_scores)
    avg_top3 = sum(top3_scores) / len(top3_scores)
    return avg_top1, avg_top3


def experiment_1_chunk_size_sweep(document_text, embedding_model):
    """
    实验1：参数扫描——chunk_size与chunk_overlap的不同组合对检索效果的影响
    预期观察：块太小→语义碎片化；块太大→向量被"平均化"；overlap缓解边界切断
    """
    print("=" * 70)
    print("实验1：参数扫描（固定递归分块策略，只变chunk_size和chunk_overlap）")
    print("=" * 70)
    print(f"{'chunk_size':>10} | {'overlap':>7} | {'块数':>4} | {'Top-1均分':>9} | {'Top-3均分':>9}")
    print("-" * 70)

    # 参数组合列表：覆盖小块/中块/大块 × 无重叠/有重叠
    param_combos = [
        (256, 0),
        (256, 128),
        (512, 0),
        (512, 128),
        (1024, 128),
        (1024, 256),
    ]

    for chunk_size, chunk_overlap in param_combos:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = text_splitter.split_text(document_text)
        avg_top1, avg_top3 = evaluate_retrieval(chunks, embedding_model, EVAL_QUERIES)
        print(f"{chunk_size:>10} | {chunk_overlap:>7} | {len(chunks):>4} | {avg_top1:>9.4f} | {avg_top3:>9.4f}")

    print()


def experiment_2_strategy_comparison(md_text, embedding_model):
    """
    实验2：策略对比——同样的Markdown文档，分别用三种策略分块
    （注意：为了控制变量，两种基于文本的策略都作用在同一份md内容上）
    """
    print("=" * 70)
    print("实验2：分块策略对比（同一份Markdown文档，三种策略）")
    print("=" * 70)
    print(f"{'策略':>14} | {'块数':>4} | {'Top-1均分':>9} | {'Top-3均分':>9}")
    print("-" * 70)

    # ---------- 策略A：递归分块（我们的基准线，与第2/3课相同） ----------
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=128
    )
    recursive_chunks = recursive_splitter.split_text(md_text)
    avg_top1, avg_top3 = evaluate_retrieval(recursive_chunks, embedding_model, EVAL_QUERIES)
    print(f"{'递归分块':>14} | {len(recursive_chunks):>4} | {avg_top1:>9.4f} | {avg_top3:>9.4f}")

    # ---------- 策略B：文档特定分块（按Markdown标题层级切分） ----------
    # MarkdownHeaderTextSplitter按标题层级切分，每个块自带标题元数据；
    # 这里strip_headers=False保留标题文字在块内（标题本身承载重要语义）
    headers_to_split_on = [
        ("#", "标题1"),
        ("##", "标题2"),
        ("###", "标题3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    # split_text返回Document对象列表，需要取出其中的文本内容
    markdown_documents = markdown_splitter.split_text(md_text)
    markdown_chunks = []
    for document in markdown_documents:
        markdown_chunks.append(document.page_content)
    # 结构化分块不保证块大小：太长的节没有被切小（这是它的特点，也是风险）
    avg_top1, avg_top3 = evaluate_retrieval(markdown_chunks, embedding_model, EVAL_QUERIES)
    print(f"{'Markdown标题':>14} | {len(markdown_chunks):>4} | {avg_top1:>9.4f} | {avg_top3:>9.4f}")

    # ---------- 策略C：简版语义分块（课程定义的语义分块思想，手工实现） ----------
    semantic_chunks = semantic_chunking(md_text, embedding_model)
    avg_top1, avg_top3 = evaluate_retrieval(semantic_chunks, embedding_model, EVAL_QUERIES)
    print(f"{'简版语义分块':>14} | {len(semantic_chunks):>4} | {avg_top1:>9.4f} | {avg_top3:>9.4f}")

    print()


def semantic_chunking(document_text, embedding_model, breakpoint_percentile=70):
    """
    简版语义分块（对齐课程对"语义分块"的定义，手工实现核心逻辑）：
        第1步 断句 → 第2步 逐句编码 → 第3步 计算相邻句子相似度
        → 第4步 相似度骤降处=主题边界 → 第5步 合并句子成块

    :param document_text: 文档全文
    :param embedding_model: Embedding模型
    :param breakpoint_percentile: 断点阈值百分位（越大切得越碎）
    :return: 返回语义分块列表
    """
    # 第1步：按中文句末标点断句（教学用简版正则；生产可用spaCy/NLTK分句）
    sentences = re.split(r"(?<=[。！？；])\s*", document_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # 第2步：逐句编码（归一化向量，点积即余弦相似度）
    sentence_embeddings = []
    for sentence in sentences:
        embedding = embedding_model.encode(sentence, normalize_embeddings=True)
        sentence_embeddings.append(embedding)

    # 第3步：计算每对相邻句子的相似度
    # similarities[i] 表示第i句和第i+1句之间的相似度
    similarities = []
    for i in range(1, len(sentence_embeddings)):
        similarity = float(np.dot(sentence_embeddings[i - 1], sentence_embeddings[i]))
        similarities.append(similarity)

    # 第4步：确定断点阈值——取相似度分布的百分位数，
    # 低于阈值的位置意味着"话题在这里切换了"
    threshold = np.percentile(similarities, breakpoint_percentile)

    # 第5步：按断点合并句子成块
    chunks = []
    current_chunk_sentences = [sentences[0]]
    for i in range(1, len(sentences)):
        # similarities[i-1]是"第i-1句与第i句"的相似度
        if similarities[i - 1] < threshold:
            # 相似度低于阈值：此处是主题边界，当前块结束
            chunks.append("".join(current_chunk_sentences))
            current_chunk_sentences = [sentences[i]]
        else:
            # 相似度高于阈值：继续并入当前块
            current_chunk_sentences.append(sentences[i])
    # 别忘了最后一块
    chunks.append("".join(current_chunk_sentences))
    return chunks


def experiment_3_parent_child(document_text, embedding_model):
    """
    实验3（加餐）：Parent-Child分块，也叫Small-to-Big
    —— 2026年工程主流做法，课程未覆盖

    核心思想：解决"检索要小块准、生成要大块全"的矛盾
        父块 = 大块（如1024字符）→ 用于最终给LLM生成答案
        子块 = 从父块切出的小块（如256字符）→ 用于向量检索匹配
        检索时命中子块，返回它所属的父块
    """
    print("=" * 70)
    print("实验3：Parent-Child分块（小块检索命中 → 返回大块给LLM）")
    print("=" * 70)

    # 第1步：切父块（大块，无重叠，保证父块之间不重复）
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=0
    )
    parent_chunks = parent_splitter.split_text(document_text)
    print(f"父块数量: {len(parent_chunks)}（每块最大1024字符）")

    # 第2步：每个父块再切成子块，同时维护"子块→父块"的映射关系
    # 注意：这和第3课闭环演示是同一个模式——靠列表下标对齐维护映射！
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256, chunk_overlap=0
    )
    child_chunks = []
    child_to_parent = []  # 第i个子块属于child_to_parent[i]号父块
    for parent_index, parent_chunk in enumerate(parent_chunks):
        children = child_splitter.split_text(parent_chunk)
        for child in children:
            child_chunks.append(child)
            child_to_parent.append(parent_index)
    print(f"子块数量: {len(child_chunks)}（每块最大256字符）")

    # 第3步：只用子块建向量索引（检索发生在小粒度上，精度高）
    index = build_index(child_chunks, embedding_model)

    # 第4步：检索演示——命中子块，返回父块
    query = "金融业数字化转型面临哪些挑战？"
    query_embedding = embedding_model.encode(query, normalize_embeddings=True)
    query_embedding = np.array([query_embedding])
    scores, indices = index.search(query_embedding, 1)

    # 命中的子块下标 → 找到它属于哪个父块
    hit_child_index = indices[0][0]
    hit_score = scores[0][0]
    hit_parent_index = child_to_parent[hit_child_index]

    print(f"\n查询: {query}")
    print(f"\n命中子块（相似度{hit_score:.4f}，共{len(child_chunks[hit_child_index])}字符）:")
    print("-" * 60)
    print(child_chunks[hit_child_index][:200] + "...")
    print(f"\n返回给LLM的父块（共{len(parent_chunks[hit_parent_index])}字符，包含完整上下文）:")
    print("-" * 60)
    print(parent_chunks[hit_parent_index][:400] + "...")
    print("-" * 60)
    print("观察：子块精准锚定到金融业挑战的段落，而返回的父块还带着该案例的")
    print("公司背景等完整上下文——检索精度和生成上下文两不误，这就是Small-to-Big。")
    print()


def main():
    print("第4课：分块策略对比实验\n")

    # 加载Embedding模型（所有实验共用，保证可比性）
    embedding_model = load_embedding_model()

    # 实验文档：用test.txt（完整版内容，约9000字符）
    # 实验1、3用同一份全文；实验2为了用Markdown结构对比，使用test.md原文
    txt_path = os.path.join(str(DATA_FOLDER), "test.txt")
    md_path = os.path.join(str(DATA_FOLDER), "test.md")
    document_text = load_document(txt_path)
    md_text = load_document(md_path)
    print(f"实验文档(test.txt)总字符数: {len(document_text)}")
    print(f"实验文档(test.md)总字符数: {len(md_text)}（节选版，实验2专用）\n")

    # 三个实验依次执行
    experiment_1_chunk_size_sweep(document_text, embedding_model)
    experiment_2_strategy_comparison(md_text, embedding_model)
    experiment_3_parent_child(document_text, embedding_model)

    print("=" * 70)
    print("实验全部完成。注意：相似度是代理指标，只衡量检索命中；")
    print("不同策略对最终答案质量的影响，需要第8课的RAG评估体系来回答。")


if __name__ == "__main__":
    main()
