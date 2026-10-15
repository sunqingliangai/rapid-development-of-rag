"""
第10课收官实战：手写mini-GraphRAG（2026-10版）

延续"手写看清本质"的传统（手写reranker、手写mini-RAGAS），用约300行代码
实现微软GraphRAG的完整骨架，对照论文《From Local to Global》：

索引阶段（一次付清，GraphRAG的"贵"就在这里）：
    原文分块 ──► LLM抽取实体关系三元组 ──► networkx建图
                                          ──► 社区检测（聚类）──► LLM社区摘要
                                          ──► 可视化输出PNG

查询阶段（两种模式）：
    Local Search局部搜索（实体中心问题）：
        问题 ──► 实体链接（LLM从图中选相关实体）──► 一跳邻居扩展
             ──► 三元组+原文块组装上下文 ──► 生成
    Global Search全局搜索（归纳总结问题，向量RAG的死穴）：
        问题 ──► Map：逐社区摘要各自作答 ──► Reduce：LLM汇总成最终答案

与生产实现的差距（教学简化，注释中说明）：
    - 社区检测用networkx的贪心模块度算法（生产用Leiden算法，效果更好）
    - 无社区层级树（原版把社区逐层合并成"摘要树"，此处单层）
    - 无实体对齐（LLM对同一实体的不同叫法不合并——生产中的真实难题）
    - 惰性思想（LazyGraphRAG）未实现：原版摘要预付，惰性版查询时再算

运行方式：
    uv run --env-file .env python lesson10_graph_rag.py
"""

import os

# 必须在导入 sentence-transformers 之前设置（原因见第2课）
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ⚠️导入顺序（第7课的教训）：先导入第6课模块带入torch/transformers，再导入其他
from lesson6_hybrid_rerank import (  # noqa: E402
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    bm25_retrieve,
    build_bm25_index,
    load_reranker,
    rrf_fusion,
    vector_retrieve,
)
from lesson7_prompt_experiments import build_prompt_production  # noqa: E402
from lesson3_multiformat_rag import load_document, load_embedding_model  # noqa: E402
from lesson8_evaluation import parse_judge_json  # noqa: E402  (复用JSON容错解析)

import chromadb  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")  # 无界面后端：只存文件不弹窗（服务器/脚本环境必需）
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

# ---------- 配置区 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_TXT = os.path.join(BASE_DIR, "data", "lesson5", "test.txt")  # 完整版文档
GRAPH_IMAGE_PATH = os.path.join(BASE_DIR, "data", "lesson10_graph.png")
# 图谱缓存文件：索引一次、查询多次（GraphRAG的核心理念；
# 也让瞬时API故障后的重跑不必重新花十几分钟和几十次调用建图）
GRAPH_CACHE_PATH = os.path.join(BASE_DIR, "data", "lesson10_graph_cache.json")

EXTRACT_CHUNK_SIZE = 1000  # 抽取用大块（实体关系需要较长上下文）

# 控制面模型：三元组抽取/摘要/实体链接（便宜的任务）
CONTROL_MODEL = os.environ.get("ZHIPU_MODEL", "glm-4.5-flash")
CONTROL_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
# 生成模型：最终答案
GENERATOR_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
GENERATOR_BASE_URL = "https://api.deepseek.com"


def get_llm(role, temperature=0.1):
    """按角色（控制面/生成）创建对应的模型客户端"""
    if role == "control":
        return ChatOpenAI(
            model=CONTROL_MODEL,
            api_key=os.environ["ZHIPU_API_KEY"],
            base_url=CONTROL_BASE_URL,
            temperature=temperature,
        )
    return ChatOpenAI(
        model=GENERATOR_MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=GENERATOR_BASE_URL,
        temperature=temperature,
    )


# ======================================================================
# 第一部分：索引——LLM抽取三元组 → 建图 → 社区检测 → 社区摘要
# ======================================================================

def invoke_with_retry(llm, messages, max_retries=2, wait_seconds=5):
    """
    带重试的LLM调用：应对云端API的瞬时故障（如服务端过载超时）。
    瞬时错误重试通常就能过；重试耗尽才把异常抛出。
    """
    for attempt in range(max_retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as e:
            if attempt >= max_retries:
                raise
            print(f"    [API瞬时故障，{wait_seconds}秒后重试{attempt + 1}/{max_retries}]: {str(e)[:60]}")
            time.sleep(wait_seconds)


def save_graph_cache(all_triples, communities):
    """把三元组和社区摘要缓存到JSON文件"""
    cache_data = {"triples": all_triples, "communities": communities}
    with open(GRAPH_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=1)
    print(f"  图谱缓存已保存: {GRAPH_CACHE_PATH}")


def load_graph_cache():
    """从缓存加载三元组和社区（文件不存在返回None）"""
    if not os.path.exists(GRAPH_CACHE_PATH):
        return None
    with open(GRAPH_CACHE_PATH, "r", encoding="utf-8") as f:
        cache_data = json.load(f)
    return cache_data["triples"], cache_data["communities"]


def extract_triples_from_chunk(chunk_text, chunk_index, control_llm):
    """
    用LLM从一个文本块中抽取实体关系三元组（GraphRAG的核心创意：
    传统知识图谱靠人工+NLP工具抽取，GraphRAG让LLM来做——又快又懂语义）
    :return: 返回三元组列表 [{"subject":..,"relation":..,"object":..}]
    """
    extract_prompt = (
        "从以下文本中抽取实体和它们之间的关系，构建知识图谱三元组。\n"
        "要求：\n"
        "1. 实体是文本中出现的具体对象（行业、公司、技术、业务概念等）；\n"
        "2. 关系用简短的动词或短语表述（如\"面临\"、\"采用\"、\"导致\"、\"属于\"）；\n"
        "3. 只抽取文本明确表达的关系，不要推测；每块最多抽取10条；\n"
        "4. 实体名称使用文本原词，保持简洁（如\"零售业\"而不是\"零售行业的企业们\"）。\n\n"
        f"文本：\n{chunk_text}\n\n"
        '只输出JSON数组，每个元素形如 {"subject":"实体1","relation":"关系","object":"实体2"}。'
    )
    response = control_llm.invoke([("human", extract_prompt)])

    try:
        triples = parse_judge_json(response.content)
    except Exception as e:
        print(f"  块{chunk_index}: 三元组解析失败，跳过 ({e})")
        return []

    # 过滤掉结构不完整的条目
    valid_triples = []
    if isinstance(triples, list):
        for t in triples:
            if isinstance(t, dict) and t.get("subject") and t.get("relation") and t.get("object"):
                triple = {
                    "subject": str(t["subject"]).strip(),
                    "relation": str(t["relation"]).strip(),
                    "object": str(t["object"]).strip(),
                    "source_chunk": chunk_index,  # 记录出处块号（用于local search回溯原文）
                }
                valid_triples.append(triple)
    return valid_triples


def build_knowledge_graph(all_triples):
    """
    用三元组列表构建networkx无向图。
    节点 = 实体；边 = 关系（同一对实体可能有多种关系，存成列表）。
    注意教学简化：无实体对齐（"零售业"和"零售企业"会被当成两个节点）。
    """
    graph = nx.Graph()

    for triple in all_triples:
        subject = triple["subject"]
        obj = triple["object"]

        # 节点带属性：出现次数、出处块集合
        for entity in (subject, obj):
            if entity not in graph:
                graph.add_node(entity, count=0, chunks=set())
            graph.nodes[entity]["count"] += 1
            graph.nodes[entity]["chunks"].add(triple["source_chunk"])

        # 边带属性：关系列表（同一对实体可叠加多种关系）
        if graph.has_edge(subject, obj):
            graph[subject][obj]["relations"].append(triple["relation"])
        else:
            graph.add_edge(subject, obj, relations=[triple["relation"]])

    return graph


def detect_and_summarize_communities(graph, control_llm):
    """
    社区检测 + 社区摘要（GraphRAG索引阶段的精髓）：
    紧密相连的实体自动聚成"社区"，LLM给每个社区写主题摘要——
    这份摘要是Global Search回答全局问题的弹药。
    （教学简化：用贪心模块度算法；生产GraphRAG用Leiden算法+多层社区树）
    :return: 返回 [{name, nodes, summary}] 列表
    """
    communities_raw = nx.community.greedy_modularity_communities(graph)

    communities = []
    for i, community_nodes in enumerate(communities_raw, start=1):
        node_list = list(community_nodes)

        # 该社区内部的关系文本（用于让LLM理解社区主题）
        relation_lines = []
        for subject, obj in graph.edges(node_list):
            if subject in community_nodes and obj in community_nodes:
                relations = "、".join(graph[subject][obj]["relations"])
                relation_lines.append(f"{subject} -[{relations}]- {obj}")
        relations_text = "\n".join(relation_lines[:30])  # 截断防超长

        summary_prompt = (
            "以下是知识图谱中一个实体社区（紧密关联的一组实体）的内部关系。\n"
            "请用一到两句话概括这组实体及其关系反映的核心主题。\n\n"
            f"社区实体：{('、'.join(node_list))[:400]}\n\n"
            f"内部关系：\n{relations_text}\n\n只输出主题概括，不要其他文字。"
        )
        response = control_llm.invoke([("human", summary_prompt)])

        community = {
            "name": f"社区{i}（{len(node_list)}个实体）",
            "nodes": node_list,
            "summary": response.content.strip(),
        }
        communities.append(community)
        print(f"  {community['name']}: {community['summary'][:60]}...")

    return communities


def visualize_graph(graph, communities, image_path):
    """
    把知识图谱画成PNG：节点按社区着色，直观展示关系网结构。
    （中文字体：macOS用系统自带的Arial Unicode MS等，找不到则标签显示为方框）
    """
    # 尝试设置支持中文的字体（macOS常见中文字体列表）
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Hiragino Sans GB", "PingFang SC", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(14, 10))
    # spring_layout：力导向布局——关系紧密的节点自然聚在一起
    pos = nx.spring_layout(graph, seed=42, k=1.2)

    # 节点着色：按所属社区分配不同颜色
    node_colors = []
    color_map = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]
    for node in graph.nodes():
        color_index = 0
        for i, community in enumerate(communities):
            if node in community["nodes"]:
                color_index = i % len(color_map)
                break
        node_colors.append(color_map[color_index])

    # 节点大小与实体出现次数成正比（核心实体更大）
    node_sizes = [150 + graph.nodes[node]["count"] * 80 for node in graph.nodes()]

    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.85)
    nx.draw_networkx_edges(graph, pos, alpha=0.3, edge_color="#7f8c8d")
    nx.draw_networkx_labels(graph, pos, font_size=8)

    plt.title(f"Mini-GraphRAG 知识图谱（{graph.number_of_nodes()}实体 / {graph.number_of_edges()}关系）", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  图谱可视化已保存: {image_path}")


# ======================================================================
# 第二部分：查询——Local Search 与 Global Search
# ======================================================================

def link_entities(query, graph, control_llm):
    """
    实体链接：把自然语言问题映射到图谱中的实体节点。
    （图是符号结构不会"语义相似匹配"，所以需要这一步——
     生产实现常用向量检索找近邻实体，这里用LLM直接选，教学更直观）
    """
    entity_list = list(graph.nodes())
    link_prompt = (
        "以下是知识图谱中的全部实体列表。请选出与问题最相关的1~5个实体"
        "（问题里直接提到的，或回答问题必需的）。\n\n"
        f"实体列表：{('、'.join(entity_list))[:600]}\n\n"
        f"问题：{query}\n\n"
        '只输出JSON数组（实体名字符串列表），不要输出其他文字。'
    )
    response = control_llm.invoke([("human", link_prompt)])

    try:
        selected = parse_judge_json(response.content)
    except Exception:
        selected = []
    # 过滤：只保留图中真实存在的实体名
    linked = [e for e in selected if e in graph]
    return linked


def local_search(query, graph, chunks, control_llm, generator_llm):
    """
    Local Search局部搜索（实体中心问题）：
    实体链接 → 一跳邻居扩展 → 三元组+原文块组装上下文 → 生成
    """
    # 第1步：实体链接
    linked_entities = link_entities(query, graph, control_llm)
    print(f"  [实体链接] 命中: {linked_entities}")

    # 第2步：一跳邻居扩展（在图上"沿边走"——多跳推理的图上实现）
    related_entities = set(linked_entities)
    for entity in linked_entities:
        for neighbor in graph.neighbors(entity):
            related_entities.add(neighbor)

    # 第3步：收集子图中的关系文本
    relation_lines = []
    for subject, obj in graph.edges(related_entities):
        relations = "、".join(graph[subject][obj]["relations"])
        relation_lines.append(f"- {subject} -[{relations}]- {obj}")
    relations_text = "\n".join(relation_lines[:25])

    # 第4步：回溯包含这些实体的原文块（图谱+原文双通道，GraphRAG的标准做法）
    related_chunks = []
    for chunk in chunks:
        for entity in linked_entities:
            if entity in chunk:
                related_chunks.append(chunk)
                break
    original_text = "\n\n".join(related_chunks[:2])

    # 第5步：组装上下文生成（图谱关系是"精炼骨架"，原文是"细节血肉"）
    context = f"【知识图谱关系】\n{relations_text}\n\n【相关原文】\n{original_text}"
    messages = build_prompt_production(query, [context])
    response = invoke_with_retry(generator_llm, messages)
    return response.content


def global_search(query, communities, generator_llm, control_llm):
    """
    Global Search全局搜索（归纳总结问题——向量RAG的死穴）：
    Map：问题对每个社区摘要各答一遍 → Reduce：LLM汇总。
    这是"自底向上汇总"：局部视角拼出全局图景。
    （简化：无原版的部分分数机制；社区少时map轮次少，成本可控）
    """
    # ---- Map阶段：每个社区独立作答 ----
    community_answers = []
    for community in communities:
        map_prompt = (
            f"以下是文档知识图谱中一个实体社区的主题摘要：\n{community['summary']}\n\n"
            f"请从该社区的视角回答问题（1~2句话）；若与本问题无关，回答\"无相关\"。\n"
            f"问题：{query}"
        )
        response = control_llm.invoke([("human", map_prompt)])
        answer = response.content.strip()
        if "无相关" not in answer:
            community_answers.append(f"[{community['name']}] {answer}")

    print(f"  [Map] {len(community_answers)}/{len(communities)}个社区提供了有效答案")

    # ---- Reduce阶段：汇总成最终答案 ----
    if not community_answers:
        return "知识库中没有相关信息。（图RAG全局搜索：各社区均未提供相关信息）"

    all_answers = "\n".join(community_answers)
    reduce_messages = [
        (
            "system",
            "你是知识库问答助手。以下是对同一问题的多份社区视角回答，"
            "请综合它们，给出完整、连贯、有依据的最终回答。先结论后依据。",
        ),
        ("human", f"问题：{query}\n\n各社区回答：\n{all_answers}"),
    ]
    response = invoke_with_retry(generator_llm, reduce_messages)
    return response.content


def vector_rag_baseline(query, collection, embedding_model, bm25, all_docs, tokenizer, reranker_model):
    """对照基线：第6课的完整向量管线（混合+RRF+重排+生产Prompt）"""
    vector_docs = vector_retrieve(query, collection, embedding_model)
    bm25_docs = bm25_retrieve(query, bm25, all_docs)
    fused = rrf_fusion([vector_docs, bm25_docs])
    fused_docs = [doc for doc, score in fused[:8]]
    from lesson6_hybrid_rerank import rerank
    reranked_docs, _ = rerank(query, fused_docs, tokenizer, reranker_model)

    messages = build_prompt_production(query, reranked_docs[:3])
    llm = get_llm("generator")
    return invoke_with_retry(llm, messages).content


# ======================================================================
# 第三部分：主流程
# ======================================================================

def main():
    if not os.environ.get("DEEPSEEK_API_KEY") or not os.environ.get("ZHIPU_API_KEY"):
        raise SystemExit("需要同时配置 DEEPSEEK_API_KEY 和 ZHIPU_API_KEY")

    print("第10课收官：手写mini-GraphRAG\n" + "=" * 70)
    control_llm = get_llm("control")
    generator_llm = get_llm("generator")

    # ---------- 索引阶段 ----------
    print("\n【索引阶段】构建知识图谱（GraphRAG成本的主要来源）")

    # 原文分块（抽取用大块；local search回溯原文也用同一批块）
    document_text = load_document(SOURCE_TXT)
    splitter = RecursiveCharacterTextSplitter(chunk_size=EXTRACT_CHUNK_SIZE, chunk_overlap=0)
    chunks = splitter.split_text(document_text)
    print(f"文档分为{len(chunks)}个抽取块")

    # ★缓存优先：索引一次、查询多次（GraphRAG的核心理念）
    cached = load_graph_cache()
    if cached is not None:
        all_triples, communities = cached
        graph = build_knowledge_graph(all_triples)
        print(f"从缓存加载图谱: {len(all_triples)}条三元组, "
              f"{graph.number_of_nodes()}实体/{graph.number_of_edges()}关系, "
              f"{len(communities)}个社区（跳过LLM抽取与摘要）")
    else:
        # 第2步：LLM逐块抽取三元组
        all_triples = []
        for i, chunk in enumerate(chunks, start=1):
            print(f"  抽取块{i}/{len(chunks)}...", end="")
            triples = extract_triples_from_chunk(chunk, i, control_llm)
            all_triples.extend(triples)
            print(f" 得到{len(triples)}条三元组")
        print(f"共抽取{len(all_triples)}条三元组")

        # 第3步：建图
        graph = build_knowledge_graph(all_triples)
        print(f"知识图谱构建完成: {graph.number_of_nodes()}个实体, {graph.number_of_edges()}条关系")

        # 第4步：社区检测与摘要
        print("\n社区检测与摘要:")
        communities = detect_and_summarize_communities(graph, control_llm)

        # 保存缓存（下次运行直接复用）
        save_graph_cache(all_triples, communities)

    # 第5步：可视化
    visualize_graph(graph, communities, GRAPH_IMAGE_PATH)

    # ---------- 查询阶段：图RAG vs 向量RAG 对照 ----------
    # 准备向量基线的组件
    client = chromadb.PersistentClient(str(CHROMA_DB_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    embedding_model = load_embedding_model()
    bm25, all_docs, all_metadatas = build_bm25_index(collection)
    tokenizer, reranker_model = load_reranker()

    # ---- 对照1：实体中心问题（Local Search的主场） ----
    local_query = "零售业案例公司采用了哪些数字化转型措施？"
    print(f"\n\n{'█' * 70}\n█ 对照1（实体中心问题）: {local_query}\n{'█' * 70}")

    print("\n► 图RAG Local Search:")
    graph_local_answer = local_search(local_query, graph, chunks, control_llm, generator_llm)
    print(f"答案: {graph_local_answer[:300]}...")

    print("\n► 向量RAG基线（第6课管线）:")
    vector_answer_1 = vector_rag_baseline(
        local_query, collection, embedding_model, bm25, all_docs, tokenizer, reranker_model
    )
    print(f"答案: {vector_answer_1[:300]}...")

    # ---- 对照2：全局归纳问题（Global Search的主场，向量RAG的死穴） ----
    global_query = "这份报告讨论的数字化转型有哪些核心主题和整体价值？"
    print(f"\n\n{'█' * 70}\n█ 对照2（全局归纳问题）: {global_query}\n{'█' * 70}")

    print("\n► 图RAG Global Search:")
    graph_global_answer = global_search(global_query, communities, generator_llm, control_llm)
    print(f"答案: {graph_global_answer}")

    print("\n► 向量RAG基线（第6课管线）:")
    vector_answer_2 = vector_rag_baseline(
        global_query, collection, embedding_model, bm25, all_docs, tokenizer, reranker_model
    )
    print(f"答案: {vector_answer_2[:400]}...")

    # ---------- 收官说明 ----------
    print(f"\n\n{'=' * 70}")
    print("十课全部完成。观察要点：")
    print("1. 打开图谱图片观察社区结构: " + GRAPH_IMAGE_PATH)
    print("2. 对照1：两种方法答实体问题都不错（向量RAG的基本盘）")
    print("3. 对照2：全局归纳问题上，社区摘要的Map-Reduce给出'俯瞰视角'，")
    print("   而向量基线只能拼接Top-3局部片段——这正是GraphRAG的设计目标")
    print("4. 记得权衡：图RAG为这次索引付了约20次LLM调用，向量RAG零额外成本")


if __name__ == "__main__":
    main()
