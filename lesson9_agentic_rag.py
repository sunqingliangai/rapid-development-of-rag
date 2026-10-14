"""
第9课实战：Agentic RAG——用LangGraph实现Corrective RAG（2026-10版）

范式升级（本课与第2~8课的本质区别）：
    管道式RAG：流程由代码写死（检索→生成，每次都一样）
    Agentic RAG：流程中出现"判断节点"，由LLM在运行时决定走哪条路

本图实现的Corrective RAG流程（对应论文arXiv 2401.15884）：

              START
                │
                ▼
           ┌──────────┐
           │ retrieve │  混合检索+RRF+重排（复用第6课组件）
           └────┬─────┘
                ▼
           ┌──────────┐   相关
           │  grade   │──────────► generate ──► END
           │ (LLM评估) │                    （生产级Prompt生成答案）
           └────┬─────┘
                │ 不相关
                ▼
           ┌──────────┐   重试次数未用尽
           │ rewrite  │──────────► (回到retrieve，用改写后的查询重新检索)
           │ (LLM改写) │
           └────┬─────┘
                │ 重试次数用尽
                ▼
           ┌──────────┐
           │ fallback │  诚实拒答（不编造）
           └──────────┘──► END

角色分工（控制面/数据面分离，延续第8课"选手与裁判分家"的纪律）：
    控制面（判断/改写）= glm-4.5-flash（便宜，负责流程决策）
    数据面（最终生成） = deepseek-flash（负责产出答案）

运行方式：
    uv run --env-file .env python lesson9_agentic_rag.py
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
from lesson3_multiformat_rag import load_embedding_model  # noqa: E402

import chromadb  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from typing import TypedDict  # noqa: E402

# ---------- 配置区 ----------
MAX_RETRIES = 2  # 改写重检的最大次数（用尽后走fallback拒答）

# 控制面模型（grade评估/rewrite改写）：GLM
CONTROL_MODEL = os.environ.get("ZHIPU_MODEL", "glm-4.5-flash")
CONTROL_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

# 数据面模型（最终生成）：DeepSeek
GENERATOR_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
GENERATOR_BASE_URL = "https://api.deepseek.com"

# 三个演示查询：分别走图的三条不同路径
DEMO_QUERIES = [
    # 路径1：一次通过（retrieve→grade→generate）
    "零售业数字化转型面临哪些挑战？",
    # 路径2：比喻式表述——首轮检索可能失败，看改写能否救回
    "线下门店和网上商城为什么总是打架？",
    # 路径3：知识库范围外——应触发完整纠错环后诚实拒答
    "这家公司的CEO是谁？",
]


def get_control_llm():
    """控制面模型客户端：负责判断和改写（决策角色）"""
    return ChatOpenAI(
        model=CONTROL_MODEL,
        api_key=os.environ["ZHIPU_API_KEY"],
        base_url=CONTROL_BASE_URL,
        temperature=0.1,  # 判断任务需要稳定，低温
    )


# ======================================================================
# 第一部分：状态定义——图中所有节点共享的数据结构
# ======================================================================

class RAGState(TypedDict):
    """
    LangGraph的State：一张共享的"工作台"，
    每个节点读取所需字段、写回更新字段，图引擎负责在节点间传递。
    """
    question: str          # 当前使用的查询（可能已被改写过）
    original_question: str # 用户的原始查询（改写后仍要记住，最终回答它）
    chunks: list           # 当前检索到的文本块
    grade_pass: bool       # 评估结论：检索结果是否与问题相关
    retries: int           # 已改写重检的次数
    answer: str            # 最终答案
    path_log: list         # 流程轨迹日志：记录实际经过了哪些节点——
                           # 亲眼看到"流程由模型在运行时决定"


# ======================================================================
# 第二部分：节点函数（工厂模式：闭包持有检索管线等重组件）
# ======================================================================

def build_graph(collection, embedding_model, bm25, all_docs, tokenizer, reranker_model):
    """
    构建Corrective RAG的StateGraph。
    用工厂函数而非全局变量：节点以闭包方式持有检索管线组件，
    组件只在建图时加载一次，之后每次invoke直接复用。
    """
    control_llm = get_control_llm()

    # ---------- 节点1：retrieve（检索） ----------
    def retrieve_node(state: RAGState) -> RAGState:
        """混合检索+RRF+重排，取Top-3（完整复用第6课管线）"""
        query = state["question"]

        vector_docs = vector_retrieve(query, collection, embedding_model)
        bm25_docs = bm25_retrieve(query, bm25, all_docs)
        fused = rrf_fusion([vector_docs, bm25_docs])

        fused_docs = []
        for doc_text, rrf_score in fused[:8]:
            fused_docs.append(doc_text)

        from lesson6_hybrid_rerank import rerank
        reranked_docs, reranked_scores = rerank(query, fused_docs, tokenizer, reranker_model)

        state["chunks"] = reranked_docs[:3]
        state["path_log"].append(f"retrieve(检索:{query[:24]}...)")
        print(f"  [retrieve] 使用查询「{query[:30]}」检索到{len(state['chunks'])}个文本块")
        return state

    # ---------- 节点2：grade（评估）——本图的"判断节点" ----------
    def grade_node(state: RAGState) -> RAGState:
        """
        控制面LLM判断：检索到的资料是否包含回答问题所需的信息。
        这就是Agentic RAG的核心动作——模型在流程中做决策。
        """
        query = state["question"]
        chunks = state["chunks"]

        # 每块截断，评估不需要全文
        chunks_text = ""
        for i, chunk in enumerate(chunks, start=1):
            chunks_text += f"[{i}] {chunk[:150]}\n"

        grade_prompt = (
            "你是检索质量评估员。判断以下检索资料是否包含回答问题所需的信息。\n"
            "判定标准：资料中有能直接用于回答问题的内容=是；资料与问题无关或"
            "不包含所需信息=否。\n\n"
            f"问题：{query}\n\n检索资料：\n{chunks_text}\n"
            "只输出\"是\"或\"否\"，不要输出其他文字。"
        )
        response = control_llm.invoke([("human", grade_prompt)])
        verdict = response.content.strip()

        state["grade_pass"] = "是" in verdict
        state["path_log"].append(f"grade(评估:{'通过' if state['grade_pass'] else '不通过'})")
        print(f"  [grade] 评估结论: {'✅ 相关，可生成' if state['grade_pass'] else '❌ 不相关'}")
        return state

    # ---------- 节点3：rewrite（改写）——纠错环的关键 ----------
    def rewrite_node(state: RAGState) -> RAGState:
        """
        控制面LLM从不同角度改写查询，然后图会回到retrieve重新检索。
        （对应第8课讲的"查询改写"技术——区别是：这里由评估结果动态触发，
         而不是每次查询都无脑执行。这正是Advanced与Agentic的分界线。）
        """
        original_question = state["original_question"]

        rewrite_prompt = (
            "以下是用户对知识库的提问，但检索没有找到相关资料。"
            "请从不同角度改写这个问题，换成更直白、更贴近文档常见表述的问法，"
            "使其更容易检索到相关资料。只输出改写后的问题，不要其他文字。\n\n"
            f"原始问题：{original_question}"
        )
        response = control_llm.invoke([("human", rewrite_prompt)])
        new_question = response.content.strip().strip('"')

        state["question"] = new_question
        state["retries"] += 1
        state["path_log"].append(f"rewrite(第{state['retries']}次改写)")
        print(f"  [rewrite] 第{state['retries']}次改写 → 「{new_question[:40]}」")
        return state

    # ---------- 节点4：generate（生成）——数据面 ----------
    def generate_node(state: RAGState) -> RAGState:
        """用生产级Prompt+DeepSeek生成最终答案（复用第7课模板）"""
        query = state["original_question"]  # 回答用户的原始问题（不是改写版）
        chunks = state["chunks"]

        messages = build_prompt_production(query, chunks)
        llm = ChatOpenAI(
            model=GENERATOR_MODEL,
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=GENERATOR_BASE_URL,
            temperature=0.1,
        )
        response = llm.invoke(messages)

        state["answer"] = response.content
        state["path_log"].append("generate(生成答案)")
        print("  [generate] 答案已生成")
        return state

    # ---------- 节点5：fallback（兜底拒答） ----------
    def fallback_node(state: RAGState) -> RAGState:
        """重试用尽仍无相关资料：诚实拒答，绝不编造（呼应第7课拒答出口）"""
        state["answer"] = (
            "知识库中没有相关信息。"
            "（系统已尝试从多个角度检索并改写查询，仍未找到与您问题相关的资料）"
        )
        state["path_log"].append("fallback(诚实拒答)")
        print("  [fallback] 重试用尽，诚实拒答")
        return state

    # ---------- 条件路由：grade之后走哪条边（流程分叉点） ----------
    def route_after_grade(state: RAGState) -> str:
        """
        LangGraph的条件边（conditional edge）：
        返回值是下一个节点的名字——图引擎据此决定流向。
        三种走向：相关→生成；不相关且还有重试→改写；不相关且重试用尽→拒答
        """
        if state["grade_pass"]:
            return "generate"
        if state["retries"] >= MAX_RETRIES:
            return "fallback"
        return "rewrite"

    # ---------- 组装图 ----------
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generate_node)
    graph.add_node("fallback", fallback_node)

    # 固定边：入口→retrieve→grade
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    # 循环边：改写后回到检索（图可以有环——这就是重试的实现方式）
    graph.add_edge("rewrite", "retrieve")
    # 出口边
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)
    # 条件边：grade之后动态分流（本图的灵魂）
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {"generate": "generate", "rewrite": "rewrite", "fallback": "fallback"},
    )

    # compile()把图编译成可执行的应用（之后可反复invoke）
    return graph.compile()


# ======================================================================
# 第三部分：主流程
# ======================================================================

def main():
    # 检查API Key
    if not os.environ.get("DEEPSEEK_API_KEY") or not os.environ.get("ZHIPU_API_KEY"):
        raise SystemExit("需要同时配置 DEEPSEEK_API_KEY 和 ZHIPU_API_KEY")

    print("第9课：Agentic RAG（Corrective RAG / LangGraph）\n" + "=" * 70)

    # ---------- 准备检索管线组件（一次加载，闭包复用） ----------
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

    # ---------- 建图 ----------
    app = build_graph(collection, embedding_model, bm25, all_docs, tokenizer, reranker_model)

    # ---------- 三个演示查询依次执行 ----------
    for q_index, query in enumerate(DEMO_QUERIES):
        print(f"\n\n{'█' * 70}\n█ 查询{q_index + 1}: {query}\n{'█' * 70}")

        # 初始化状态（所有字段显式给初值，避免歧义）
        initial_state = {
            "question": query,
            "original_question": query,
            "chunks": [],
            "grade_pass": False,
            "retries": 0,
            "answer": "",
            "path_log": [],
        }

        # invoke驱动图执行：图引擎按边（含条件边）调度节点，直到到达END
        final_state = app.invoke(initial_state)

        # ---------- 展示流程轨迹（本课最重要的观察对象） ----------
        print(f"\n  实际执行路径: {' → '.join(final_state['path_log'])}")
        print(f"\n  最终答案:\n  {'-' * 60}")
        answer_text = final_state["answer"]
        # 缩进展示，最多600字符
        for line in answer_text[:600].split("\n"):
            print(f"  {line}")
        print(f"  {'-' * 60}")

    print("\n" + "=" * 70)
    print("观察要点：三个查询走出了三条不同的路径——")
    print("1. 简单问题直达生成；2. 比喻式问题可能触发改写重检；")
    print("3. 范围外问题走完整纠错环后诚实拒答。")
    print("同样的图结构，不同的问题走不同的路——这就是Agentic RAG。")


if __name__ == "__main__":
    main()
