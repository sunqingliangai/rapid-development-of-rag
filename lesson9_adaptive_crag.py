"""
第9课进阶实战：Adaptive + CRAG 组合体（LangGraph，2026-10版）

在 lesson9_agentic_rag.py（Corrective RAG）基础上，按 Adaptive-RAG 论文思想
增加"路由节点"：由LLM在入口处判断问题复杂度，动态选择处理管道。

图的完整结构：

                 START
                   │
            ┌──────▼──────┐
            │ route(路由)  │  LLM分类问题复杂度（Adaptive-RAG的核心动作）
            └──┬────┬────┬─┘
       闲聊│   简单│   复杂│
           ▼      ▼      ▼
    ┌─────────┐ ┌────────┐ ┌───────────┐
    │chat_answer│ │retrieve│ │decompose  │  LLM把复杂问题拆成子问题
    └────┬────┘ └───┬────┘ └─────┬─────┘
         │          ▼            ▼
         │      ┌────────┐  ┌──────────────┐
         │      │ grade  │  │ sub_retrieve  │  逐子问题检索+评估，
         │      └───┬────┘  │ (节点内循环)   │  汇总通过评估的块
         │   通过│  │不通过  └──────┬───────┘
         │       ▼  │        有通过│    全部失败
         │  ┌────────┐│            ▼        ▼
         │  │generate││        generate   fallback
         │  └────────┘│
         │     ▲      │重试未用尽
         │     │      ▼
         │     │  ┌────────┐
         │     └──│rewrite │──►(回到retrieve，≤2次)
         │        └────────┘
         │            │重试用尽
         │            ▼
         │        ┌────────┐
         │        │fallback│
         │        └────────┘
         ▼          ▼        ▼
        END        END      END

三条管道的分工：
    闲聊 → 完全跳过检索（省成本省延迟）——"按难度付费"
    简单 → 单轮CRAG（第9课的完整纠错环）
    复杂 → 拆解成子问题，逐个检索评估，汇总后生成（多跳思想的落地）

运行方式：
    uv run --env-file .env python lesson9_adaptive_crag.py
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
from lesson8_evaluation import parse_judge_json  # noqa: E402  (复用JSON容错解析)

import chromadb  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from typing import TypedDict  # noqa: E402

# ---------- 配置区 ----------
MAX_RETRIES = 2  # 简单管道的改写重检次数上限

CONTROL_MODEL = os.environ.get("ZHIPU_MODEL", "glm-4.5-flash")
CONTROL_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
GENERATOR_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
GENERATOR_BASE_URL = "https://api.deepseek.com"

# 四个演示查询：分别走图的四条路径
DEMO_QUERIES = [
    "你好！简单介绍一下你自己，你是做什么的？",            # 闲聊→直接回答
    "零售业数字化转型面临哪些挑战？",                      # 简单→单轮CRAG
    "这家公司的CEO是谁？",                                # 简单(范围外)→纠错环→fallback
    "三个行业案例各自的核心转型措施是什么？它们面临的挑战有什么共同点？",  # 复杂→拆解
]


# ======================================================================
# 第一部分：状态定义
# ======================================================================

class RAGState(TypedDict):
    """图引擎在节点间传递的共享状态（第9课版基础上新增3个字段）"""
    question: str          # 当前查询（简单管道中可能被改写）
    original_question: str # 用户的原始问题
    complexity: str        # ★新增：路由结论（闲聊/简单/复杂）
    sub_questions: list    # ★新增：复杂问题拆解出的子问题列表
    chunks: list           # 当前检索到的文本块
    grade_pass: bool       # 评估结论
    retries: int           # 改写重检次数
    answer: str            # 最终答案
    path_log: list         # 流程轨迹


# ======================================================================
# 第二部分：图构建（工厂闭包持有重组件）
# ======================================================================

def build_graph(collection, embedding_model, bm25, all_docs, tokenizer, reranker_model):
    """构建 Adaptive + CRAG 组合图"""
    control_llm = ChatOpenAI(
        model=CONTROL_MODEL,
        api_key=os.environ["ZHIPU_API_KEY"],
        base_url=CONTROL_BASE_URL,
        temperature=0.1,
    )

    # ---------- 通用小工具：对一批chunk做相关性评估（grade_node和sub_retrieve共用） ----------
    def grade_chunks(query, chunks, require_sufficient=True):
        """
        控制面LLM判断检索资料与问题的关系。返回True/False。
        两档判定标准（★重要区别，子问题聚合翻车的教训）：
        - require_sufficient=True（严格档）：资料是否"足够完整回答"问题
          → 用于简单管道：资料不完整就生成必然翻车，该卡就卡
        - require_sufficient=False（宽松档）：资料是否"包含相关信息"
          → 用于复杂管道的子问题：子问题只是拼图的一块，单块资料
            不需要"完整回答"它，包含有用信息就该放行——充分性由
            最终生成环节综合判断，不该在半路卡死
        """
        chunks_text = ""
        for i, chunk in enumerate(chunks, start=1):
            chunks_text += f"[{i}] {chunk[:150]}\n"

        if require_sufficient:
            criterion = (
                "判定标准：资料中有能直接用于回答问题的内容=是；"
                "资料与问题无关或不包含所需信息=否。"
            )
        else:
            criterion = (
                "判定标准：资料中包含与问题相关的有用信息=是；"
                "资料与问题完全无关=否。注意：不要求资料能完整回答问题，"
                "只要包含相关信息就算\"是\"。"
            )

        grade_prompt = (
            "你是检索质量评估员。判断以下检索资料与问题的关系。\n"
            f"{criterion}\n\n"
            f"问题：{query}\n\n检索资料：\n{chunks_text}\n"
            "只输出\"是\"或\"否\"，不要输出其他文字。"
        )
        response = control_llm.invoke([("human", grade_prompt)])
        return "是" in response.content.strip()

    # ---------- 通用小工具：完整检索（复用第6课管线） ----------
    def retrieve_full(query, top_k=3):
        """混合检索+RRF+重排，返回Top-K文本块"""
        vector_docs = vector_retrieve(query, collection, embedding_model)
        bm25_docs = bm25_retrieve(query, bm25, all_docs)
        fused = rrf_fusion([vector_docs, bm25_docs])

        fused_docs = []
        for doc_text, rrf_score in fused[:8]:
            fused_docs.append(doc_text)

        from lesson6_hybrid_rerank import rerank
        reranked_docs, reranked_scores = rerank(query, fused_docs, tokenizer, reranker_model)
        return reranked_docs[:top_k]

    # ================================================================
    # 节点1：route（路由）——本图的新灵魂，Adaptive-RAG的核心动作
    # ================================================================
    def route_node(state: RAGState) -> RAGState:
        """
        LLM判断问题复杂度，三选一：
        - 闲聊：问候、常识闲聊，与知识库文档内容无关
        - 简单：单一主题的事实/列表问题，一次检索可以覆盖
        - 复杂：涉及多个主题、多方面对比或综合分析，需要拆解
        """
        query = state["question"]
        route_prompt = (
            "你是RAG系统的查询路由器。知识库内容是《数字化转型》报告"
            "（含制造业/零售业/金融业三个行业案例）。\n"
            "请判断用户问题的类型，三选一：\n"
            "- 闲聊：问候、自我介绍请求、与知识库内容无关的常识问题\n"
            "- 简单：单一主题的事实型问题，一次检索即可回答\n"
            "- 复杂：需要综合多个主题/多个方面信息才能回答的分析对比型问题\n\n"
            f"用户问题：{query}\n\n"
            "只输出\"闲聊\"、\"简单\"或\"复杂\"，不要输出其他文字。"
        )
        response = control_llm.invoke([("human", route_prompt)])
        verdict = response.content.strip()

        # 从输出中识别分类（容错：输出可能带标点或解释文字）
        if "闲聊" in verdict:
            state["complexity"] = "闲聊"
        elif "复杂" in verdict:
            state["complexity"] = "复杂"
        else:
            state["complexity"] = "简单"

        state["path_log"].append(f"route(分类:{state['complexity']})")
        print(f"  [route] 问题分类: {state['complexity']}")
        return state

    # ================================================================
    # 闲聊管道：直接回答，完全跳过检索
    # ================================================================
    def chat_answer_node(state: RAGState) -> RAGState:
        """闲聊问题直接用通用知识回答——省下整个检索管线的成本和延迟"""
        query = state["original_question"]
        messages = [
            (
                "system",
                "你是知识库问答助手。当前问题不需要查询知识库，"
                "请直接用你的通用知识友好、简洁地回答。",
            ),
            ("human", query),
        ]
        llm = ChatOpenAI(
            model=GENERATOR_MODEL,
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=GENERATOR_BASE_URL,
            temperature=0.5,  # 闲聊可以稍微放开
        )
        response = llm.invoke(messages)

        state["answer"] = response.content
        state["path_log"].append("chat_answer(直接回答)")
        print("  [chat_answer] 未检索，直接回答")
        return state

    # ================================================================
    # 简单管道：第9课的CRAG（retrieve → grade → generate/rewrite/fallback）
    # ================================================================
    def retrieve_node(state: RAGState) -> RAGState:
        """混合检索+RRF+重排（复用第6课管线）"""
        state["chunks"] = retrieve_full(state["question"])
        state["path_log"].append(f"retrieve(检索:{state['question'][:20]}...)")
        print(f"  [retrieve] 检索到{len(state['chunks'])}个文本块")
        return state

    def grade_node(state: RAGState) -> RAGState:
        """评估检索结果与问题的相关性（CRAG的判断节点）"""
        state["grade_pass"] = grade_chunks(state["question"], state["chunks"])
        state["path_log"].append(f"grade(评估:{'通过' if state['grade_pass'] else '不通过'})")
        print(f"  [grade] 评估结论: {'✅ 相关' if state['grade_pass'] else '❌ 不相关'}")
        return state

    def rewrite_node(state: RAGState) -> RAGState:
        """改写查询后回到retrieve重检（纠错环）"""
        rewrite_prompt = (
            "以下是用户对知识库的提问，但检索没有找到相关资料。"
            "请从不同角度改写这个问题，换成更直白、更贴近文档常见表述的问法，"
            "使其更容易检索到相关资料。只输出改写后的问题，不要其他文字。\n\n"
            f"原始问题：{state['original_question']}"
        )
        response = control_llm.invoke([("human", rewrite_prompt)])

        state["question"] = response.content.strip().strip('"')
        state["retries"] += 1
        state["path_log"].append(f"rewrite(第{state['retries']}次改写)")
        print(f"  [rewrite] 第{state['retries']}次改写 → 「{state['question'][:40]}」")
        return state

    # ================================================================
    # 复杂管道：decompose → sub_retrieve → generate/fallback
    # ================================================================
    def decompose_node(state: RAGState) -> RAGState:
        """
        LLM把复杂问题拆成2~4个子问题（对应第8课讲的"子问题分解"）。
        例："三个行业的措施和共同挑战" → ["制造业的措施？"，"零售业的措施？"，...]
        """
        decompose_prompt = (
            "把下面的复杂问题拆解为2~5个各自独立的子问题。\n"
            "拆解要求（★重要）：\n"
            "1. 每个子问题只涉及【单一主题】——如果原问题涉及多个对象"
            "（如多个行业/多个公司），每个对象单独出一个子问题；\n"
            "2. 子问题必须使用原问题中的【实际对象名称】——"
            "禁止使用\"行业A\"\"公司B\"之类的代称，代称无法用于检索；\n"
            "3. 子问题合起来能覆盖原问题的全部信息需求。\n\n"
            "示例：原问题\"制造业和服务业的转型措施有什么异同？\"应拆解为：\n"
            "[\"制造业的转型措施是什么？\", \"服务业的转型措施是什么？\", "
            "\"制造业和服务业的转型措施有什么异同？\"]\n\n"
            f"原问题：{state['original_question']}\n\n"
            '只输出JSON数组（子问题字符串列表），不要输出其他文字。'
        )
        response = control_llm.invoke([("human", decompose_prompt)])

        sub_questions = parse_judge_json(response.content)
        # 防御：解析失败时退化为"原问题当单个子问题处理"
        if not isinstance(sub_questions, list) or len(sub_questions) == 0:
            sub_questions = [state["original_question"]]

        state["sub_questions"] = sub_questions
        state["path_log"].append(f"decompose(拆解为{len(sub_questions)}个子问题)")
        print(f"  [decompose] 拆解结果:")
        for i, sub_q in enumerate(sub_questions, start=1):
            print(f"    子问题{i}: {sub_q}")
        return state

    def sub_retrieve_node(state: RAGState) -> RAGState:
        """
        逐个子问题：完整检索 → 相关性评估。
        通过评估的块汇总（按文本去重——同一块回答多个子问题只保留一份）。
        注：这里在节点内部用普通Python循环处理子问题，教学上更直观；
        生产中可用LangGraph的Send API把子问题并行分发到子图。
        """
        aggregated_chunks = []  # 汇总列表
        seen_texts = set()      # 去重集合（文本内容作为键）
        passed_count = 0        # 通过评估的子问题计数

        for sub_q in state["sub_questions"]:
            sub_chunks = retrieve_full(sub_q)
            # 宽松档评估：子问题只求"包含相关信息"，不求"完整回答"（见grade_chunks注释）
            is_relevant = grade_chunks(sub_q, sub_chunks, require_sufficient=False)
            status = "✅" if is_relevant else "❌"
            print(f"    子问题「{sub_q[:30]}」评估: {status}")

            if is_relevant:
                passed_count += 1
                for chunk in sub_chunks:
                    # 按文本内容去重（同一块被多个子问题命中只保留一份）
                    if chunk not in seen_texts:
                        seen_texts.add(chunk)
                        aggregated_chunks.append(chunk)

        state["chunks"] = aggregated_chunks
        state["grade_pass"] = passed_count > 0  # 至少一个子问题通过才算成功
        state["path_log"].append(
            f"sub_retrieve({passed_count}/{len(state['sub_questions'])}个子问题通过,汇总{len(aggregated_chunks)}块)"
        )
        print(f"  [sub_retrieve] 汇总完成: {len(aggregated_chunks)}个文本块")
        return state

    # ================================================================
    # 公共出口节点：generate / fallback
    # ================================================================
    def generate_node(state: RAGState) -> RAGState:
        """生产级Prompt+DeepSeek生成最终答案（两条管道共用）"""
        query = state["original_question"]
        chunks = state["chunks"]
        # 复杂问题汇总的块可能较多，最多取6块防止上下文过长
        chunks = chunks[:6]

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

    def fallback_node(state: RAGState) -> RAGState:
        """重试用尽仍无相关资料：诚实拒答"""
        state["answer"] = (
            "知识库中没有相关信息。"
            "（系统已尝试从多个角度检索并改写查询，仍未找到与您问题相关的资料）"
        )
        state["path_log"].append("fallback(诚实拒答)")
        print("  [fallback] 诚实拒答")
        return state

    # ---------- 条件路由函数们 ----------
    def route_after_router(state: RAGState) -> str:
        """路由节点之后的三分叉（Adaptive）"""
        return {"闲聊": "chat_answer", "简单": "retrieve", "复杂": "decompose"}[state["complexity"]]

    def route_after_grade(state: RAGState) -> str:
        """简单管道grade之后的三分叉（CRAG）"""
        if state["grade_pass"]:
            return "generate"
        if state["retries"] >= MAX_RETRIES:
            return "fallback"
        return "rewrite"

    def route_after_sub_retrieve(state: RAGState) -> str:
        """复杂管道：至少一个子问题通过→生成；全军覆没→拒答"""
        if state["grade_pass"]:
            return "generate"
        return "fallback"

    # ---------- 组装图 ----------
    graph = StateGraph(RAGState)
    graph.add_node("route", route_node)
    graph.add_node("chat_answer", chat_answer_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("decompose", decompose_node)
    graph.add_node("sub_retrieve", sub_retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("fallback", fallback_node)

    # 入口 → 路由
    graph.add_edge(START, "route")
    # 路由三岔口（条件边）
    graph.add_conditional_edges(
        "route", route_after_router,
        {"chat_answer": "chat_answer", "retrieve": "retrieve", "decompose": "decompose"},
    )
    # 闲聊管道出口
    graph.add_edge("chat_answer", END)
    # 简单管道（CRAG环）
    graph.add_edge("retrieve", "grade")
    graph.add_edge("rewrite", "retrieve")
    graph.add_conditional_edges(
        "grade", route_after_grade,
        {"generate": "generate", "rewrite": "rewrite", "fallback": "fallback"},
    )
    # 复杂管道
    graph.add_edge("decompose", "sub_retrieve")
    graph.add_conditional_edges(
        "sub_retrieve", route_after_sub_retrieve,
        {"generate": "generate", "fallback": "fallback"},
    )
    # 公共出口
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)

    return graph.compile()


# ======================================================================
# 第三部分：主流程
# ======================================================================

def main():
    if not os.environ.get("DEEPSEEK_API_KEY") or not os.environ.get("ZHIPU_API_KEY"):
        raise SystemExit("需要同时配置 DEEPSEEK_API_KEY 和 ZHIPU_API_KEY")

    print("第9课进阶：Adaptive + CRAG 组合体\n" + "=" * 70)

    # ---------- 准备组件 ----------
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

    # ---------- 建图并执行 ----------
    app = build_graph(collection, embedding_model, bm25, all_docs, tokenizer, reranker_model)

    for q_index, query in enumerate(DEMO_QUERIES):
        print(f"\n\n{'█' * 70}\n█ 查询{q_index + 1}: {query}\n{'█' * 70}")

        initial_state = {
            "question": query,
            "original_question": query,
            "complexity": "",
            "sub_questions": [],
            "chunks": [],
            "grade_pass": False,
            "retries": 0,
            "answer": "",
            "path_log": [],
        }
        final_state = app.invoke(initial_state)

        print(f"\n  实际执行路径: {' → '.join(final_state['path_log'])}")
        print(f"\n  最终答案:\n  {'-' * 60}")
        for line in final_state["answer"][:600].split("\n"):
            print(f"  {line}")
        print(f"  {'-' * 60}")

    print("\n" + "=" * 70)
    print("观察要点：一个入口，四条路径（闲聊直答/简单直达/简单纠错/复杂拆解）。")
    print("路由节点花一次廉价LLM调用，为每个问题选择最合适的处理成本——")
    print("这就是Adaptive-RAG'按难度付费'的工程价值。")


if __name__ == "__main__":
    main()
