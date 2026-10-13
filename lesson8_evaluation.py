"""
第8课实验：RAG效果评估——手写mini-RAGAS + 双系统对比 + 裁判偏差实验

实验设计（对应第8课讲解的三部分）：

1. 评测集：8个问题（5个可回答 + 2个不可回答 + 1个多跳综合），
   每题带参考答案——评估体系的地基。
2. 手写mini-RAGAS三个指标（不装RAGAS库，看清每个指标的计算本质）：
   - CR 检索相关性(Context Relevancy)：裁判对每个检索块打分(1/0.5/0)取均值
   - F  忠实度(Faithfulness)：答案拆原子论断 → 逐条判断是否被context支持 → 支持率
   - AR 答案相关性(Answer Relevancy)：RAGAS原版机制——让裁判从答案反向生成
     3个问题，用本地Embedding计算与原问题的平均余弦相似度
   - 附加：拒答正确率（不可回答问题的专属指标，关键词规则判定）
3. 双系统对比（控制变量：同一生产级Prompt、同一生成模型，只变检索）：
   - 系统A(朴素)：纯向量检索Top-3（第2课的检索方式）
   - 系统B(完整)：混合检索+RRF融合+重排Top-3（第6课的检索方式）
   → 量化"前六课的检索优化到底值多少分"
4. 裁判偏差实验：同一对答案交换位置让裁判二选一，观察"位置偏差"是否出现

角色分配（规避LLM-as-judge的自我偏好偏差）：
   生成模型 = deepseek-flash（选手）
   裁判模型 = glm-4.5-flash（裁判，与选手不同厂商）

运行方式：
    uv run --env-file .env python lesson8_evaluation.py
"""

import os

# 必须在导入 sentence-transformers 之前设置（原因见第2课）
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ⚠️导入顺序说明（第7课踩过的坑）：
# 先导入第6课模块（按已验证顺序带入torch/transformers），再导入其他原生库包
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

import json  # noqa: E402
import chromadb  # noqa: E402
import numpy as np  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

# ---------- 模型配置 ----------
TEMPERATURE = 0.1  # 生成与裁判都用低温：事实判断任务要稳定

# 生成模型（选手）：DeepSeek
GENERATOR_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
GENERATOR_BASE_URL = "https://api.deepseek.com"

# 裁判模型（judge）：智谱GLM（与选手不同厂商，规避自我偏好偏差）
JUDGE_MODEL = os.environ.get("ZHIPU_MODEL", "glm-4.5-flash")
JUDGE_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

# 拒答判定的关键词（用于"拒答正确率"指标；生产中应结合裁判判定更稳）
REFUSAL_KEYWORDS = [
    "知识库中没有", "无法回答", "未提及", "没有相关", "无法根据",
    "没有提供", "没有提到", "无法确定", "未包含",
]

# ======================================================================
# 第一部分：评测集（8题，覆盖三种题型）
# ======================================================================

TEST_DATASET = [
    # ---- 可回答题×5 ----
    {
        "question": "零售业数字化转型面临哪些挑战？",
        "reference": "线上线下渠道割裂（库存管理不统一、客户体验不一致）；数据利用率低（缺乏数据分析工具）；供应链效率低下；客户体验滞后。",
        "type": "可回答",
    },
    {
        "question": "制造业案例中的公司是什么背景？",
        "reference": "成立于20世纪初的德国老牌汽车制造公司，拥有悠久的历史和丰富的制造经验，面对激烈市场竞争决定数字化转型。",
        "type": "可回答",
    },
    {
        "question": "金融业数字化转型前面临哪些主要挑战？",
        "reference": "客户服务模式过时（依赖实体网点、效率低体验差）；金融科技企业竞争压力；数据孤岛和风险管理滞后。",
        "type": "可回答",
    },
    {
        "question": "零售业案例公司通过什么措施解决渠道割裂问题？",
        "reference": "构建全渠道零售平台，实现线上与线下购物渠道的无缝整合，提升顾客便利性和满意度。",
        "type": "可回答",
    },
    {
        "question": "数字化转型对客户体验有什么影响？",
        "reference": "提供个性化服务、无缝体验和即时服务，深入了解客户偏好，提升客户满意度和忠诚度，提高客户终身价值。",
        "type": "可回答",
    },
    # ---- 不可回答题×2（考拒答） ----
    {
        "question": "这家公司的CEO是谁？",
        "reference": "（知识库未提及，正确行为是拒答）",
        "type": "不可回答",
    },
    {
        "question": "这份报告的发布机构是什么？",
        "reference": "（知识库未提及，正确行为是拒答）",
        "type": "不可回答",
    },
    # ---- 多跳综合题×1（考跨案例整合） ----
    {
        "question": "三个行业的案例面临的挑战有什么共同主题？",
        "reference": "共同主题：传统运营模式效率低（依赖线下/人工/老旧流程）；数据利用能力不足（孤岛、缺乏分析工具）；客户体验与服务模式落后。",
        "type": "多跳",
    },
]


# ======================================================================
# 第二部分：两套检索系统（同一Prompt同一生成模型，只变检索）
# ======================================================================

def retrieval_naive(query, collection, embedding_model, top_k=3):
    """系统A：朴素检索——纯向量检索Top-3（第2课的方式）"""
    return vector_retrieve(query, collection, embedding_model, top_k=top_k)


def retrieval_full(query, collection, embedding_model, bm25, all_docs, tokenizer, reranker_model, top_k=3):
    """系统B：完整检索——向量+BM25两路 → RRF融合 → 重排Top-3（第6课的方式）"""
    vector_docs = vector_retrieve(query, collection, embedding_model)
    bm25_docs = bm25_retrieve(query, bm25, all_docs)
    fused = rrf_fusion([vector_docs, bm25_docs])

    fused_docs = []
    for doc_text, rrf_score in fused[:8]:
        fused_docs.append(doc_text)

    from lesson6_hybrid_rerank import rerank
    reranked_docs, reranked_scores = rerank(query, fused_docs, tokenizer, reranker_model)
    return reranked_docs[:top_k]


def generate_answer(query, chunks):
    """生成回答：两套系统共用（生产级Prompt + deepseek-flash，控制变量）"""
    messages = build_prompt_production(query, chunks)
    llm = ChatOpenAI(
        model=GENERATOR_MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=GENERATOR_BASE_URL,
        temperature=TEMPERATURE,
    )
    response = llm.invoke(messages)
    return response.content


# ======================================================================
# 第三部分：手写mini-RAGAS（裁判=glm-4.5-flash）
# ======================================================================

def get_judge_llm():
    """创建裁判模型客户端（GLM，与生成模型不同厂商）"""
    return ChatOpenAI(
        model=JUDGE_MODEL,
        api_key=os.environ["ZHIPU_API_KEY"],
        base_url=JUDGE_BASE_URL,
        temperature=TEMPERATURE,
    )


def parse_judge_json(text):
    """
    解析裁判输出的JSON（带容错：剥离markdown代码围栏、重试提示）
    裁判偶尔会输出 ```json ... ``` 包裹的内容或多余文字，需要清洗
    """
    # 剥离可能的markdown代码围栏
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # 去掉开头的 ```json 或 ```，以及结尾的 ```
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    # 截取第一个[或{到最后一个]或}（防御裁判输出前后闲聊）
    start_candidates = [i for i in (cleaned.find("["), cleaned.find("{")) if i >= 0]
    if not start_candidates:
        raise ValueError(f"裁判输出中未找到JSON: {text[:100]}")
    start = min(start_candidates)
    end = max(cleaned.rfind("]"), cleaned.rfind("}"))
    return json.loads(cleaned[start:end + 1])


def judge_context_relevancy(query, chunks, judge_llm):
    """
    指标1 CR 检索相关性：裁判对每个检索块独立打分，输出均值
    打分rubric（锚定档位，抗宽容偏差）：1=直接相关 / 0.5=部分相关 / 0=无关
    """
    chunk_list_text = ""
    for i, chunk in enumerate(chunks, start=1):
        # 每块截断到200字符，裁判看得快也够判断
        chunk_list_text += f"[{i}] {chunk[:200]}\n"

    prompt = (
        "你是检索质量评估员。根据问题对每个文本块独立打分：\n"
        "- 1：直接包含回答问题所需的信息\n"
        "- 0.5：包含部分相关信息或背景信息\n"
        "- 0：与问题无关\n\n"
        f"问题：{query}\n\n文本块：\n{chunk_list_text}\n"
        f"只输出一个JSON数组，含{len(chunks)}个数字（按文本块顺序），"
        "不要输出其他任何文字。"
    )
    response = judge_llm.invoke([("human", prompt)])
    scores = parse_judge_json(response.content)
    # 防御：确保返回数量正确
    if isinstance(scores, list) and len(scores) == len(chunks):
        return float(np.mean(scores))
    return -1.0  # 解析失败标记


def judge_faithfulness(query, answer, chunks, judge_llm):
    """
    指标2 F 忠实度（RAGAS原版算法，两步）：
    步骤1：把答案拆成原子论断（claims）
    步骤2：逐条判断每条论断是否被context支持
    忠实度 = 被支持的论断数 / 总论断数
    """
    context_text = "\n\n".join(chunks)

    # ---- 步骤1：拆论断 ----
    step1_prompt = (
        "把下面的回答拆分为原子论断（每条独立可验证的事实陈述）。\n"
        f"回答：{answer}\n\n"
        '只输出JSON数组（字符串列表），例如["论断1","论断2"]。'
        "如果回答没有陈述任何事实（如纯拒答），输出[]。"
    )
    step1_response = judge_llm.invoke([("human", step1_prompt)])
    claims = parse_judge_json(step1_response.content)

    # 没有论断（纯拒答等）→ 无从幻觉，忠实度记满分
    if not isinstance(claims, list) or len(claims) == 0:
        return 1.0, 0, 0

    # ---- 步骤2：逐条判断支持性 ----
    claims_text = ""
    for i, claim in enumerate(claims, start=1):
        claims_text += f"({i}) {claim}\n"

    step2_prompt = (
        "参考以下资料，逐条判断每个论断是否被资料支持：\n"
        "- 支持：论断的信息可以从资料中直接得出\n"
        "- 不支持：论断的信息在资料中没有依据\n\n"
        f"资料：\n{context_text[:3000]}\n\n论断：\n{claims_text}\n"
        f"只输出JSON数组，含{len(claims)}个元素，每个元素为\"支持\"或\"不支持\"。"
    )
    step2_response = judge_llm.invoke([("human", step2_prompt)])
    verdicts = parse_judge_json(step2_response.content)

    if isinstance(verdicts, list) and len(verdicts) == len(claims):
        supported_count = sum(1 for v in verdicts if v == "支持")
        return supported_count / len(claims), supported_count, len(claims)
    return -1.0, 0, len(claims)  # 解析失败标记


def judge_answer_relevancy(query, answer, embedding_model, judge_llm):
    """
    指标3 AR 答案相关性（RAGAS原版机制，反向提问法）：
    让裁判从答案反向生成3个"该答案像是在回答什么问题"，
    用本地Embedding计算这些问题与原问题的平均余弦相似度。
    直觉：答非所问的答案，反向生成不出原问题。
    """
    prompt = (
        "根据下面的回答，推测它最可能在回答哪3个不同的问题。\n"
        f"回答：{answer[:500]}\n\n"
        "只输出JSON数组（3个问题字符串），不要输出其他文字。"
    )
    response = judge_llm.invoke([("human", prompt)])
    generated_questions = parse_judge_json(response.content)

    if not isinstance(generated_questions, list) or len(generated_questions) == 0:
        return -1.0

    # 原问题与生成的问题全部编码（归一化向量，点积=余弦）
    texts_to_encode = [query] + generated_questions
    embeddings = embedding_model.encode(texts_to_encode, normalize_embeddings=True)

    # 计算每个生成问题与原问题的余弦相似度，取均值
    similarities = []
    for i in range(1, len(embeddings)):
        similarity = float(np.dot(embeddings[0], embeddings[i]))
        similarities.append(similarity)
    return float(np.mean(similarities))


def is_refusal(answer):
    """拒答判定：关键词规则（简单但有效的工程手段）"""
    for keyword in REFUSAL_KEYWORDS:
        if keyword in answer:
            return True
    return False


# ======================================================================
# 第四部分：实验主流程
# ======================================================================

def run_pairwise_bias_demo(answers, judge_llm):
    """
    裁判偏差实验：同一对答案（系统A vs 系统B），交换位置各评一次，
    观察"位置偏差"——如果两次选择不一致，偏差就现形了
    """
    print("\n" + "=" * 70)
    print("裁判偏差实验：交换位置的二选一测试")
    print("=" * 70)

    answer_a = answers[0]
    answer_b = answers[1]
    question = TEST_DATASET[0]["question"]

    # 第一次：A在前B在后
    prompt_1 = (
        f"问题：{question}\n\n回答一：\n{answer_a}\n\n回答二：\n{answer_b}\n\n"
        "哪个回答更好地回答了问题？只输出\"回答一\"或\"回答二\"。"
    )
    choice_1 = judge_llm.invoke([("human", prompt_1)]).content

    # 第二次：交换位置（B在前A在后）
    prompt_2 = (
        f"问题：{question}\n\n回答一：\n{answer_b}\n\n回答二：\n{answer_a}\n\n"
        "哪个回答更好地回答了问题？只输出\"回答一\"或\"回答二\"。"
    )
    choice_2 = judge_llm.invoke([("human", prompt_2)]).content

    # 把两次选择统一换算成"选了系统A还是系统B"
    pick_1 = "系统A" if "一" in choice_1 else "系统B"
    pick_2 = "系统B" if "一" in choice_2 else "系统A"

    print(f"第一次（A在前）：裁判选择 → {choice_1.strip()[:10]}（即{pick_1}）")
    print(f"第二次（B在前）：裁判选择 → {choice_2.strip()[:10]}（即{pick_2}）")
    if pick_1 == pick_2:
        print("两次选择一致：此样本上裁判稳定（未表现出位置偏差）")
    else:
        print("两次选择相反：位置偏差现形！这正是LLM-as-judge需要校准的原因")


def main():
    # 检查两家的API Key
    if not os.environ.get("DEEPSEEK_API_KEY") or not os.environ.get("ZHIPU_API_KEY"):
        raise SystemExit("需要同时配置 DEEPSEEK_API_KEY 和 ZHIPU_API_KEY")

    print("第8课实验：mini-RAGAS评估 + 双系统对比\n" + "=" * 70)

    # ---------- 准备工作 ----------
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
    judge_llm = get_judge_llm()

    # 两套系统的配置（检索函数不同，其余完全一致）
    systems = {
        "A(朴素向量检索)": lambda q: retrieval_naive(q, collection, embedding_model),
        "B(混合+RRF+重排)": lambda q: retrieval_full(
            q, collection, embedding_model, bm25, all_docs, tokenizer, reranker_model
        ),
    }

    # 指标累积器：{系统名: {指标名: [各题得分]}}
    metrics = {}
    sample_answers = {}  # 用于偏差实验的第1题答案

    # ---------- 主循环：2系统 × 8题 ----------
    for system_name, retrieve_fn in systems.items():
        print(f"\n\n{'█' * 70}\n█ 系统{system_name}\n{'█' * 70}")
        metrics[system_name] = {"CR": [], "F": [], "AR": [], "拒答正确": []}

        for q_index, item in enumerate(TEST_DATASET):
            query = item["question"]
            print(f"\n--- 题目{q_index + 1}（{item['type']}）: {query[:30]}... ---")

            # ① 检索（两系统唯一的不同点）
            chunks = retrieve_fn(query)

            # ② 生成（同一Prompt同一模型）
            answer = generate_answer(query, chunks)
            print(f"答案预览: {answer[:80]}...")

            # 保存第1题的答案供偏差实验
            if system_name.startswith("A") and q_index == 0:
                sample_answers["A"] = answer
            if system_name.startswith("B") and q_index == 0:
                sample_answers["B"] = answer

            # ③ 三个指标依次计算
            cr = judge_context_relevancy(query, chunks, judge_llm)
            faith, supported, total = judge_faithfulness(query, answer, chunks, judge_llm)
            ar = judge_answer_relevancy(query, answer, embedding_model, judge_llm)

            print(f"CR检索相关性={cr:.2f} | F忠实度={faith:.2f}({supported}/{total}) | AR答案相关性={ar:.2f}")
            if cr >= 0:
                metrics[system_name]["CR"].append(cr)
            if faith >= 0:
                metrics[system_name]["F"].append(faith)
            if ar >= 0:
                metrics[system_name]["AR"].append(ar)

            # ④ 不可回答题：额外记录拒答正确性
            if item["type"] == "不可回答":
                refused = is_refusal(answer)
                metrics[system_name]["拒答正确"].append(1.0 if refused else 0.0)
                print(f"拒答测试: {'✅ 正确拒答' if refused else '❌ 未拒答（可能幻觉）'}")

    # ---------- 汇总报告 ----------
    print(f"\n\n{'=' * 70}")
    print("评估汇总报告（各指标为全部题目的平均值）")
    print(f"{'=' * 70}")
    print(f"{'系统':<20} | {'CR检索':>7} | {'F忠实':>6} | {'AR答案':>7} | {'拒答正确率':>8}")
    print("-" * 70)
    for system_name, system_metrics in metrics.items():
        cr_mean = np.mean(system_metrics["CR"]) if system_metrics["CR"] else -1
        f_mean = np.mean(system_metrics["F"]) if system_metrics["F"] else -1
        ar_mean = np.mean(system_metrics["AR"]) if system_metrics["AR"] else -1
        refusal_mean = np.mean(system_metrics["拒答正确"]) if system_metrics["拒答正确"] else -1
        print(f"{system_name:<20} | {cr_mean:>7.3f} | {f_mean:>6.3f} | {ar_mean:>7.3f} | {refusal_mean:>8.2f}")

    print("\n解读提示：CR差距=检索质量差距（第3~6课优化的直接量化）；")
    print("F差距=答案忠实度；AR差距=答案针对性；拒答正确率=幻觉抑制能力。")

    # ---------- 裁判偏差实验 ----------
    run_pairwise_bias_demo([sample_answers["A"], sample_answers["B"]], judge_llm)


if __name__ == "__main__":
    main()
