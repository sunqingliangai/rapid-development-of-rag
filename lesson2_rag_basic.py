"""
第2课实战：从0到1的第一个RAG应用（2026-10 现代化版本）

完整流程（Naive RAG）：
    PDF解析 → 文本分块 → Embedding向量化 → FAISS索引 → 向量检索 → LLM生成

与2024课程示例（GeekbangCourse.../rag_app_lesson2.py）的对照：
- 文档加载   pypdf 直接解析（★2026改造点：课程用的 langchain-community
             包已于2026年6月被官方弃用归档，且Loader无官方替代包，
             官方建议直接使用解析库本身；pypdf正是PyPDFLoader的底层库）
- 文本分块   RecursiveCharacterTextSplitter（独立包）           课程同款
- Embedding  本地 bge-small-zh-v1.5（sentence-transformers）     课程同款
- 向量检索   faiss-cpu 裸用法（第5课将换成 ChromaDB）            课程同款
- 大模型生成 ★现代化改造点：课程用 dashscope SDK 调 Qwen；
             我们用 langchain-openai 的 ChatOpenAI 调 DeepSeek 的
             OpenAI 兼容端点，不装任何厂商专有 SDK

运行方式（推荐用 uv 的 --env-file 加载密钥）：
    uv run --env-file .env python lesson2_rag_basic.py
"""

import os

# 必须在导入 sentence-transformers 之前设置：
# 禁用分词器并行（多线程/多进程场景下会死锁或告警），课程代码同样处理
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ---------- 配置区 ----------
# 用"当前文件所在目录"拼出绝对路径，从任何目录启动都能找到文件
# （课程代码用相对路径 'rag_app/test_lesson2.pdf'，依赖启动目录，容易踩坑）
BASE_DIR = Path(__file__).resolve().parent
EMBEDDING_MODEL_PATH = BASE_DIR / "models" / "bge-small-zh-v1.5"
PDF_PATH = BASE_DIR / "data" / "test_lesson2.pdf"

CHUNK_SIZE = 512  # 每个文本块的最大字符数。受Embedding模型输入上限约束：
# bge-small-zh-v1.5 的 max_seq_length=512（token），
# 中文约1字≈1token，块太大超出上限会被截断、语义丢失
CHUNK_OVERLAP = 128  # 相邻文本块之间的重叠字符数，保证跨块的语义不被切断
TOP_K = 3  # 检索返回最相似的前K个文本块


def load_embedding_model():
    """
    加载本地 bge-small-zh-v1.5 模型（512维中文Embedding模型，离线运行）
    :return: 返回加载好的 SentenceTransformer 模型对象
    """
    print("加载Embedding模型中...")
    # SentenceTransformer 从本地目录读取模型（不需要联网下载）
    embedding_model = SentenceTransformer(str(EMBEDDING_MODEL_PATH))
    # 打印模型最大输入长度，确认与我们的 chunk_size 匹配
    print(f"模型最大输入长度(max_seq_length): {embedding_model.max_seq_length}")
    return embedding_model


def indexing_process(pdf_file, embedding_model):
    """
    索引流程：加载PDF文件提取文本，分割成文本块，计算嵌入向量，存入FAISS索引。

    :param pdf_file: PDF文件路径
    :param embedding_model: 预加载的Embedding模型
    :return: 返回 (FAISS向量索引, 文本块原文列表)
             注意：向量库中"向量"与"原文"的对应关系靠两者下标一致来维护，
             即第i个向量对应 chunks[i]，这是裸用FAISS时最容易出错的地方
    """
    # ---------- 第1步：文档解析 ----------
    # 用 pypdf 直接读取PDF（它就是课程里 PyPDFLoader 底层调用的库；
    # PyPDFLoader所在的langchain-community包已于2026年弃用，我们直接用底层库）
    pdf_reader = PdfReader(str(pdf_file))
    # 逐页提取文本，用换行符拼接成PDF的完整文本，（合并后丢失了页边界信息，更精细的做法见第3课）
    pdf_text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        pdf_text += page_text + "\n"
    print(f"PDF共{len(pdf_reader.pages)}页，总字符数: {len(pdf_text)}")

    # ---------- 第2步：文本分块 ----------
    # 配置递归字符分割器：优先按 段落→句子→字符 的层级尝试切分，
    # 尽量让chunk落在自然语义边界上（原理与更多策略见第4课）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,  # 每块最大512字符
        chunk_overlap=CHUNK_OVERLAP,  # 相邻块重叠128字符
    )
    chunks = text_splitter.split_text(pdf_text)
    print(f"分割为 {len(chunks)} 个文本块")

    # ---------- 第3步：向量化 ----------
    # 逐块编码为嵌入向量。normalize_embeddings=True 对向量做L2归一化，
    # 归一化后"内积"就等于"余弦相似度"，FAISS用内积索引即可做余弦检索
    # （小知识：encode也支持一次传入整个列表批量编码，速度更快，结果相同）
    embeddings = []
    for chunk in chunks:
        embedding = embedding_model.encode(chunk, normalize_embeddings=True)
        embeddings.append(embedding)
    print(f"向量化完成，向量维度: {len(embeddings[0])}")

    # 将嵌入向量列表转化为numpy数组，FAISS的索引操作要求numpy数组输入，
    # 且必须是float32类型（SentenceTransformer输出的本来就是float32，这里显式写明）
    embeddings_np = np.array(embeddings, dtype="float32")

    # ---------- 第4步：构建向量索引 ----------
    # 获取嵌入向量的维度（每个向量的长度，本模型为512）
    dimension = embeddings_np.shape[1]
    # IndexFlatIP：内积(IP)索引，暴力精确检索，适合小规模知识库；
    # 大规模库会换IVF/HNSW等近似索引（第5课讲向量数据库原理时展开）
    index = faiss.IndexFlatIP(dimension)
    # 将所有向量加入索引，之后就能用它做相似度检索
    index.add(embeddings_np)
    print("索引流程完成\n")

    return index, chunks


def retrieval_process(query, index, chunks, embedding_model, top_k=TOP_K):
    """
    检索流程：将用户查询转化为向量，在FAISS索引中检索最相似的前K个文本块。

    :param query: 用户查询语句
    :param index: 索引阶段建好的FAISS向量索引
    :param chunks: 文本块原文列表（用于把命中的下标换回原文）
    :param embedding_model: 预加载的Embedding模型
    :param top_k: 返回最相似的前K个结果
    :return: 返回最相似的K个文本块原文列表
    """
    # 查询向量化：normalize_embeddings=True 表示对嵌入向量做归一化。
    # 注意：必须用与索引阶段【同一个】Embedding模型编码，保证query和chunk落在同一向量空间，相似度才有意义
    query_embedding = embedding_model.encode(query, normalize_embeddings=True)
    # 将嵌入向量转化为numpy数组。FAISS要求"批量"输入：
    # 即使只有一条查询，也要外包一层列表，变成 (1, 512) 的二维形状
    query_embedding = np.array([query_embedding])

    # 在FAISS索引中检索与查询向量最相似的前top_k个结果
    # 返回值：scores是相似度得分矩阵，indices是命中文本块的下标矩阵，
    # 两者的第[0]行对应第一条（也是唯一一条）查询的结果
    scores, indices = index.search(query_embedding, top_k)

    print(f"查询语句: {query}")
    print(f"最相似的前{top_k}个文本块:")

    results = []
    for i in range(top_k):
        # 取出第i个命中结果的文本块下标，再从chunks列表取回原文
        result_chunk = chunks[indices[0][i]]
        # 取出第i个命中结果的相似度得分：归一化后即余弦相似度，范围[-1,1]，越大越相似
        result_score = scores[0][i]
        print(f"\n--- 文本块{i} (相似度 {result_score:.4f}) ---\n{result_chunk}")
        # 将命中的文本块存入结果列表
        results.append(result_chunk)

    print("\n检索流程完成")
    return results


def generate_process(query, chunks):
    """
    生成流程：把检索到的文本块与用户问题组装成Prompt，调用DeepSeek大模型生成回答。

    :param query: 用户查询语句
    :param chunks: 检索获得的相关文本块
    :return: 返回模型生成的回答文本
    """
    # 构建参考文档内容，格式为"【参考文档1】\n...\n\n【参考文档2】..."，
    # 给每块编号，方便模型在回答中对应引用
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"【参考文档{i + 1}】\n{chunk}\n\n"

    # 构建生成模型所需的Prompt：参考文档 + 用户问题
    prompt = f"请根据以下参考文档回答问题。\n\n{context}\n问题：{query}"
    print(f"生成模型的Prompt:\n{'-' * 60}\n{prompt}\n{'-' * 60}")

    # 创建大模型客户端。ChatOpenAI走OpenAI兼容协议，只改base_url就能接入DeepSeek，无需安装dashscope等厂商专有SDK
    # 这是与课程代码最主要的现代化差异点
    llm = ChatOpenAI(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",  # DeepSeek的OpenAI兼容端点
        temperature=0.3,  # 事实型问答调低温度，减少自由发挥
    )

    # LangChain 1.x 的消息格式：由 (角色, 内容) 元组组成的列表
    # system提示约束"只依据文档回答"，是抑制幻觉的第一道闸门（第7课展开）
    messages = [
        (
            "system",
            "你是一个严谨的知识库问答助手。只依据用户提供的参考文档回答；"
            "若参考文档不足以回答问题，请直接说明，不要编造。",
        ),
        ("human", prompt),
    ]

    print("\n生成过程开始:")
    # 用于累积完整回答（流式输出是一片一片到达的）
    generated_response = ""
    try:
        # .stream() 流式输出：模型每生成一小段就立即返回一片，体验更好
        # （课程代码用dashscope的stream参数实现同样效果）
        for piece in llm.stream(messages):
            # 取出这一片回答文本
            content = piece.content
            generated_response += content
            # 实时打印，不换行
            print(content, end="", flush=True)
        print("\n生成流程完成")
        return generated_response
    except Exception as e:
        print(f"\n大模型调用失败: {e}")
        raise


def main():
    # 检查API Key是否已配置，给出友好提示而不是抛出难懂的KeyError
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit(
            "未检测到 DEEPSEEK_API_KEY。请编辑 .env 文件填入你的Key，"
            "然后用 `uv run --env-file .env python lesson2_rag_basic.py` 运行"
        )

    print("RAG流程开始\n" + "=" * 60)
    # query = "下面报告中涉及了哪几个行业的案例以及总结各自面临的挑战？"
    query = "这家公司的 CEO 是谁？"

    # 加载Embedding模型（索引和检索两个阶段都要用它）
    embedding_model = load_embedding_model()

    # 索引流程：PDF → 分块 → 向量化 → FAISS索引（全部在内存中）
    index, chunks = indexing_process(PDF_PATH, embedding_model)

    # 检索流程：查询向量化 → 相似度检索 → 取回Top-K文本块
    retrieval_chunks = retrieval_process(query, index, chunks, embedding_model)

    # 生成流程：组装Prompt → 调用DeepSeek → 流式输出答案
    generate_process(query, retrieval_chunks)

    print("\n" + "=" * 60 + "\nRAG流程结束")


if __name__ == "__main__":
    main()
