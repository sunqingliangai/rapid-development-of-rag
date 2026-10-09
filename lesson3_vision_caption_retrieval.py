"""
第3课闭环演示：图片描述并入索引后，图片内容变得"可检索"

实验逻辑：
    1. 用第3课的索引流程构建向量库（10种格式文档 → 130个文本块）
    2. 把视觉模型生成的图片描述，当作一个"额外的文本块"加入向量库
    3. 用一个"只有图片里才有答案"的问题测试检索效果

预期结果：
    加入描述前：没有任何文本块能回答"RAG标准流程分哪三个阶段"
    加入描述后：图片描述块以明显高分成为Top-1

运行方式：
    uv run python lesson3_caption_retrieval_demo.py
    （本实验只用本地Embedding模型，不需要API Key）
"""

import os

# 必须在导入 sentence-transformers 之前设置（原因见第2课）
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from sentence_transformers import SentenceTransformer

# 复用第3课主代码里的现成组件，避免重复代码
from lesson3_multiformat_rag import DATA_FOLDER, indexing_process, load_embedding_model

# 图片的文字描述（来自 lesson3_vision_caption.py 的视觉模型输出，此处手工摘录核心部分。
# 生产系统中应该由程序自动衔接：视觉模型输出 → 直接作为chunk字符串，无需人工搬运）
IMAGE_CAPTION = (
    "RAG标准流程图分为三个阶段：1)索引阶段：知识库内容提取转换为字符串，"
    "划分段落区块生成文本块chunk，再经向量嵌入形成向量库；"
    "2)检索阶段：用户问题经向量嵌入得到问题向量，与向量库中的向量进行相似匹配；"
    "3)生成阶段：匹配出K段相关段落，与原始问题组装成提示词Prompt，"
    "输入大语言模型LLM，生成基于知识库的回答。"
)


def main():
    print("第3课闭环演示：图片描述并入索引\n" + "=" * 60)

    # 第1步：构建第3课的向量库（10种格式文档 → 130个chunks）
    embedding_model = load_embedding_model()
    index, chunks = indexing_process(str(DATA_FOLDER), embedding_model)

    # 第2步：把图片描述作为一个额外的chunk加入索引
    # 2a. 用同一个Embedding模型把描述文本编码成归一化向量
    # 2b. 转为float32二维数组，FAISS的add要求批量形状 (1, 512)
    caption_embedding = embedding_model.encode(IMAGE_CAPTION, normalize_embeddings=True)
    caption_embedding = np.array([caption_embedding], dtype="float32")
    # 2c. 把新向量追加进FAISS索引（IndexFlatIP支持随时动态添加）
    index.add(caption_embedding)
    # 2d. ★关键一步：chunks列表必须同步追加原文！
    # 裸用FAISS时，"向量"与"原文"的对应关系靠下标一致来维护：
    # 第i个向量 ↔ chunks[i]。如果只add向量不追加原文，
    # 检索命中这个下标时会取到错位的（甚至越界的）文本块
    chunks.append(IMAGE_CAPTION)
    print(f"\n图片描述已并入索引，当前共 {len(chunks)} 个文本块")

    # 第3步：用"只有图片里才有答案"的问题测试检索
    query = "RAG标准流程分为哪三个阶段？"
    query_embedding = embedding_model.encode(query, normalize_embeddings=True)
    query_embedding = np.array([query_embedding])
    scores, indices = index.search(query_embedding, 3)

    print(f"\n闭环测试查询: {query}")
    print("-" * 60)
    for i in range(3):
        # indices[0][i]是命中块的下标，从chunks取回原文对照
        hit_chunk = chunks[indices[0][i]]
        hit_score = scores[0][i]
        print(f"Top{i + 1} 相似度{hit_score:.4f}: {hit_chunk[:60]}...")
    print("-" * 60)

    print("\n观察：Top-1以显著高分命中图片描述块——")
    print("图片内容从'文本检索不可见'变成了'可精准检索'。")


if __name__ == "__main__":
    main()
