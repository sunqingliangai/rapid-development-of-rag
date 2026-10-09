"""
第3课加餐实验：用视觉模型把文档中的图片转成"可检索的文字"

对应本课讲解中的"VLM生成文字描述"路线：
    文档中的图片 → 视觉语言模型(VLM)生成上下文化文字描述 → 描述作为普通文本进入RAG索引

模型选型说明（2026-10）：
- 智谱老的 glm-4v 已停用；现役视觉模型为 GLM-4.5V/4.6V/5V 系列
- 我们默认用 glm-4.6v-flash：官方"永久免费"，做实验零成本
- 智谱同样提供OpenAI兼容端点，所以依然用 ChatOpenAI，只改base_url

运行方式：
    uv run --env-file .env python lesson3_vision_caption.py
"""

import base64
import os
from pathlib import Path

from langchain_openai import ChatOpenAI
from pypdf import PdfReader

# ---------- 配置区 ----------
BASE_DIR = Path(__file__).resolve().parent
# 从第3课的测试PDF中提取嵌入图片（图文混排文档的典型场景）
PDF_PATH = BASE_DIR / "data" / "lesson3" / "test.pdf"
# 备用方案：如果测试PDF里没有图片，就用课程素材里的RAG流程图演示
FALLBACK_IMAGE_PATH = (
    BASE_DIR.parent / "courses" / "assets" / "7bc529003e05a3ab0561204230a83bdc.png"
)
# 提取出的图片保存目录（方便人工查看验证）
OUTPUT_DIR = BASE_DIR / "data" / "lesson3_extracted_images"


def extract_first_image_from_pdf(pdf_path):
    """
    用 pypdf 从PDF中提取第一张嵌入图片
    :param pdf_path: PDF文件路径
    :return: 返回 (图片原始字节, 所在页码)；没有图片时返回 (None, None)
    """
    reader = PdfReader(str(pdf_path))
    # 逐页扫描，页码从1开始计
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            # page.images 是该页所有嵌入图片的列表
            for image_file in page.images:
                # image_file.data 是图片的原始二进制数据
                return image_file.data, page_number
        except Exception:
            # 个别页面图片对象损坏时跳过该页，继续找下一页
            continue
    return None, None


def caption_image_with_vlm(image_bytes):
    """
    调用智谱视觉模型，把图片转写成文字描述
    :param image_bytes: 图片的二进制数据
    :return: 返回模型生成的文字描述
    """
    # 智谱的OpenAI兼容端点：只改base_url即可，无需专有SDK
    llm = ChatOpenAI(
        model=os.environ.get("ZHIPU_VISION_MODEL", "glm-4.6v-flash"),
        api_key=os.environ["ZHIPU_API_KEY"],
        base_url="https://open.bigmodel.cn/api/paas/v4/",  # 智谱OpenAI兼容端点
        temperature=0.1,  # 描述任务要求忠实于图片内容，温度调低
    )

    # 本地图片无法提供公网URL，转成base64内嵌到请求里（公网图片可直接传URL）
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = "data:image/png;base64," + image_base64

    # OpenAI多模态消息格式：content是列表，混合"文本块"和"图片块"两种类型
    # 提示词要点：告诉模型图片的来源背景，并要求输出"能被文本检索"的描述
    messages = [
        (
            "human",
            [
                {
                    "type": "text",
                    "text": "这张图片来自一份《数字化转型》报告的PDF文档。"
                    "请把图片内容完整转写成文字描述：如果是图表，说明图表类型、标题和关键数据；"
                    "如果是流程图，描述各个步骤和流向。要求描述详细、信息完整，能被文本搜索引擎检索到。",
                },
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        )
    ]

    # 多模态描述不需要流式，直接invoke拿到完整结果
    response = llm.invoke(messages)
    return response.content


def main():
    # 检查智谱API Key是否已配置
    if not os.environ.get("ZHIPU_API_KEY"):
        raise SystemExit(
            "未检测到 ZHIPU_API_KEY。请编辑 .env 文件填入你的智谱Key，"
            "然后用 `uv run --env-file .env python lesson3_vision_caption.py` 运行"
        )

    print("第3课加餐实验：文档图片 → 视觉模型 → 可检索文本\n" + "=" * 60)

    # 第1步：从PDF中提取第一张嵌入图片
    image_bytes, page_number = extract_first_image_from_pdf(PDF_PATH)
    if image_bytes:
        source_desc = f"测试PDF第{page_number}页"
        # 保存提取出的图片，方便你人工查看，对照模型的描述是否准确
        OUTPUT_DIR.mkdir(exist_ok=True)
        output_path = OUTPUT_DIR / "extracted_image_1.png"
        output_path.write_bytes(image_bytes)
        print(f"从{source_desc}提取到嵌入图片，已保存到: {output_path}")
    else:
        # 备用方案：测试PDF里没有嵌入图片，用课程素材的RAG流程图演示同样的流程
        image_bytes = FALLBACK_IMAGE_PATH.read_bytes()
        source_desc = "课程素材（RAG标准流程图）"
        print(f"测试PDF中没有嵌入图片，改用{source_desc}演示")

    # 第2步：调用视觉模型生成文字描述
    vision_model_name = os.environ.get("ZHIPU_VISION_MODEL", "glm-4.6v-flash")
    print(f"\n调用智谱视觉模型({vision_model_name})生成描述...")
    caption = caption_image_with_vlm(image_bytes)

    print("\n生成的文字描述:")
    print("-" * 60)
    print(caption)
    print("-" * 60)

    # 第3步：说明这段描述如何进入RAG索引
    print("\n这段描述就是普通的文本字符串，可以直接像其他chunk一样进入")
    print("分块 → 向量化 → 检索流程——这就是'图片转可检索文本'的完整路线。")
    print(
        "生产建议：描述时带上文档标题、页码等上下文信息（即Contextual Retrieval思想）"
    )


if __name__ == "__main__":
    main()
