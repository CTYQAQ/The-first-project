"""生成一份中文示例 PDF（用于演示 PDFLearningAssistant 的加载与问答）。"""
import fitz  # PyMuPDF

FONT = "C:/Windows/Fonts/msyh.ttc"

doc_text = """大语言模型（Large Language Model，简称 LLM）是基于海量文本数据训练的人工智能模型。
它能够理解和生成自然语言，是当前人工智能领域最重要的突破之一。

一、什么是大语言模型
大语言模型通常采用 Transformer 架构，通过在大规模语料上进行自监督学习，
习得语言的统计规律与世界知识。代表性的模型包括 GPT 系列、DeepSeek、通义千问等。

二、核心能力
1. 文本生成：写文章、写代码、翻译。
2. 问答与推理：基于给定上下文回答问题。
3. 摘要与总结：对长文本进行压缩提炼。

三、检索增强生成（RAG）
RAG 指在大语言模型回答问题时，先从外部知识库检索相关资料，再将资料注入
提示词，从而生成有据可依的答案。RAG 能显著降低模型幻觉，适合私有文档问答。

四、记忆机制
智能体通常具备多种记忆：工作记忆（当前上下文）、情景记忆（发生过的具体事件）、
语义记忆（沉淀的知识与笔记）。合理的记忆管理可让助手具备个性化与连续性。
"""

pdf = fitz.open()
page = pdf.new_page()
tw = fitz.TextWriter(page.rect)
# 逐行写入，使用系统微软雅黑字体渲染中文
y = 72
for line in doc_text.splitlines():
    if line.strip() == "":
        y += 16
        continue
    tw.append((72, y), line, font=fitz.Font(fontfile=FONT), fontsize=12)
    y += 22
tw.write_text(page)
pdf.subset_fonts()
pdf.save("data_base/sample.pdf")
pdf.close()
print("已生成 sample.pdf")
