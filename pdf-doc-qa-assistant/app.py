"""Gradio Web 界面 — 智能文档问答助手

基于 pdf_assistant.PDFLearningAssistant，提供：
  1. 初始化助手（加载模型/API/知识库）
  2. 上传并加载 PDF 文档
  3. 智能问答（基于 RAG，返回答案与参考来源）
  4. 学习笔记（写入语义记忆）
  5. 学习回顾（从记忆系统检索）
  6. 统计查看与报告生成（导出 JSON）

运行：py -3.12 app.py
"""

import json
import os

import gradio as gr

from pdf_assistant import PDFLearningAssistant

# 全局助手实例（懒初始化）
assistant = None


def init_assistant(user_id: str):
    global assistant
    user_id = (user_id or "default_user").strip()
    try:
        assistant = PDFLearningAssistant(user_id=user_id)
        return (
            f"✅ 助手初始化成功！\n"
            f"👤 用户: {user_id}\n"
            f"🆔 会话ID: {assistant.session_id}"
        )
    except Exception as e:  # noqa: BLE001
        return f"❌ 初始化失败: {e}"


def load_doc(pdf_file):
    if assistant is None:
        return "⚠️ 请先点击「初始化助手」"
    if pdf_file is None:
        return "⚠️ 请先上传 PDF 文件"
    path = pdf_file.name if hasattr(pdf_file, "name") else pdf_file
    res = assistant.load_document(path)
    return res["message"]


def do_ask(question: str, use_advanced: bool):
    if assistant is None:
        return "⚠️ 请先初始化助手并加载文档"
    if not question or not question.strip():
        return "⚠️ 请输入你的问题"
    return assistant.ask(question, use_advanced_search=bool(use_advanced))


def do_note(content: str, concept: str):
    if assistant is None:
        return "⚠️ 请先初始化助手"
    return assistant.add_note(content, concept)


def do_recall(query: str):
    if assistant is None:
        return "⚠️ 请先初始化助手"
    return assistant.recall(query)


def show_stats():
    if assistant is None:
        return "⚠️ 请先初始化助手"
    s = assistant.get_stats()
    return "\n".join(f"{k}: {v}" for k, v in s.items())


def make_report():
    if assistant is None:
        return "⚠️ 请先初始化助手"
    rep = assistant.generate_report()
    msg = "✅ 学习报告已生成\n"
    msg += f"📄 文件: {rep.get('report_file', '')}\n\n"
    msg += json.dumps(rep, ensure_ascii=False, indent=2, default=str)
    return msg


def build_ui():
    with gr.Blocks(title="智能文档问答助手") as demo:
        gr.Markdown(
            "# 📚 智能文档问答助手\n"
            "基于 **RAGTool**（检索增强生成）与 **MemoryTool**（多类型记忆）构建。\n"
            "流程：初始化助手 → 上传 PDF → 智能问答 / 记笔记 / 回顾 / 导出报告。"
        )

        # ---------- 初始化 & 文档加载 ----------
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ① 初始化助手")
                user_id_in = gr.Textbox(
                    label="用户ID（用于隔离不同用户数据）", value="default_user"
                )
                init_btn = gr.Button("🚀 初始化助手", variant="primary")
                init_out = gr.Textbox(label="初始化状态", lines=4, interactive=False)
            with gr.Column(scale=2):
                gr.Markdown("### ② 加载 PDF 文档")
                pdf_file = gr.File(label="上传 PDF", file_types=[".pdf"])
                load_btn = gr.Button("📄 加载文档", variant="secondary")
                load_out = gr.Textbox(label="加载结果", lines=6, interactive=False)

        init_btn.click(init_assistant, inputs=[user_id_in], outputs=[init_out])
        load_btn.click(load_doc, inputs=[pdf_file], outputs=[load_out])

        # ---------- 功能标签页 ----------
        with gr.Tabs():
            with gr.TabItem("💬 智能问答"):
                with gr.Row():
                    adv_chk = gr.Checkbox(label="启用高级检索 (MQE + HyDE)", value=True)
                q_in = gr.Textbox(
                    label="你的问题", placeholder="例如：什么是大语言模型？", lines=2
                )
                ask_btn = gr.Button("提问", variant="primary")
                a_out = gr.Textbox(label="答案（含参考来源）", lines=12, interactive=False)
                ask_btn.click(
                    do_ask, inputs=[q_in, adv_chk], outputs=[a_out]
                )

            with gr.TabItem("📝 学习笔记"):
                note_concept = gr.Textbox(
                    label="关联概念（可选）", placeholder="例如：LLM、Transformer"
                )
                note_content = gr.Textbox(
                    label="笔记内容", placeholder="记录你学到的知识点...", lines=4
                )
                note_btn = gr.Button("添加笔记", variant="primary")
                note_out = gr.Textbox(label="结果", lines=3, interactive=False)
                note_btn.click(
                    do_note, inputs=[note_content, note_concept], outputs=[note_out]
                )

            with gr.TabItem("🔍 学习回顾"):
                recall_q = gr.Textbox(
                    label="回顾关键词", placeholder="例如：加载过哪些文档？", lines=2
                )
                recall_btn = gr.Button("回顾", variant="secondary")
                recall_out = gr.Textbox(label="回顾结果", lines=12, interactive=False)
                recall_btn.click(do_recall, inputs=[recall_q], outputs=[recall_out])

            with gr.TabItem("📊 统计与报告"):
                with gr.Row():
                    stats_btn = gr.Button("查看统计", variant="secondary")
                    report_btn = gr.Button("生成学习报告", variant="primary")
                stats_out = gr.Textbox(label="学习统计", lines=6, interactive=False)
                report_out = gr.Textbox(label="学习报告 (JSON)", lines=16, interactive=False)
                stats_btn.click(show_stats, inputs=[], outputs=[stats_out])
                report_btn.click(make_report, inputs=[], outputs=[report_out])

    return demo


if __name__ == "__main__":
    ui = build_ui()
    # 仅本机访问；如需局域网可设 share=True
    ui.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )
