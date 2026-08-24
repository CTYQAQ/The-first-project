# 项目架构

## 总体流程

```
┌─────────────────────────────────────────────────────┐
│              Gradio Web (app.py)                    │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐           │
│   │  加载PDF │  │ 智能问答 │  │ 学习笔记 │  ...     │
│   └────┬────┘  └────┬────┘  └────┬────┘           │
└────────┼────────────┼────────────┼────────────────┘
         │            │            │
         ▼            ▼            ▼
   ┌─────────────────────────────────────────┐
   │  PDFLearningAssistant (pdf_assistant.py) │
   │  封装 RAGTool + MemoryTool 的业务逻辑     │
   └──────┬────────────────────┬─────────────┘
          │                    │
          ▼                    ▼
   ┌──────────────┐    ┌────────────────┐
   │   RAGTool    │    │   MemoryTool   │
   │ (检索增强生成) │    │  (多类型记忆)   │
   └──────┬───────┘    └────────┬───────┘
          │                     │
          ▼                     ▼
   ┌──────────────┐    ┌────────────────┐
   │  Qdrant 内存 │    │ 语义/情景/工作 │
   │  + 哈希嵌入  │    │   记忆 + 摘要  │
   └──────────────┘    └────────────────┘
```

## 目录职责

| 目录/文件 | 职责 |
| --- | --- |
| `app.py` | Gradio Web 入口，提供 4 个 Tab（问答/笔记/回顾/统计） |
| `pdf_assistant.py` | `PDFLearningAssistant`：封装 RAGTool + MemoryTool 的核心业务类 |
| `make_sample_pdf.py` | 用 PyMuPDF 生成中文示例 PDF，写入 `data_base/sample.pdf` |
| `notebook/demo.py` | SimpleAgent 基础对话示例 |
| `notebook/memory_demo.py` | MemoryTool 最小可运行示例（含三处框架补丁） |
| `data_base/sample.pdf` | 示例文档（大语言模型简介） |
| `docs/` | 架构与扩展文档 |
| `figures/` | 项目截图、架构图等 |
| `启动8.4应用.bat` | Windows 一键启动脚本 |

## 框架补丁说明

`pdf_assistant.py` 与 `notebook/memory_demo.py` 顶部包含三处最小化框架补丁，使本项目在
**无 Docker / 无 Neo4j / 无真实 Qdrant 服务**的本机环境也能运行：

1. **Qdrant 内存模式** — 把"连 localhost:6333"替换为 `qdrant-client` 自带的 `location=":memory:"`。
2. **Neo4j 降级** — `SemanticMemory` 需要 Neo4j 图库，不可用时降级为 `None`，纯文本语义记忆照常工作。
3. **哈希嵌入器** — 框架自带的 TF-IDF 嵌入器需先 `fit` 语料才能 `encode`，换成无需训练的中文感知哈希嵌入器（384 维）。

> 若环境已具备 Qdrant / Neo4j 服务，可移除对应补丁使用完整能力。

## 扩展方向

- 替换 `data_base/` 下的 `sample.pdf` 为你自己的文档即可让助手回答对应领域问题。
- `.env` 中切换 `LLM_BASE_URL` 与 `LLM_MODEL_ID` 即可接入任意 OpenAI 兼容服务（OpenAI / 通义千问 / 智谱等）。
- `notebook/` 下追加 Jupyter/脚本即可记录更多实验。
