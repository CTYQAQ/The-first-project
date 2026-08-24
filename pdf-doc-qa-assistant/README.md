# 智能文档问答助手（PDF Doc QA Assistant）

基于 [hello-agents](https://pypi.org/project/hello-agents/) 框架（`0.2.8`）的检索增强生成（RAG）+ 多类型记忆（Memory）文档问答工具，提供 Gradio Web 界面。

## 功能

- 📄 **PDF 加载**：上传 PDF → MarkItDown 转换 → 智能分块 → 向量化入库
- 💬 **智能问答**：基于 RAG 检索，返回答案与参考来源，支持高级检索（MQE + HyDE）
- 📝 **学习笔记**：写入语义记忆（Semantic Memory）
- 🔍 **学习回顾**：从记忆系统检索历史学习历程
- 📊 **统计与报告**：查看学习统计，导出 JSON 学习报告

## 目录结构

```
.
├── app.py                 # Gradio Web 界面入口（python app.py 启动，端口 7860）
├── pdf_assistant.py       # PDFLearningAssistant：封装 RAGTool + MemoryTool
├── make_sample_pdf.py     # 生成中文示例 PDF（输出到 data_base/sample.pdf）
├── 启动8.4应用.bat          # Windows 一键启动脚本（释放 7860 端口后运行 app.py）
├── .env.example           # 环境变量模板（复制为 .env 并填入你的 Key）
├── requirements.txt       # Python 依赖
├── README.md              # 项目说明
├── data_base/             # 示例与业务数据
│   └── sample.pdf         # 示例文档（由 make_sample_pdf.py 生成）
├── notebook/              # 示例脚本 / 最小可运行 demo
│   ├── demo.py            # SimpleAgent 基础对话示例
│   └── memory_demo.py     # MemoryTool 最小可运行示例（含三处框架补丁）
├── docs/                  # 架构与扩展文档
│   └── ARCHITECTURE.md    # 项目架构与流程图
└── figures/               # 项目截图、架构图等图片资源
```

## 快速开始

```bash
# 1. 安装依赖（建议 Python 3.10+）
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key

# 3. 启动 Web 界面
python app.py
# 浏览器自动打开 http://127.0.0.1:7860/
```

或双击 `启动8.4应用.bat`（Windows，使用系统 Python 3.12 运行）。

## 在本机跑通的说明（框架补丁）

`pdf_assistant.py` / `memory_demo.py` 顶部包含三处最小化框架补丁，使本项目在
**无 Docker / 无 Neo4j / 无真实 Qdrant 服务**的本机环境也能运行：

1. **Qdrant 内存模式**：把“连 localhost:6333”替换为 `qdrant-client` 自带的
   `location=":memory:"`，无需任何外部服务。
2. **Neo4j 降级**：`SemanticMemory` 需要 Neo4j 图库，不可用时降级为 `None`，
   纯文本语义记忆照常工作。
3. **哈希嵌入器**：框架自带的 TF-IDF 嵌入器需先 `fit` 语料才能 `encode`（增量写入
   不触发），这里换成无需训练的中文感知哈希嵌入器（384 维），保证中文检索命中。

> 若你的环境已具备 Qdrant / Neo4j 服务，可移除对应补丁使用完整能力。

## 安全提示

- **切勿提交真实 `.env`**（含 API Key）。`.env` 已在 `.gitignore` 中排除。
- 提交仓库时请使用 `.env.example` 模板。

## 依赖

```
hello-agents==0.2.8
gradio
python-dotenv
qdrant-client
numpy
PyMuPDF
```
