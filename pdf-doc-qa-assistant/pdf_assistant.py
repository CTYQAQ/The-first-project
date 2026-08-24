"""PDFLearningAssistant — 智能文档问答助手

封装 RAGTool（检索增强生成）和 MemoryTool（多类型记忆）的调用逻辑，
提供：加载 PDF、智能问答、笔记记录、学习回顾、统计查看、报告生成。

依赖 hello-agents==0.2.8 的工具 API：
  - Tool.execute(action: str, **kwargs) -> str  （返回字符串，不是 dict）

为在本机（无 Docker / 无 Neo4j / 无真实 Qdrant 服务）跑通，文件顶部包含三处
最小化框架补丁（已在 memory_demo.py 验证）：
  1. Qdrant 内存模式：把“连 localhost:6333”替换为 qdrant-client 自带的
     location=":memory:"，无需任何外部服务。
  2. Neo4j 降级：SemanticMemory 需要 Neo4j 图库，不可用时降级为 None，
     纯文本语义记忆照常工作。
  3. 哈希嵌入器：框架自带的 TF-IDF 兜底需先 fit 语料才能 encode（增量写入不
     触发），这里换成无需训练的中文感知哈希嵌入器（384 维），保证中文检索命中。
"""

import os
import json
import time
import datetime
from typing import Any, Dict, Optional

import numpy as np
from dotenv import load_dotenv

# ===========================================================================
# 补丁 1：Qdrant 向量库 → 内存模式（无需 Docker / 真实服务）
# ===========================================================================
import hello_agents.memory.storage.qdrant_store as _qdrant_store
from qdrant_client import QdrantClient


def _patched_initialize_client(self):
    self.client = QdrantClient(location=":memory:")
    self.client.get_collections()
    self._ensure_collection()


_qdrant_store.QdrantVectorStore._initialize_client = _patched_initialize_client

# ===========================================================================
# 补丁 2：SemanticMemory 的 Neo4j 图库不可用时降级为 None
# ===========================================================================
import logging as _logging
import hello_agents.memory.types.semantic as _sem

_orig_init_db = _sem.SemanticMemory._init_databases


def _patched_init_db(self):
    try:
        _orig_init_db(self)
    except Exception as e:  # noqa: BLE001
        if getattr(self, "vector_store", None) is None:
            raise
        _logging.getLogger(__name__).warning("Neo4j 不可用，降级为纯向量记忆: %s", e)
        self.graph_store = None


_sem.SemanticMemory._init_databases = _patched_init_db

# ===========================================================================
# 补丁 3：中文感知哈希嵌入器（384 维，无需训练）
# ===========================================================================
import hashlib as _hashlib
import re as _re

import hello_agents.memory.embedding as _emb


class _HashingEmbedding:
    def __init__(self, dim: int = 384):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def _features(self, text):
        text = str(text)
        feats = set()
        chars = [c for c in text if not c.isspace()]
        for c in chars:
            feats.add(("c", c))
        for i in range(len(chars) - 1):
            feats.add(("b", chars[i] + chars[i + 1]))
        for w in _re.findall(r"[A-Za-z0-9]+", text.lower()):
            feats.add(("w", w))
        return feats

    def encode(self, texts):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        out = []
        for t in texts:
            vec = np.zeros(self._dim, dtype=np.float32)
            for f in self._features(t):
                h = int(_hashlib.md5(str(f).encode()).hexdigest(), 16)
                vec[h % self._dim] += 1.0 if (h & 1) else -1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            out.append(vec)
        return out[0] if single else out


_embedder_instance = None


def _patched_get_text_embedder():
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = _HashingEmbedding(dim=384)
    return _embedder_instance


_emb.get_text_embedder = _patched_get_text_embedder

# 记忆类型模块用 `from ... import get_text_embedder` 按引用导入，
# 改模块属性无效，需把引用重绑到各消费模块上（含 rag.pipeline）。
import importlib

for _m in [
    "hello_agents.memory.types.episodic",
    "hello_agents.memory.types.semantic",
    "hello_agents.memory.types.working",
    "hello_agents.memory.types.base",
    "hello_agents.memory.manager",
    "hello_agents.memory.rag.pipeline",
]:
    try:
        _mod = importlib.import_module(_m)
        if hasattr(_mod, "get_text_embedder"):
            _mod.get_text_embedder = _patched_get_text_embedder
    except Exception:
        pass

# 加载 .env 中的大模型配置（补丁全部就位后再导入工具并实例化）
load_dotenv()

from hello_agents.tools import MemoryTool, RAGTool  # noqa: E402


class PDFLearningAssistant:
    """智能文档问答助手"""

    def __init__(self, user_id: str = "default_user"):
        """初始化学习助手

        Args:
            user_id: 用户ID，用于隔离不同用户的数据
        """
        self.user_id = user_id
        self.session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 初始化工具（补丁已就位，会走内存 Qdrant + 哈希嵌入）
        self.memory_tool = MemoryTool(user_id=user_id)
        self.rag_tool = RAGTool(rag_namespace=f"pdf_{user_id}")

        # 学习统计
        self.stats = {
            "session_start": datetime.datetime.now(),
            "documents_loaded": 0,
            "questions_asked": 0,
            "concepts_learned": 0,
        }

        # 当前加载的文档
        self.current_document = None

    def load_document(self, pdf_path: str) -> Dict[str, Any]:
        """加载 PDF 文档到知识库

        Args:
            pdf_path: PDF 文件路径

        Returns:
            Dict: 包含 success 和 message 的结果
        """
        if not pdf_path or not os.path.exists(pdf_path):
            return {"success": False, "message": f"文件不存在: {pdf_path}"}

        start_time = time.time()

        # 【RAGTool】处理 PDF: MarkItDown 转换 → 智能分块 → 向量化
        # 注：0.2.8 的 execute 返回字符串（成功以 ✅ 开头）
        result = self.rag_tool.execute(
            "add_document",
            file_path=pdf_path,
            chunk_size=1000,
            chunk_overlap=200,
        )

        process_time = time.time() - start_time

        if result.startswith("✅"):
            self.current_document = os.path.basename(pdf_path)
            self.stats["documents_loaded"] += 1

            # 【MemoryTool】记录到情景记忆
            self.memory_tool.execute(
                "add",
                content=f"加载了文档《{self.current_document}》",
                memory_type="episodic",
                importance=0.9,
                event_type="document_loaded",
                session_id=self.session_id,
            )

            return {
                "success": True,
                "message": f"加载成功！(耗时: {process_time:.1f}秒)\n{result}",
                "document": self.current_document,
            }
        else:
            return {
                "success": False,
                "message": f"加载失败:\n{result}",
            }

    def ask(self, question: str, use_advanced_search: bool = True) -> str:
        """向文档提问

        Args:
            question: 用户问题
            use_advanced_search: 是否使用高级检索（MQE + HyDE）

        Returns:
            str: 答案
        """
        if not self.current_document:
            return "⚠️ 请先加载文档！"

        # 【MemoryTool】记录问题到工作记忆
        self.memory_tool.execute(
            "add",
            content=f"提问: {question}",
            memory_type="working",
            importance=0.6,
            session_id=self.session_id,
        )

        # 【RAGTool】使用高级检索获取答案
        answer = self.rag_tool.execute(
            "ask",
            question=question,
            limit=5,
            enable_advanced_search=use_advanced_search,
            enable_mqe=use_advanced_search,
            enable_hyde=use_advanced_search,
        )

        # 【MemoryTool】记录到情景记忆
        self.memory_tool.execute(
            "add",
            content=f"关于'{question}'的学习",
            memory_type="episodic",
            importance=0.7,
            event_type="qa_interaction",
            session_id=self.session_id,
        )

        self.stats["questions_asked"] += 1

        return answer

    def add_note(self, content: str, concept: Optional[str] = None) -> str:
        """添加学习笔记（保存到语义记忆）

        Args:
            content: 笔记内容
            concept: 关联概念（可选）

        Returns:
            str: 操作结果
        """
        if not content or not content.strip():
            return "⚠️ 笔记内容不能为空"

        result = self.memory_tool.execute(
            "add",
            content=content,
            memory_type="semantic",
            importance=0.8,
            concept=concept or "general",
            session_id=self.session_id,
        )
        self.stats["concepts_learned"] += 1
        return result

    def recall(self, query: str, limit: int = 5) -> str:
        """回顾学习历程（从记忆系统检索）

        Args:
            query: 检索关键词
            limit: 返回条数

        Returns:
            str: 检索结果
        """
        if not query or not query.strip():
            return "⚠️ 请输入回顾关键词"
        return self.memory_tool.execute("search", query=query, limit=limit)

    def get_stats(self) -> Dict[str, Any]:
        """获取学习统计"""
        duration = (datetime.datetime.now() - self.stats["session_start"]).total_seconds()
        return {
            "会话时长": f"{duration:.0f}秒",
            "加载文档": self.stats["documents_loaded"],
            "提问次数": self.stats["questions_asked"],
            "学习笔记": self.stats["concepts_learned"],
            "当前文档": self.current_document or "未加载",
        }

    def generate_report(self, save_to_file: bool = True) -> Dict[str, Any]:
        """生成学习报告"""
        memory_summary = self.memory_tool.execute("summary", limit=10)
        rag_stats = self.rag_tool.execute("stats")

        duration = (datetime.datetime.now() - self.stats["session_start"]).total_seconds()
        report = {
            "session_info": {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "start_time": self.stats["session_start"].isoformat(),
                "duration_seconds": duration,
            },
            "learning_metrics": {
                "documents_loaded": self.stats["documents_loaded"],
                "questions_asked": self.stats["questions_asked"],
                "concepts_learned": self.stats["concepts_learned"],
            },
            "memory_summary": memory_summary,
            "rag_status": rag_stats,
        }

        if save_to_file:
            report_file = f"learning_report_{self.session_id}.json"
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            report["report_file"] = report_file

        return report


# 兼容直接运行：快速自测（用 txt 验证链路，不依赖外部 PDF）
if __name__ == "__main__":
    a = PDFLearningAssistant(user_id="demo")
    txt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_doc.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write("# 大语言模型简介\n大语言模型（LLM）是基于海量文本训练的神经网络模型...\n")
    print(a.load_document(txt))
    print(a.ask("什么是大语言模型？"))
    print(a.add_note("LLM 即 Large Language Model，大语言模型", concept="LLM"))
    print("STATS:", a.get_stats())
    rep = a.generate_report()
    print("REPORT FILE:", rep.get("report_file"))
