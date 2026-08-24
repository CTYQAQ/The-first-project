from hello_agents import SimpleAgent, HelloAgentsLLM, ToolRegistry
from hello_agents.tools import MemoryTool
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 补丁：框架的 Qdrant 向量库默认连 localhost:6333 真实服务，本机没有 Docker /
# Qdrant 二进制，起不了服务。这里把“连本地服务”替换成 qdrant-client 自带的
# 内存模式（location=":memory:"），无需任何外部服务即可体验 MemoryTool。
# ---------------------------------------------------------------------------
import hello_agents.memory.storage.qdrant_store as _qdrant_store
from qdrant_client import QdrantClient


def _patched_initialize_client(self):
    self.client = QdrantClient(location=":memory:")
    self.client.get_collections()
    self._ensure_collection()


_qdrant_store.QdrantVectorStore._initialize_client = _patched_initialize_client

# ---------------------------------------------------------------------------
# 补丁2：SemanticMemory 还需要 Neo4j 图数据库，本机同样没服务。
# 让 Neo4j 初始化失败时降级为 None（跳过知识图谱功能，纯文本记忆的
# 增删查依旧可用）。仅当 Qdrant 已就绪时才降级，否则原样抛错。
# ---------------------------------------------------------------------------
import logging as _logging
import hello_agents.memory.types.semantic as _sem

_orig_init_db = _sem.SemanticMemory._init_databases


def _patched_init_db(self):
    try:
        _orig_init_db(self)
    except Exception as e:
        if getattr(self, "vector_store", None) is None:
            raise
        _logging.getLogger(__name__).warning("Neo4j 不可用，降级为纯向量记忆: %s", e)
        self.graph_store = None


_sem.SemanticMemory._init_databases = _patched_init_db

# ---------------------------------------------------------------------------
# 补丁3：框架自带的 TF-IDF 嵌入器必须先 fit 语料才能 encode，而增量写入不会
# 触发训练，导致 add/search 全部失败。这里用一个「无需训练」的中文感知哈希
# 嵌入器（字符/词 n-gram 哈希到 384 维并归一化），让中文检索也能命中。
# ---------------------------------------------------------------------------
import hashlib
import re
import numpy as np
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
        for w in re.findall(r"[A-Za-z0-9]+", text.lower()):
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
                h = int(hashlib.md5(str(f).encode()).hexdigest(), 16)
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
# 改模块属性无效，需把引用重绑到各消费模块上。
import importlib

for _m in [
    "hello_agents.memory.types.episodic",
    "hello_agents.memory.types.semantic",
    "hello_agents.memory.types.working",
    "hello_agents.memory.types.base",
    "hello_agents.memory.manager",
]:
    try:
        _mod = importlib.import_module(_m)
        if hasattr(_mod, "get_text_embedder"):
            _mod.get_text_embedder = _patched_get_text_embedder
    except Exception:
        pass

# 加载 .env 中的大模型配置
load_dotenv()

# 创建LLM实例
llm = HelloAgentsLLM()

# 创建记忆工具
memory_tool = MemoryTool(user_id="user123")
tool_registry = ToolRegistry()
tool_registry.register_tool(memory_tool)

# 创建具有记忆能力的Agent
agent = SimpleAgent(name="记忆助手", llm=llm, tool_registry=tool_registry)

# 体验记忆功能
print("=== 添加多个记忆 ===")

# 添加第一个记忆
result1 = memory_tool.run({"action": "add", "content": "用户张三是一名Python开发者，专注于机器学习和数据分析", "memory_type": "semantic", "importance": 0.8})
print(f"记忆1: {result1}")

# 添加第二个记忆
result2 = memory_tool.run({"action": "add", "content": "李四是前端工程师，擅长React和Vue.js开发", "memory_type": "semantic", "importance": 0.7})
print(f"记忆2: {result2}")

# 添加第三个记忆
result3 = memory_tool.run({"action": "add", "content": "王五是产品经理，负责用户体验设计和需求分析", "memory_type": "semantic", "importance": 0.6})
print(f"记忆3: {result3}")

print("\n=== 搜索特定记忆 ===")
# 搜索前端相关的记忆
print("🔍 搜索 '前端工程师':")
result = memory_tool.run({"action": "search", "query": "前端工程师", "limit": 3})
print(result)

print("\n=== 记忆摘要 ===")
result = memory_tool.run({"action": "summary"})
print(result)
