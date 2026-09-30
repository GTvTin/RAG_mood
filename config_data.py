"""
项目统一配置模块。

其他 Python 文件通过 `import config_data as config` 读取这里的配置，
这样模型名称、检索数量、数据库路径等参数只需要在一个地方维护。
"""

from pathlib import Path


# 当前项目根目录。
# 使用 Path(__file__) 计算绝对路径，避免从不同工作目录启动 Streamlit 时找不到文件。
BASE_DIR = Path(__file__).resolve().parent

# 知识库相关配置
# md5_path：记录已经导入过的文本内容，防止同一份资料被重复写入向量库。
# collection_name：Chroma 中集合的名称，可以理解成数据库里的“表名”。
# persist_directory：Chroma 向量数据和索引的本地保存目录。
md5_path = str(BASE_DIR / "md5.text")
collection_name = "rag"
persist_directory = str(BASE_DIR / "chroma_db")

# 文本切分配置
# 长文本会按 chunk_size 切分为多个片段，相邻片段保留 chunk_overlap 个字符，
# 这样可以减少切分位置造成的信息断裂。
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""]
max_split_char_number = 1000

# 检索与模型配置
# similarity_threshold 表示检索器返回的相关文档数量，这里设置为返回最相近的 3 段。
similarity_threshold = 3  # Kept for compatibility with the example.

# DashScopeEmbeddings 使用该模型把文本转换为向量。
embedding_model_name = "text-embedding-v4"

# ChatTongyi 使用该对话模型根据问题和检索结果生成回答。
chat_model_name = "qwen3-max"

# 本地持久化目录
# chat_history_path：LangChain 对话消息文件，用于给模型提供历史上下文。
# session_path：页面侧保存的完整会话 JSON，用于恢复聊天记录和朋友设置。
# resources_path：Logo 等页面资源目录。
chat_history_path = str(BASE_DIR / "chat_history")
session_path = str(BASE_DIR / "sessions")
resources_path = str(BASE_DIR / "resources")
