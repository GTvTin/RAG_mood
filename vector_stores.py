"""
向量检索服务。

知识入库由 KnowledgeBaseService 完成；本模块只负责连接同一个 Chroma 集合，
并把向量库包装成 LangChain Retriever，供 RAG 执行链调用。
"""

from langchain_chroma import Chroma
import config_data as config


class VectorStoreService(object):
    """连接 Chroma，并对外提供文档检索器。"""

    def __init__(self, embedding):
        """初始化向量库连接。

        :param embedding: 查询和文档共用的嵌入模型。
            必须和知识入库时使用的模型保持一致，否则向量空间不匹配。
        """
        self.embedding = embedding

        # 使用与知识入库相同的集合名和持久化目录，因此可以读取已经上传的数据。
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def get_retriever(self):
        """返回 LangChain 检索器。

        search_kwargs={"k": ...} 表示每次提问返回相似度最高的 K 个文本片段。
        当前 config.similarity_threshold 的值为 3。
        """
        return self.vector_store.as_retriever(search_kwargs={"k": config.similarity_threshold})
