"""
知识入库模块。

完整流程：
1. 接收上传页读取到的 UTF-8 文本；
2. 计算文本的 MD5，检查是否已经导入过；
3. 长文本使用 RecursiveCharacterTextSplitter 切分成多个片段；
4. 通过 DashScopeEmbeddings 把每个片段转换为向量；
5. 将文本、向量和元数据一起保存到 Chroma。
"""

from datetime import datetime
import hashlib
import os

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config_data as config


def check_md5(md5_str: str) -> bool:
    """检查一份资料是否已经写入过知识库。

    :param md5_str: 由资料正文计算得到的 MD5 字符串
    :return: False 表示未处理过；True 表示已经处理过
    """
    if not os.path.exists(config.md5_path):
        # 记录文件不存在说明这是第一次导入，先创建空文件。
        open(config.md5_path, 'w', encoding='utf-8').close()
        return False
    else:
        # 每行保存一个已经处理过的 MD5，逐行比较即可判断是否重复。
        for line in open(config.md5_path, 'r', encoding='utf-8').readlines():
            # 去掉文件换行符和前后空格，避免格式差异影响比较。
            line = line.strip()
            if line == md5_str:
                return True

        return False


def save_md5(md5_str: str) :
    """将成功导入的 MD5 追加到记录文件。

    只有在内容真正写入 Chroma 之后才调用该函数，避免入库失败却误判为已处理。
    """
    with open(config.md5_path, "a", encoding="utf-8") as file:
        file.write(md5_str + "\n")


def get_string_md5(input_str: str, encoding: str = "utf-8"):
    """把字符串转换为 MD5 十六进制字符串。

    :param input_str: 需要计算的内容
    :param encoding: 字符编码，默认使用 UTF-8
    :return: 32 位十六进制 MD5 字符串
    """

    # hashlib 只能处理字节，因此先把字符串编码为 bytes。
    str_bytes = input_str.encode(encoding=encoding)

    # 创建 MD5 对象并传入字节内容。
    md5_obj = hashlib.md5()
    md5_obj.update(str_bytes)

    # hexdigest() 返回方便保存和比较的十六进制字符串。
    md5_hex = md5_obj.hexdigest()

    return md5_hex


class KnowledgeBaseService:
    """封装知识库的初始化、文本切分和入库操作。"""

    def __init__(self, embedding=None):
        """创建 Chroma 向量库和文本切分器。

        :param embedding: 可选的嵌入模型。
            正常运行时为 None，函数会自动创建 DashScopeEmbeddings；
            离线测试时可以传入假模型，避免调用真实接口。
        """
        # exist_ok=True：目录存在时不报错，不存在时自动创建。
        os.makedirs(config.persist_directory, exist_ok=True)

        if embedding is None:
            # DashScopeEmbeddings 默认从环境变量 DASHSCOPE_API_KEY 读取密钥。
            embedding = DashScopeEmbeddings(model=config.embedding_model_name)

        # Chroma 负责保存文本向量和原始文本，后续可按照相似度搜索。
        self.chroma = Chroma(
            collection_name=config.collection_name,
            embedding_function=embedding,
            persist_directory=config.persist_directory,
        )

        # 优先按自然段和句子切分，尽量避免直接把一句话从中间截断。
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )

    def upload_by_str(self, data: str, filename: str) -> str:
        """将一份文本向量化并保存到 Chroma。

        :param data: 从 TXT 文件中读取出的完整文本
        :param filename: 原始文件名，会作为资料来源写入 metadata
        :return: 面向上传页显示的成功或跳过提示
        """
        # 去掉正文首尾无意义的空白，再判断内容是否为空。
        data = data.strip()
        if not data:
            return "[跳过]内容为空，未写入知识库"

        # 先通过 MD5 判断是否重复，避免重复调用嵌入接口产生额外费用。
        md5_hex = get_string_md5(data)
        if check_md5(md5_hex):
            return "[跳过]内容已经存在知识库中"

        # 长文本必须切分：
        # 1. 单次向量化内容更聚焦；
        # 2. 后续检索可以只返回最相关的片段，而不是整份长文档。
        if len(data) > config.max_split_char_number:
            knowledge_chunks = self.splitter.split_text(data)
        else:
            knowledge_chunks = [data]

        # metadata 会跟随每个文本片段保存，用于说明片段来自哪个文件、何时导入。
        metadata = {
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": "Simple",
        }

        # add_texts 内部会调用 embedding_function，把文本转换为向量后写入 Chroma。
        self.chroma.add_texts(
            knowledge_chunks,
            metadatas=[metadata.copy() for _ in knowledge_chunks],
        )

        # 写入成功后才记录 MD5，保证重复判断和实际入库状态一致。
        save_md5(md5_hex)
        return "[成功]内容已经成功载入向量库"
