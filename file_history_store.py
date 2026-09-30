"""
文件版对话历史存储。

LangChain 的 RunnableWithMessageHistory 需要知道：
1. 当前使用哪个 session_id；
2. 如何读取旧消息；
3. 如何保存新加入的用户消息和 AI 消息。

本项目使用 JSON 文件实现上述能力，每个会话对应 chat_history/ 下的一个文件。
"""

import json
import os
from collections.abc import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import  BaseMessage, message_to_dict,messages_from_dict

import config_data as config


def get_history(session_id: str) :
    """根据会话 ID 创建消息历史对象。

    该函数会传给 RunnableWithMessageHistory 作为 history_factory，
    每次执行某个会话时，LangChain 都会通过它获得对应的历史文件。
    """
    return FileChatMessageHistory(session_id, config.chat_history_path)


class FileChatMessageHistory(BaseChatMessageHistory):
    """把 LangChain 消息列表序列化为 JSON，并持久化到本地文件。"""

    def __init__(self, session_id: str, storage_path: str):
        """确定当前会话历史文件的完整路径。"""
        self.session_id = str(session_id)
        self.storage_path = storage_path

        # 每个会话使用独立文件，避免不同聊天窗口的消息互相混入。
        self.file_path = os.path.join(self.storage_path, self.session_id)

        # 首次运行项目时目录可能不存在，因此创建文件前先创建父目录。
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    @property
    def messages(self) -> list[BaseMessage]:
        """读取当前会话的 LangChain 消息对象列表。"""
        try:
            with open(self.file_path, "r", encoding="utf-8") as file:
                # JSON 中保存的是字典，messages_from_dict 会还原成 HumanMessage、
                # AIMessage 等 LangChain 消息对象。
                return messages_from_dict(json.load(file))
        except FileNotFoundError:
            # 新会话还没有历史文件，返回空列表表示没有历史消息。
            return []

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """在已有历史后追加新消息，并同步写回文件。"""
        # 先读取旧消息，再把本次新增消息追加到列表末尾。
        all_messages = list(self.messages)
        all_messages.extend(messages)
        self.replace_messages(all_messages)

    def replace_messages(self, messages: Sequence[BaseMessage]) -> None:
        """用传入的完整消息列表覆盖当前会话文件。

        当页面从 sessions/*.json 恢复历史后，会调用该方法，
        让 LangChain 的内存历史与页面显示内容保持一致。
        """
        serialized = [message_to_dict(message) for message in messages]
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, "w", encoding="utf-8") as file:
            # ensure_ascii=False 让中文按原字符保存，便于直接查看 JSON 内容。
            json.dump(serialized, file, ensure_ascii=False)

    def clear(self) -> None:
        """清空当前会话的全部历史消息。"""
        self.replace_messages([])
