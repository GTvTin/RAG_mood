"""
Simple 心情小站聊天页面。

页面同时维护两份与“会话状态”有关的数据：
1. sessions/*.json：保存页面显示需要的信息，如消息、昵称、朋友性格；
2. chat_history/*：保存 LangChain 对话历史，供大模型理解上下文。

RagService 将用户问题、知识库检索结果和对话历史组合后交给 ChatTongyi。
"""

import json
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

import config_data as config
from file_history_store import FileChatMessageHistory
from rag import RagService


# 必须尽早调用，设置浏览器标题、图标、宽屏布局和侧边栏初始状态。
st.set_page_config(
    page_title="Simple 心情小站",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "# 🌷 Simple 的心情小站\n\n随时找我聊聊，我会认真听，用心答。"
    },
)


def current_session_name() -> str:
    """使用当前时间生成新的会话名称。

    例如 2026-09-15_00-30-20。
    时间字符串既方便排序，也可以直接作为本地 JSON 文件名。
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def session_file_path(session_name: str) -> Path:
    """根据会话名称得到对应的页面会话 JSON 路径。"""
    return Path(config.session_path) / f"{session_name}.json"


def save_session() -> None:
    """把当前页面消息、昵称和性格保存到 sessions/*.json。"""
    # 没有当前会话时不做任何操作。
    if not st.session_state.current_session:
        return

    # 首次保存时 sessions 目录可能尚未创建。
    Path(config.session_path).mkdir(parents=True, exist_ok=True)

    # 字典结构与 load_session() 读取时使用的字段一一对应。
    session_data = {
        "current_session": st.session_state.current_session,
        "messages": st.session_state.messages,
        "nick_name": st.session_state.nick_name,
        "nature": st.session_state.nature,
    }

    # ensure_ascii=False 确保中文直接可读；
    # indent=2 让 JSON 文件具有更清晰的层级。
    with open(
        session_file_path(st.session_state.current_session),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(session_data, file, ensure_ascii=False, indent=2)


def load_sessions() -> list[str]:
    """读取 sessions 目录中的全部会话名称，并按时间倒序返回。"""
    storage = Path(config.session_path)
    if not storage.exists():
        return []

    # path.stem 会去掉 .json 后缀，得到纯会话名称。
    return sorted(
        (path.stem for path in storage.glob("*.json")),
        reverse=True,
    )


def sync_history_from_state() -> None:
    """将页面展示消息同步为 LangChain 历史消息。

    页面消息是普通字典，LangChain 需要 HumanMessage 和 AIMessage 对象，
    因此这里先转换类型，再覆盖当前会话对应的 chat_history 文件。
    """
    history = FileChatMessageHistory(
        st.session_state.current_session,
        config.chat_history_path,
    )
    messages = []
    for message in st.session_state.messages:
        # 页面的 role="user" 对应 HumanMessage。
        if message["role"] == "user":
            messages.append(HumanMessage(content=message["content"]))
        # 页面的 role="assistant" 对应 AIMessage。
        elif message["role"] == "assistant":
            messages.append(AIMessage(content=message["content"]))
    history.replace_messages(messages)


def load_session(session_name: str) -> None:
    """从 JSON 恢复历史会话，并同步 LangChain 的历史记录。"""
    try:
        with open(session_file_path(session_name), "r", encoding="utf-8") as file:
            session_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        # 文件被删除或内容损坏时给出页面提示，不让程序直接崩溃。
        st.error("加载会话信息出错")
        return

    # 把 JSON 中的内容恢复到 Streamlit Session State。
    st.session_state.messages = session_data.get("messages", [])
    st.session_state.nick_name = session_data.get("nick_name", "猫猫")
    st.session_state.nature = session_data.get(
        "nature",
        "温柔、体贴、幽默、风趣",
    )
    st.session_state.current_session = session_name

    # 页面状态恢复完成后，让 LangChain 历史与该会话保持一致。
    sync_history_from_state()


def delete_session(session_name: str) -> None:
    """删除指定会话的页面 JSON 和 LangChain 历史文件内容。"""
    try:
        # 删除页面侧会话文件。
        session_file_path(session_name).unlink()
    except FileNotFoundError:
        pass

    # 清空 LangChain 历史文件。
    FileChatMessageHistory(
        session_name,
        config.chat_history_path,
    ).clear()

    # 如果删除的是当前正在使用的会话，就立即切换到新会话并清空页面消息。
    if session_name == st.session_state.current_session:
        st.session_state.messages = []
        st.session_state.current_session = current_session_name()
        FileChatMessageHistory(
            st.session_state.current_session,
            config.chat_history_path,
        ).clear()


# 初始化页面所需的 Session State。
# `not in` 判断保证页面重跑时不会把已有聊天记录重置。
if "messages" not in st.session_state:
    st.session_state.messages = []
if "nick_name" not in st.session_state:
    st.session_state.nick_name = "猫猫"
if "nature" not in st.session_state:
    st.session_state.nature = "温柔、体贴、幽默、风趣"
if "current_session" not in st.session_state:
    st.session_state.current_session = current_session_name()

# RagService 会创建向量库连接、提示词模板和 ChatTongyi 模型。
# 它只初始化一次，之后每个页面重跑都会复用该对象。
if "rag" not in st.session_state:
    st.session_state.rag = RagService()

# 页面标题和 Logo。
st.title("🌷 Simple 的心情小站")
logo_path = Path(config.resources_path) / "logo.jpg"
if logo_path.exists():
    st.logo(str(logo_path))

# 把内部会话名转换成更容易阅读的时间文本。
st.caption(
    "本次会话开始于 "
    + st.session_state.current_session.replace("_", " ").replace("-", ":")
)

# 遍历页面消息，根据 role 显示用户气泡或 AI 气泡。
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

# 侧边栏负责会话管理和聊天对象设置。
with st.sidebar:
    st.subheader("Simple 的心情日记本")

    # 新建会话前先保存旧会话，再清空页面消息和 LangChain 历史。
    if st.button("🌱 开始新的倾诉", width="stretch"):
        save_session()
        st.session_state.messages = []
        st.session_state.current_session = current_session_name()
        FileChatMessageHistory(
            st.session_state.current_session,
            config.chat_history_path,
        ).clear()
        save_session()
        st.rerun()

    # 显示已经保存的会话，支持加载和删除。
    st.text("📱 过往聊天")
    for session in load_sessions():
        load_col, delete_col = st.columns([4, 1])

        with load_col:
            is_current = session == st.session_state.current_session
            # 当前会话按钮使用 primary 样式，其他会话使用 secondary 样式。
            if st.button(
                session,
                width="stretch",
                icon="🩷",
                key=f"load_{session}",
                type="primary" if is_current else "secondary",
            ):
                load_session(session)
                st.rerun()

        with delete_col:
            # 删除后 rerun，让侧边栏列表立即刷新。
            if st.button(
                "",
                width="stretch",
                icon="❌️",
                key=f"delete_{session}",
                help="删除这段会话",
            ):
                delete_session(session)
                st.rerun()

    st.divider()
    st.subheader("好朋友")

    # 昵称和性格会作为变量传入 RagService 的提示词模板。
    nick_name = st.text_input(
        "朋友昵称",
        placeholder="请输入朋友昵称",
        value=st.session_state.nick_name,
    )
    if nick_name:
        st.session_state.nick_name = nick_name

    nature = st.text_area(
        "朋友性格",
        placeholder="请输入朋友性格",
        value=st.session_state.nature,
    )
    if nature:
        st.session_state.nature = nature


# chat_input 固定在页面底部，用户提交后返回值才会从 None 变为问题文本。
prompt = st.chat_input("💬 跟我说说你的烦恼吧...")
if not st.session_state.messages:
    st.info("你好 Simple，我随时在这里，有什么想聊的都可以告诉我。")

if prompt:
    # 先把用户消息显示并加入页面状态。
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 流式返回的每个文本片段暂存到这里，结束后合并为完整回复。
    ai_response_parts = []

    # RunnableWithMessageHistory 使用该配置确定要读写哪个会话历史。
    session_config = {
        "configurable": {
            "session_id": st.session_state.current_session,
        }
    }

    # 进入 RAG 链：
    # 问题 -> 向量检索 -> 拼接资料和提示词 -> ChatTongyi -> 流式文本。
    with st.spinner("AI 思考中..."):
        response = st.session_state.rag.chain.stream(
            {
                "input": prompt,
                "nick_name": st.session_state.nick_name,
                "nature": st.session_state.nature,
            },
            session_config,
        )

        def capture(generator, cache):
            """一边把文本片段交给页面显示，一边缓存起来用于最终保存。"""
            for chunk in generator:
                cache.append(chunk)
                yield chunk

        # write_stream 会把生成器中的每个片段立即显示到聊天窗口。
        st.chat_message("assistant").write_stream(capture(response, ai_response_parts))

        # 流结束后把全部片段拼接为完整回答，更新页面记录并保存 JSON。
        st.session_state.messages.append({"role": "assistant", "content": "".join(ai_response_parts)})
        save_session()
