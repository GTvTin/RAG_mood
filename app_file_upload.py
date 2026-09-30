"""
基于 Streamlit 的知识库上传页面。

页面职责：
1. 接收 UTF-8 编码的 TXT 文件；
2. 显示文件名、类型和大小；
3. 调用 KnowledgeBaseService 对文本去重、切分、向量化并保存到 Chroma；
4. 将处理结果显示给用户。

Streamlit 的特点：页面交互会触发整个脚本重新执行，
因此需要长期保存的对象通常放在 st.session_state 中。
"""
import time

import streamlit as st
from knowledge_base import KnowledgeBaseService

# 页面标题
st.title("知识库更新服务")

# 创建文件上传控件。
# 当前只允许一次上传一个 TXT 文件。
uploader_file = st.file_uploader(
    "请上传TXT文件",
    type=['txt'],
    accept_multiple_files=False,    # False表示仅接受一个文件的上传
)

# st.session_state 可以理解成和当前浏览器会话绑定的字典。
# 页面重跑时，如果 service 已存在就继续复用，避免反复创建向量库连接。
if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()


# 只有用户真正选择并上传文件后才执行入库逻辑。
if uploader_file is not None:
    # 读取上传控件提供的文件信息。
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size / 1024    # KB

    # 在页面中显示文件摘要。
    st.subheader(f"文件名：{file_name}")
    st.write(f"格式：{file_type} | 大小：{file_size:.2f} KB")

    # get_value() 返回 bytes，需要按 UTF-8 解码成 Python 字符串。
    text = uploader_file.getvalue().decode("utf-8")

    # spinner 在文本向量化和写入 Chroma 期间显示加载动画。
    with st.spinner("载入知识库中。。。"):
        # 这里会依次执行 MD5 去重、文本切分、DashScope 向量化和 Chroma 入库。
        time.sleep(1)
        result = st.session_state["service"].upload_by_str(text, file_name)
        st.write(result)




