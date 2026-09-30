# Simple 心情小站 RAG

这是一个保留会话记忆、由个人知识库增强回答的中文情绪陪伴应用。项目结构参考 `P4_RAG项目案例`，原文件 `01.API调用.py` 保持不改动。

## 环境准备

项目使用 Python 3.10，并在 PowerShell 中设置 DashScope 密钥：

```powershell
conda activate nlpbase
pip install -r requirements.txt
$env:DASHSCOPE_API_KEY="你的 DashScope API Key"
```

## 启动

知识库上传页与聊天页是两个独立 Streamlit 入口，请分别在两个终端运行：

```powershell
streamlit run app_file_upload.py
```

```powershell
streamlit run app_qa.py
```

上传页读取 UTF-8 编码的 TXT 文件。首次上传后，向量数据保存在 `chroma_db/`，相同内容不会重复写入。聊天会检索最相关的三个片段，并结合昵称、性格和当前会话历史生成回复。

## 数据与安全

`data/` 中提供了六份原创的日常陪伴资料，仅用于一般情绪支持、学习压力、人际沟通、作息放松和紧急求助提醒，不用于医疗诊断，也不能替代专业心理服务或现实中的紧急救援。

