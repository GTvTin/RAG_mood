import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

import config_data as config
from file_history_store import FileChatMessageHistory, get_history
from knowledge_base import KnowledgeBaseService
from rag import RagService


class OfflineRagTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(self.temp_dir.name)
        config.md5_path = str(root / "md5.text")
        config.persist_directory = str(root / "chroma_db")
        config.chat_history_path = str(root / "chat_history")
        self.embedding = DeterministicFakeEmbedding(size=16)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_upload_and_md5_deduplication(self):
        service = KnowledgeBaseService(embedding=self.embedding)
        text = "考试前感到焦虑时，可以先拆分任务并安排短暂休息。"

        self.assertEqual(
            service.upload_by_str(text, "考试压力.txt"),
            "[成功]内容已经成功载入向量库",
        )
        self.assertEqual(
            service.upload_by_str(text, "重复文件.txt"),
            "[跳过]内容已经存在知识库中",
        )
        self.assertEqual(service.chroma._collection.count(), 1)

    def test_file_history_round_trip(self):
        history = FileChatMessageHistory("test_session", config.chat_history_path)
        history.replace_messages(
            [
                HumanMessage(content="最近学习压力很大"),
                AIMessage(content="先说说最让你疲惫的部分吧。"),
            ]
        )

        restored = get_history("test_session")
        self.assertEqual(len(restored.messages), 2)
        self.assertEqual(restored.messages[0].content, "最近学习压力很大")
        self.assertEqual(restored.messages[1].type, "ai")

    def test_rag_chain_retrieval_history_and_streaming(self):
        knowledge = KnowledgeBaseService(embedding=self.embedding)
        knowledge.upload_by_str(
            "面对学业压力，可以把大任务拆成二十到四十分钟可以启动的小步骤。",
            "学业压力与拖延.txt",
        )

        service = RagService(
            embedding=self.embedding,
            chat_model=FakeListChatModel(responses=["先把任务缩小成一个动作。"]),
        )
        session_config = {"configurable": {"session_id": "chain_session"}}
        payload = {
            "input": "学业压力很大，不知道从哪里开始",
            "nick_name": "猫猫",
            "nature": "温柔、耐心",
        }

        result = service.chain.invoke(payload, session_config)
        self.assertEqual(result, "先把任务缩小成一个动作。")

        streamed = "".join(
            service.chain.stream(
                {
                    "input": "可以再具体一点吗",
                    "nick_name": "猫猫",
                    "nature": "温柔、耐心",
                },
                session_config,
            )
        )
        self.assertEqual(streamed, "先把任务缩小成一个动作。")

        history = get_history("chain_session").messages
        self.assertEqual(len(history), 4)
        self.assertEqual(history[0].content, "学业压力很大，不知道从哪里开始")
        self.assertEqual(history[-1].type, "ai")


class StreamlitRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("DASHSCOPE_API_KEY", "offline-test-key")

    def test_qa_page_renders(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(
            str(Path(__file__).resolve().parents[1] / "app_qa.py")
        )
        app.run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertGreaterEqual(len(app.chat_input), 1)

    def test_qa_page_explains_missing_api_key(self):
        from streamlit.testing.v1 import AppTest

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DASHSCOPE_API_KEY", None)
            app = AppTest.from_file(
                str(Path(__file__).resolve().parents[1] / "app_qa.py")
            )
            app.run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(
            any("RAG 服务初始化失败" in error.value for error in app.error)
        )

    def test_upload_page_renders(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(
            str(Path(__file__).resolve().parents[1] / "app_file_upload.py")
        )
        app.run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertGreaterEqual(len(app.file_uploader), 1)


if __name__ == "__main__":
    unittest.main()
