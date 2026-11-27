# mypy: disable-error-code="arg-type,assignment,override"
# (the tests pass small fakes where the code expects database, Redis and service objects)
"""Model calls made while answering a chat prompt are recorded by that conversation's token counter.

These tests need no database, Redis or network: models built by the AICatalog are replaced by llama-index mock
models that keep the callback manager the AICatalog was given, and Redis vector stores are replaced by in-memory
ones. Run them with --noconftest.
"""

import asyncio
import uuid
from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import pytest
import sqlalchemy as sa
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.vector_stores.redis.base import NO_DOCS

from basejump.core.database.client.query import ClientQueryRecorder
from basejump.core.database.crud import crud_chat
from basejump.core.database.crud.utils import create_callback_mgrs
from basejump.core.database.result.store import LocalResultStore
from basejump.core.database.vector import utils as vector_utils
from basejump.core.models import enums
from basejump.core.models import schemas as sch
from basejump.core.models.ai import formats as fmt
from basejump.core.models.ai.catalog import AICatalog
from basejump.core.models.ai.formatter import JSONResponseFormatter
from basejump.core.service.agents import utils as agent_utils
from basejump.core.service.agents.memory import agent as memory_agent
from basejump.core.service.agents.tools import docs as docs_tool
from basejump.core.service.agents.tools.sql.retriever import TableRetrieverTool
from basejump.core.service.agents.tools.visualize import VisTool

PROMPT_TOKENS = 11
COMPLETION_TOKENS = 5


class TextVectorStore(SimpleVectorStore):
    """An in-memory stand-in for the Redis vector store, which stores text"""

    stores_text: bool = True


class FakeToolLLM(FunctionCallingLLM):
    """A function calling LLM that always calls the first tool it is given with tool_kwargs."""

    tool_kwargs: dict = {}
    max_tokens: Optional[int] = None

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_function_calling_model=True, is_chat_model=True, model_name="fake-llm")

    def _response(self) -> ChatResponse:
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="False"),
            raw={
                "model": "fake-llm",
                "usage": {"prompt_tokens": PROMPT_TOKENS, "completion_tokens": COMPLETION_TOKENS},
            },
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self._response()

    @llm_chat_callback()
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self._response()

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError

    @llm_completion_callback()
    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError

    def stream_chat(self, messages, **kwargs):
        raise NotImplementedError

    async def astream_chat(self, messages, **kwargs):
        raise NotImplementedError

    def stream_complete(self, prompt, formatted=False, **kwargs):
        raise NotImplementedError

    async def astream_complete(self, prompt, formatted=False, **kwargs):
        raise NotImplementedError

    def _prepare_chat_with_tools(self, tools, user_msg=None, chat_history=None, **kwargs):
        messages = list(chat_history or [])
        if user_msg:
            messages.append(user_msg if isinstance(user_msg, ChatMessage) else ChatMessage(content=user_msg))
        return {"messages": messages, "tools": tools}

    def get_tool_calls_from_response(self, response, error_on_no_tool_call=True, **kwargs):
        return [ToolSelection(tool_id="call_1", tool_name=self._tool_name, tool_kwargs=self.tool_kwargs)]

    async def achat_with_tools(self, tools, user_msg=None, chat_history=None, **kwargs):
        self._tool_name = tools[0].metadata.name
        return await super().achat_with_tools(tools, user_msg=user_msg, chat_history=chat_history, **kwargs)

    def chat_with_tools(self, tools, user_msg=None, chat_history=None, **kwargs):
        self._tool_name = tools[0].metadata.name
        return super().chat_with_tools(tools, user_msg=user_msg, chat_history=chat_history, **kwargs)


@pytest.fixture
def conversation() -> sch.CallbackMgrs:
    return create_callback_mgrs(enums.AIModelSchema.GPT4o)


@pytest.fixture
def other_conversation() -> sch.CallbackMgrs:
    """A second, concurrent conversation that must never receive the first conversation's usage."""
    return create_callback_mgrs(enums.AIModelSchema.GPT4o)


@pytest.fixture
def mock_models(monkeypatch):
    """Models built by an AICatalog report to the callback manager the AICatalog was given, as the real ones do."""
    llm_kwargs: dict = {}

    def get_llm(self, model_info):
        return FakeToolLLM(callback_manager=self.callback_manager, tool_kwargs=llm_kwargs.get("tool_kwargs", {}))

    def get_embedding_model(self, model_info):
        embed_model = MockEmbedding(embed_dim=8, callback_manager=self.callback_manager)
        # The token counter records the serialized model_name of each embedding event
        embed_model.model_name = enums.AIModelSchema.COHERE_EMBED_V4.value
        return embed_model

    monkeypatch.setattr(AICatalog, "get_llm", get_llm)
    monkeypatch.setattr(AICatalog, "get_embedding_model", get_embedding_model)
    return llm_kwargs


@pytest.fixture
def in_memory_vector_store(monkeypatch):
    monkeypatch.setattr(vector_utils, "get_redis_vector_store", lambda **kwargs: TextVectorStore())


def prompt_metadata(conversation: sch.CallbackMgrs) -> sch.PromptMetadata:
    return sch.PromptMetadata(
        initial_prompt="How many orders were placed last month?",
        prompt_uuid=uuid.uuid4(),
        prompt_id=42,
        model_name=enums.AIModelSchema.GPT4o,
        llm_type=enums.LLMType.DATA_AGENT,
        prompt_time=datetime.now(),
        user_id=7,
        user_uuid=uuid.uuid4(),
        client_id=3,
        client_uuid=uuid.uuid4(),
        user_role=enums.UserRoles.ADMIN.value,
        token_counter=conversation.token_counter,
        llama_debug=conversation.llama_debug,
        callback_manager=conversation.callback_manager,
    )


def model_info() -> sch.AWSModelInfo:
    endpoint_info = sch.AWSEndpointInfo(endpoint="bedrock", deployment_region="us-east-2")
    return sch.AWSModelInfo(model_name=enums.AIModelSchema.GPT_OSS_120B, endpoint_info=endpoint_info)


def service_context() -> SimpleNamespace:
    return SimpleNamespace(
        small_model_info=model_info(),
        large_model_info=model_info(),
        embedding_model_info=model_info(),
        redis_client_async=None,
        redis_client=None,
    )


def counts(token_counter: TokenCountingHandler) -> tuple[int, int]:
    return len(token_counter.llm_token_counts), len(token_counter.embedding_token_counts)


def assert_nothing_global(other_conversation: sch.CallbackMgrs) -> None:
    assert counts(other_conversation.token_counter) == (0, 0)
    assert not any(isinstance(handler, TokenCountingHandler) for handler in Settings.callback_manager.handlers)


# ---------------------------------------------------------------------------------------------------------------
# The llama-index behavior that hid embedding usage
# ---------------------------------------------------------------------------------------------------------------


def test_an_index_without_a_callback_manager_detaches_the_embed_model(conversation):
    """Documents why every index is built with the embed model's callback manager: if this starts failing,
    llama-index no longer overwrites it and the explicit callback_manager arguments are redundant but harmless."""
    embed_model = MockEmbedding(embed_dim=8, callback_manager=conversation.callback_manager)
    VectorStoreIndex.from_vector_store(TextVectorStore(), embed_model=embed_model)
    assert embed_model.callback_manager is not conversation.callback_manager


def test_get_vector_idx_keeps_the_embed_model_callback_manager(conversation, in_memory_vector_store):
    embed_model = MockEmbedding(embed_dim=8, callback_manager=conversation.callback_manager)
    vector_schema = SimpleNamespace(index_name="idx", vector_database_vendor=enums.VectorVendorType.REDIS)
    index = vector_utils.get_vector_idx(
        client_id=1, vector_schema=vector_schema, embed_model=embed_model, redis_client_async=None
    )
    asyncio.run(index.as_retriever(similarity_top_k=1).aretrieve("orders by month"))

    assert embed_model.callback_manager is conversation.callback_manager
    assert counts(conversation.token_counter) == (0, 1)


# ---------------------------------------------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------------------------------------------


def test_formatter_records_its_llm_call_once(conversation, other_conversation, mock_models):
    mock_models["tool_kwargs"] = {"true_false_bool": True}
    formatter = JSONResponseFormatter(
        response="Yes",
        pydantic_format=fmt.TrueFalseBool,
        small_model_info=model_info(),
        callback_manager=conversation.callback_manager,
    )
    extract = asyncio.run(formatter.format())

    assert extract.true_false_bool is True
    assert counts(conversation.token_counter) == (1, 0)
    event = conversation.token_counter.llm_token_counts[0]
    assert (event.prompt_token_count, event.completion_token_count) == (PROMPT_TOKENS, COMPLETION_TOKENS)
    assert_nothing_global(other_conversation)


def test_formatter_without_a_callback_manager_is_unchanged(conversation, mock_models):
    """Callers that pass no callback manager (indexing, API endpoints) keep their current behavior."""
    mock_models["tool_kwargs"] = {"true_false_bool": True}
    formatter = JSONResponseFormatter(response="Yes", pydantic_format=fmt.TrueFalseBool, small_model_info=model_info())
    asyncio.run(formatter.format())
    assert counts(conversation.token_counter) == (0, 0)


def test_title_description_uses_the_prompt_callback_manager(conversation, other_conversation, mock_models):
    mock_models["tool_kwargs"] = {"title": "Orders", "subtitle": "Orders last month", "description": "Orders"}
    extract = asyncio.run(
        agent_utils.get_title_description(
            db=None,
            prompt_metadata=prompt_metadata(conversation),
            sql_query="select 1",
            query_result="1",
            small_model_info=model_info(),
        )
    )
    assert extract.title == "Orders"
    assert counts(conversation.token_counter) == (1, 0)
    assert_nothing_global(other_conversation)


def test_contextualize_prompt_records_when_given_a_callback_manager(conversation, mock_models):
    mock_models["tool_kwargs"] = {"full_context_prompt": "Orders last month"}
    pair = sch.MessagePair(prompt=sch.ChatPrompt(prompt="Orders?"))
    prompt = asyncio.run(
        agent_utils.contextualize_prompt(
            recent_interactions=[pair], small_model_info=model_info(), callback_manager=conversation.callback_manager
        )
    )
    assert prompt == "Orders last month"
    assert counts(conversation.token_counter) == (1, 0)


def test_metric_formatting_in_a_worker_thread_is_recorded(conversation, other_conversation, mock_models):
    """ClientQueryRecorder stores results in a worker thread; the formatting call there is still recorded."""
    mock_models["tool_kwargs"] = {"metric_value": "4000", "metric_value_formatted": "$4,000"}

    class RecordingStore(LocalResultStore):
        def store(self, file_info, result, small_model_info, initial_prompt, sql_query, callback_manager=None):
            import io

            self.get_metric_value(
                small_model_info=small_model_info,
                initial_prompt=initial_prompt,
                sql_query=sql_query,
                buffer=io.BytesIO(b"4000"),
                callback_manager=callback_manager,
            )
            return self.metric_value_formatted

        def get_file_info(self, result_uuid):
            return None

    recorder = ClientQueryRecorder.__new__(ClientQueryRecorder)
    recorder.client_id = 3
    recorder.initial_prompt = "Total revenue?"
    recorder.small_model_info = model_info()
    recorder.result_store = RecordingStore(client_id=3)
    recorder.callback_manager = conversation.callback_manager
    recorder.__dict__["sql_query"] = "select 4000 as revenue"
    recorder._client_engine = sa.create_engine("sqlite://")

    assert asyncio.run(recorder.astore_query_result()) == "$4,000"
    assert counts(conversation.token_counter) == (1, 0)
    assert_nothing_global(other_conversation)


def test_sub_question_decision_is_recorded(conversation, other_conversation, mock_models):
    """use_sub_questions makes a chat call and a formatting call with its own LLM: both are recorded once."""
    mock_models["tool_kwargs"] = {"true_false_bool": False}
    tool = TableRetrieverTool.__new__(TableRetrieverTool)
    tool.service_context = service_context()
    tool.prompt_metadata = prompt_metadata(conversation)

    assert asyncio.run(tool.use_sub_questions(prompt="Orders last month")) == []
    assert counts(conversation.token_counter) == (2, 0)
    assert len({event.event_id for event in conversation.token_counter.llm_token_counts}) == 2
    assert_nothing_global(other_conversation)


# ---------------------------------------------------------------------------------------------------------------
# Embedding calls
# ---------------------------------------------------------------------------------------------------------------


def test_table_retrieval_query_embedding_is_recorded(
    conversation, other_conversation, mock_models, in_memory_vector_store, monkeypatch
):
    async def get_vector_connection_from_id(db, vector_id):
        return SimpleNamespace(
            vector_database_vendor=enums.VectorVendorType.REDIS,
            vector_datasource_type=enums.VectorSourceType.TABLE,
            index_name="idx",
        )

    from basejump.core.service.agents.tools.sql import retriever

    monkeypatch.setattr(retriever.crud_connection, "get_vector_connection_from_id", get_vector_connection_from_id)
    tool = TableRetrieverTool.__new__(TableRetrieverTool)
    tool.db = None
    tool.service_context = service_context()
    tool.prompt_metadata = prompt_metadata(conversation)

    table_index = asyncio.run(tool.setup_sql_table_vector_index(vector_id=1, client_id=3))
    asyncio.run(table_index.as_retriever(similarity_top_k=8).aretrieve("orders by month"))

    assert counts(conversation.token_counter) == (0, 1)
    assert_nothing_global(other_conversation)


def test_docs_tool_query_embedding_is_recorded(
    conversation, other_conversation, mock_models, in_memory_vector_store, monkeypatch
):
    class FakeHandler:
        def __init__(self, **kwargs):
            pass

        async def create_message(self, **kwargs):
            pass

        async def send_api_message(self):
            pass

    monkeypatch.setattr(docs_tool, "ChatMessageHandler", FakeHandler)
    tool = docs_tool.DocsTool(
        db=None,
        client_id=3,
        db_id=1,
        db_uuid=uuid.uuid4(),
        llm=FakeToolLLM(callback_manager=conversation.callback_manager),
        service_context=service_context(),
        prompt_metadata=prompt_metadata(conversation),
        chat_metadata=None,
    )
    response = asyncio.run(tool.get_docs(prompt="What is an active customer?"))

    assert "no documentation found" in response
    assert counts(conversation.token_counter) == (0, 1)
    assert_nothing_global(other_conversation)


def test_vis_tool_date_column_embeddings_are_recorded(conversation, other_conversation, mock_models):
    tool = VisTool(
        db=None,
        llm=FakeToolLLM(callback_manager=conversation.callback_manager),
        query_result=None,
        service_context=service_context(),
        user_uuid=uuid.uuid4(),
        parent_msg_uuid=uuid.uuid4(),
        result_store=None,
    )
    cols = ["order_date", "customer_name"]
    asyncio.run(tool.select_date_cols(cols=cols))

    events = conversation.token_counter.embedding_token_counts
    # 7 reference documents embedded in one batch, then one query embedding per column
    assert counts(conversation.token_counter) == (0, 7 + len(cols))
    assert len({event.event_id for event in events}) == 1 + len(cols)
    assert_nothing_global(other_conversation)


def test_chat_history_retrieval_embedding_is_recorded(conversation, other_conversation, mock_models, monkeypatch):
    class EmptyRedisVectorStore(TextVectorStore):
        def __init__(self, **kwargs):
            super().__init__()

        async def aquery(self, query, **kwargs):
            raise ValueError(NO_DOCS)

    monkeypatch.setattr(memory_agent, "RedisVectorStore", EmptyRedisVectorStore)
    memory = memory_agent.AgentMemory.__new__(memory_agent.AgentMemory)
    memory.service_context = service_context()
    memory.prompt_metadata = prompt_metadata(conversation)
    memory.chat_metadata = SimpleNamespace(index_name="idx", chat_uuid=uuid.uuid4())
    history = [ChatMessage(role=MessageRole.USER, content="Earlier question")]

    asyncio.run(memory._retrieve_chats_from_index(full_chat_history=history, team_info=None, system_prompt="s"))

    assert counts(conversation.token_counter) == (0, 1)
    assert_nothing_global(other_conversation)


def test_chat_history_indexing_embedding_is_recorded(conversation, other_conversation, mock_models, monkeypatch):
    class FakeDB:
        async def commit(self):
            pass

    async def get_chat_from_id(db, chat_id):
        return SimpleNamespace(chat_in_index=True)

    monkeypatch.setattr(crud_chat, "get_chat_from_id", get_chat_from_id)
    messages = [
        SimpleNamespace(
            content=content,
            sql_query=None,
            timestamp=datetime.now(),
            result_uuid=None,
            visual_json=None,
            role=role,
            msg_uuid=uuid.uuid4(),
        )
        for content, role in (("Orders?", MessageRole.USER), ("There were 12 orders.", MessageRole.ASSISTANT))
    ]
    asyncio.run(
        crud_chat.index_chat_history(
            db=FakeDB(),
            client_uuid=uuid.uuid4(),
            chat_id=1,
            chat_uuid=uuid.uuid4(),
            vector_id=1,
            chat_history=messages,
            callback_manager=conversation.callback_manager,
            vector_store=TextVectorStore(),
            embedding_model_info=model_info(),
        )
    )
    assert counts(conversation.token_counter) == (0, len(messages))
    assert_nothing_global(other_conversation)


def test_parallel_tools_in_one_conversation_are_all_recorded(conversation, mock_models, in_memory_vector_store):
    """Tools running concurrently in one conversation share its counter without losing events."""
    mock_models["tool_kwargs"] = {"true_false_bool": False}
    callback_manager = conversation.callback_manager

    async def run_all():
        tool = TableRetrieverTool.__new__(TableRetrieverTool)
        tool.service_context = service_context()
        tool.prompt_metadata = prompt_metadata(conversation)
        vis = VisTool(
            db=None,
            llm=FakeToolLLM(callback_manager=callback_manager),
            query_result=None,
            service_context=service_context(),
            user_uuid=uuid.uuid4(),
            parent_msg_uuid=uuid.uuid4(),
            result_store=None,
        )
        await asyncio.gather(*[tool.use_sub_questions(prompt="q") for _ in range(5)], vis.select_date_cols(["a"]))

    asyncio.run(run_all())
    assert counts(conversation.token_counter) == (5 * 2, 7 + 1)


# ---------------------------------------------------------------------------------------------------------------
# Saving the recorded usage
# ---------------------------------------------------------------------------------------------------------------


def test_save_token_counts_writes_embedding_rows_with_the_embedding_price(conversation, mock_models, monkeypatch):
    from basejump.core.models.ai import token_price

    monkeypatch.setattr(token_price.settings, "bedrock_embedding_price_per_1k", Decimal("0.0001"))
    monkeypatch.setattr(token_price.settings, "bedrock_llm_input_price_per_1k", Decimal("0.00015"))
    monkeypatch.setattr(token_price.settings, "bedrock_llm_output_price_per_1k", Decimal("0.0006"))

    class FakeDB:
        def __init__(self):
            self.added: list = []
            self.next_id = 0

        async def execute(self, statement):
            self.next_id += 1
            return SimpleNamespace(scalar=lambda: self.next_id)

        def add(self, obj):
            self.added.append(obj)

        async def commit(self):
            pass

    metadata = prompt_metadata(conversation)
    llm = FakeToolLLM(callback_manager=conversation.callback_manager)
    asyncio.run(llm.achat([ChatMessage(role=MessageRole.USER, content="hi")]))
    embed_model = AICatalog(callback_manager=conversation.callback_manager).get_embedding_model(model_info())
    embed_model.get_query_embedding("monthly revenue")
    # The fake LLM reports no Bedrock model ID; record it as the Bedrock LLM so the price is deterministic
    conversation.token_counter.llm_token_counts[0].model = enums.AIModelSchema.GPT_OSS_120B.value

    db = FakeDB()
    asyncio.run(crud_chat.save_token_counts(db=db, prompt_metadata=metadata))

    from basejump.core.models import models

    rows = [obj for obj in db.added if isinstance(obj, models.TokenCount)]
    assert len(rows) == 2
    llm_row, embedding_row = rows
    assert llm_row.ai_model_nm == enums.AIModelSchema.GPT_OSS_120B.value
    assert llm_row.total_embedding_token_count == 0
    assert (llm_row.prompt_llm_token_count, llm_row.completion_llm_token_count) == (PROMPT_TOKENS, COMPLETION_TOKENS)
    assert llm_row.cost_per_1k_tokens_output == Decimal("0.0006")
    assert embedding_row.ai_model_nm == enums.AIModelSchema.COHERE_EMBED_V4.value
    assert embedding_row.ai_model_provider == enums.AIModelProvider.AWS.value
    assert embedding_row.cost_per_1k_tokens_input == Decimal("0.0001")
    assert embedding_row.cost_per_1k_tokens_output == 0
    assert embedding_row.total_embedding_token_count > 0
    assert embedding_row.prompt_llm_token_count == embedding_row.total_embedding_token_count
    assert embedding_row.completion_llm_token_count == 0
    assert all(row.prompt_id == metadata.prompt_id and row.client_id == metadata.client_id for row in rows)
    # Each row is linked to the user, and the counter is emptied for the next prompt
    assert len([obj for obj in db.added if isinstance(obj, models.TokenUserAssociation)]) == 2
    assert counts(conversation.token_counter) == (0, 0)


def test_callback_managers_are_per_conversation():
    first = create_callback_mgrs(enums.AIModelSchema.GPT4o)
    second = create_callback_mgrs(enums.AIModelSchema.GPT4o)
    assert first.callback_manager is not second.callback_manager
    assert first.token_counter is not second.token_counter
    assert isinstance(first.callback_manager, CallbackManager)
