"""Agent memory is anything that will likely be stored in the Vector store that is injected into
the context for the agent to remember.
"""

from datetime import datetime, timedelta
from typing import Optional

import redis
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms import MessageRole
from llama_index.core.memory import VectorMemory
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.vector_stores.redis.base import NO_DOCS
from redisvl.schema import IndexSchema
from sqlalchemy.ext.asyncio import AsyncSession

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.crud import crud_chat
from basejump.core.database.result import store
from basejump.core.models import constants, enums, errors, models, prompts
from basejump.core.models import schemas as sch
from basejump.core.models.ai.catalog import AICatalog
from basejump.core.service.agents.base import BaseAgentMemory
from basejump.core.service.agents.memory.semantic import SemanticMemory

logger = set_logging(handler_option="stream", name=__name__)


class SimpleAgentMemory(BaseAgentMemory):
    def __init__(
        self,
        service_context: sch.ServiceContext,
        prompt_metadata: sch.PromptMetadata,
        chat_history: Optional[list[ChatMessage]] = None,
        query_result: Optional[sch.MessageQueryResult] = None,
        result_store: Optional[store.ResultStore] = None,
    ):
        super().__init__(
            service_context=service_context,
            prompt_metadata=prompt_metadata,
            chat_history=chat_history,
            query_result=query_result,
            result_store=result_store,
        )


class AgentMemory(BaseAgentMemory):
    """Manages retrieval of chat history as well as the semantic cache."""

    def __init__(
        self,
        service_context: sch.ServiceContext,
        prompt_metadata: sch.PromptMetadata,
        chat_metadata: sch.ChatMetadata,
        conn_params: sch.SQLDBSchema,
        chat_history: Optional[list[ChatMessage]] = None,
        query_result: Optional[sch.MessageQueryResult] = None,
        result_store: Optional[store.ResultStore] = None,
        semantic_memory: Optional[SemanticMemory] = None,
    ):
        super().__init__(
            service_context=service_context,
            prompt_metadata=prompt_metadata,
            chat_history=chat_history,
            query_result=query_result,
            result_store=result_store,
        )
        self.chat_metadata = chat_metadata
        self.conn_params = conn_params
        self.semantic_memory = semantic_memory or self.load_semantic_memory()

    def load_semantic_memory(self) -> SemanticMemory:
        return SemanticMemory(
            redis_client_async=self.service_context.redis_client_async,
        )

    async def get_chat_history(
        self, db: AsyncSession, chat: models.Chat, team_info: sch.TeamFields, system_prompt: Optional[str] = None
    ) -> list[ChatMessage]:
        try:
            full_chat_history = await crud_chat.get_chat_history_for_ai(db=db, chat_id=chat.chat_id)
        except Exception as e:
            logger.error("Error retrieving get chat history %s", str(e))
            raise errors.GetChatHistoryError
        return await self._retrieve_chats_from_index(
            full_chat_history=full_chat_history, team_info=team_info, system_prompt=system_prompt
        )

    async def _retrieve_chats_from_index(
        self, full_chat_history: list[ChatMessage], team_info: sch.TeamFields, system_prompt: Optional[str]
    ) -> list[ChatMessage]:
        schema = IndexSchema.from_dict(
            {
                "index": {"name": self.chat_metadata.index_name, "prefix": self.chat_metadata.index_name + "/vector"},
                "fields": [
                    # Required fields
                    {"name": "id", "type": "tag"},
                    {"name": "doc_id", "type": "tag"},
                    {"name": "text", "type": "text"},
                    {"name": "vector", "type": "vector", "attrs": {"dims": 1536, "algorithm": "flat"}},
                    *constants.VECTOR_FILTERS,
                ],
            }
        )
        vector_store = RedisVectorStore(
            redis_client_async=self.service_context.redis_client_async, schema=schema, legacy_filters=True
        )
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="chat_uuid", value=str(self.chat_metadata.chat_uuid), operator=FilterOperator.EQ),
                MetadataFilter(
                    key="client_uuid", value=str(self.prompt_metadata.client_uuid), operator=FilterOperator.EQ
                ),
                MetadataFilter(key="vector_type", value=enums.VectorSourceType.CHAT.value, operator=FilterOperator.EQ),
            ]
        )
        TOP_K = 2
        ai_catalog = AICatalog(callback_manager=self.prompt_metadata.callback_manager)
        embed_model = ai_catalog.get_embedding_model(model_info=self.service_context.embedding_model_info)
        vector_memory = VectorMemory.from_defaults(
            vector_store=vector_store,
            embed_model=embed_model,
            retriever_kwargs={"similarity_top_k": TOP_K, "filters": filters},
        )
        system_prompt = system_prompt or prompts.get_system_prompt(team_info=team_info)
        if not full_chat_history:
            return [ChatMessage(role=MessageRole.SYSTEM, content=system_prompt, timestamp=datetime.now())]
        try:
            msgs = await vector_memory.aget(input=self.prompt_metadata.initial_prompt)
            logger.info("Retrieved the following messages for chat history: %s", msgs)
            if not all([message.timestamp for message in msgs]):
                return [ChatMessage(role=MessageRole.SYSTEM, content=system_prompt, timestamp=datetime.now())]
            min_timestamp = min([message.timestamp for message in msgs]) - timedelta(seconds=1)  # type: ignore
            # Always add the system prompt as the earliest timestamped message
            chat_history = [ChatMessage(role=MessageRole.SYSTEM, content=system_prompt, timestamp=min_timestamp)]
            # Add the retrieved messages to chat history
            chat_history += msgs
            # Check if retrieved messages include the latest message. If not, then add it
            latest_msg = full_chat_history[-1]
            msg_included = any([True if latest_msg.content == msg.content else False for msg in msgs])
            if not msg_included:
                # HACK: Sometimes throws IndexError
                warning_message = (
                    "A user and assistant pair is missing from the latest message. "
                    "Omitting latest message to prevent the AI from being confused since there "
                    "always needs to be user and assistant role pair."
                )
                try:
                    if (
                        full_chat_history[-1].role == MessageRole.ASSISTANT
                        and full_chat_history[-2].role == MessageRole.USER
                    ):
                        chat_history += full_chat_history[-2:]
                    else:
                        logger.warning(warning_message)
                except IndexError:
                    logger.warning(warning_message)
            # Sort all messages in order
            chat_history.sort(key=lambda msg: msg.timestamp)  # type: ignore
        except (redis.exceptions.ResponseError, ValueError) as e:
            if isinstance(e, ValueError) and NO_DOCS not in str(e):
                raise e
            logger.warning("Redis error when retrieving chat history: %s", str(e))
            # This is ok to pass since the index is overwritten every time and will
            # be re-created at the end of the chat
            return []
        return chat_history
