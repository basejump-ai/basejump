"""Agent memory is anything that will likely be stored in the Vector store that is injected into
the context for the agent to remember.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

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
from redisvl.query.filter import Tag
from redisvl.schema import IndexSchema
from sqlalchemy.ext.asyncio import AsyncSession

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database import auth
from basejump.core.database.crud import crud_chat, crud_result
from basejump.core.database.result import store
from basejump.core.database.vector_utils import init_semcache
from basejump.core.models import constants, enums, errors, models, prompts
from basejump.core.models import schemas as sch
from basejump.core.models.ai.catalog import AICatalog
from basejump.core.service.agents.results.refresh import ResultRefresher

logger = set_logging(handler_option="stream", name=__name__)


class AgentMemory:
    """Manages retrieval of chat history as well as the semantic cache."""

    def __init__(
        self,
        db: AsyncSession,
        service_context: sch.ServiceContext,
        prompt_metadata: sch.PromptMetadata,
        chat_metadata: sch.ChatMetadata,
        conn_params: sch.SQLDBSchema,
        query_result: Optional[sch.MessageQueryResult] = None,
        result_store: Optional[store.ResultStore] = None,
    ):
        self.db = db
        self.service_context = service_context
        self.prompt_metadata = prompt_metadata
        self.chat_metadata = chat_metadata
        self.conn_params = conn_params
        self.result_store = result_store or store.LocalResultStore(client_id=self.prompt_metadata.client_id)
        self.query_result = query_result

    async def get_cached_prompt(
        self, prompt: str, distance_threshold: float, db_uuids: set[str], num_results=1
    ) -> list[dict[str, Any]]:
        try:
            # TODO: Determine why the semantic cache has issues initializing sometimes
            semcache_init_timeout = 10
            async with asyncio.timeout(semcache_init_timeout):
                llmcache = await init_semcache(
                    client_id=self.prompt_metadata.client_id,
                    redis_client_async=self.service_context.redis_client_async,
                )
        except TimeoutError:
            logger.warning(f"Connection to the semcache timed out after {semcache_init_timeout} seconds")
            return []
        client_id_filter = Tag("client_id") == str(self.prompt_metadata.client_id)
        db_uuid_filter = Tag("db_uuid") == db_uuids
        complex_filter = db_uuid_filter & client_id_filter
        return await llmcache.acheck(
            prompt=prompt,
            filter_expression=complex_filter,
            distance_threshold=distance_threshold,
            num_results=num_results,
        )

    async def check_semcache(self, prompt, conn_uuids: set[str], db_uuids: set[str]) -> Optional[sch.MessagePair]:
        # See if a similar prompt has been cached
        semcache_response_raw = await self.get_cached_prompt(
            prompt=prompt, distance_threshold=constants.REDIS_SEMCACHE_EXACT_DISTANCE, db_uuids=db_uuids
        )
        if not semcache_response_raw:
            return None

        # Load cached response
        logger.info("Semantic similarity distance: %s", semcache_response_raw[0]["vector_distance"])
        metadata = semcache_response_raw[0]["metadata"]
        can_verify = auth.check_can_verify(
            required_role=enums.UserRoles(metadata["verified_user_role"]),
            user_role=enums.UserRoles(self.prompt_metadata.user_role),
        )
        semcache_response = sch.SemCacheResponse(
            response=semcache_response_raw[0]["response"],
            prompt=semcache_response_raw[0]["prompt"],
            vector_dist=semcache_response_raw[0]["vector_distance"],
            can_verify=can_verify,
            verified=True,
            **metadata,
        )
        self.chat_metadata.semcache_response = semcache_response  # save for later use in SQL query tool

        # Confirm permissions
        if metadata["conn_uuid"] not in conn_uuids:
            return None

        # Get the response
        return await self.get_cached_response(semcache_response=semcache_response)

    # TODO: Clean this up
    async def get_cached_response(self, semcache_response: sch.SemCacheResponse) -> Optional[sch.QueryResult]:
        # Get query result
        result = await crud_result.get_result(db=self.db, result_uuid=uuid.UUID(semcache_response.result_uuid))
        visual_result = await crud_result.get_visual_result_from_result(db=self.db, result_id=result.result_id)
        self.query_result = sch.MessageQueryResult.from_orm(result)
        if visual_result:
            self.query_result.visual_result_uuid = visual_result.visual_result_uuid
            self.query_result.visual_json = visual_result.visual_json
            self.query_result.visual_explanation = visual_result.visual_explanation

        # Return recent response
        cached_response_timestamp = datetime.strptime(semcache_response.timestamp, "%Y-%m-%d %H:%M:%S.%f%z")
        refresh_not_needed = datetime.now(cached_response_timestamp.tzinfo) - cached_response_timestamp < timedelta(
            days=1
        )
        if refresh_not_needed:
            # Create a message to return
            logger.info("Cached message found - returning cached message.")
            return sch.ChatResponse(
                response=semcache_response.response,
                query_result=self.query_result,
                metadata=sch.ResponseMetadata.parse_obj(semcache_response),
            )
        else:
            # Refresh the results
            refresher = ResultRefresher(
                db=self.db,
                prompt_metadata=self.prompt_metadata,
                service_context=self.service_context,
                conn_params=self.conn_params,
                result_store=self.result_store,
                query_result=self.query_result,
            )
            return await refresher.refresh_results(result=result, visual_result=visual_result)

    async def get_chat_history(
        self, chat: models.Chat, team_info: sch.TeamFields, system_prompt: Optional[str]
    ) -> list[ChatMessage]:
        try:
            full_chat_history = await crud_chat.get_chat_history_for_ai(db=self.db, chat_id=chat.chat_id)
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
