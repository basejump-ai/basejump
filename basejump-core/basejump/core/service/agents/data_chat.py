"""Defines the AI models and routers to use for text to SQL"""

import asyncio
import uuid
from datetime import datetime, timedelta
from random import choice
from typing import Optional, Sequence

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database import db_auth
from basejump.core.database.crud import crud_chat, crud_result
from basejump.core.database.db_connect import ConnectDB
from basejump.core.database.vector_utils import init_semcache
from basejump.core.models import constants, enums, models
from basejump.core.models import schemas as sch
from basejump.core.models.prompts import NO_DB_ACCESS_PROMPT, sql_result_prompt_basic
from basejump.core.service import service_utils
from basejump.core.service.base import BaseChatAgent, ChatAgentSetup, ChatMessageHandler
from basejump.core.service.tools import sql, visualize
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools.types import AsyncBaseTool
from redis.asyncio import Redis as RedisAsync
from redisvl.query.filter import Tag
from sqlalchemy.ext.asyncio import AsyncEngine

logger = set_logging(handler_option="stream", name=__name__)


class DataChatAgent(BaseChatAgent):
    """
    An AI Agent used for chatting with data in relational or unstructured formats

    NOTES
    -----
    This agent currently only has the ability to chat with databases. However, additional
    functionality will be added in the future
    """

    def __init__(
        self,
        db_conn_params: sch.SQLDBSchema,
        prompt_metadata: sch.PromptMetadata,
        chat_metadata: sch.ChatMetadata,
        redis_client_async: RedisAsync,
        large_model_info: sch.ModelInfo,
        small_model_info: sch.ModelInfo,
        embedding_model_info: sch.AzureModelInfo,
        sql_engine: AsyncEngine,
        chat_history: Optional[list[ChatMessage]] = None,
        max_iterations: int = constants.MAX_ITERATIONS,
        agent_llm: Optional[FunctionCallingLLM] = None,
    ):
        self.redis_client_async = redis_client_async
        self.large_model_info = large_model_info
        self.small_model_info = small_model_info
        self.embedding_model_info = embedding_model_info
        self.sql_engine = sql_engine
        self.db_conn_params = db_conn_params
        logger.debug("Here is the chat history %s", chat_history)
        super().__init__(
            prompt_metadata=prompt_metadata,
            chat_metadata=chat_metadata,
            chat_history=chat_history,
            max_iterations=max_iterations,
            agent_llm=agent_llm,
            sql_engine=sql_engine,
            redis_client_async=redis_client_async,
            large_model_info=large_model_info,
        )

    @staticmethod
    def get_llm_type() -> enums.LLMType:
        return enums.LLMType.DATA_AGENT

    async def setup_tools(self) -> Sequence[AsyncBaseTool]:
        """Setup tools for the AI Agent to use"""
        tools = []
        # Loop over the available connections and setup the various tools
        connections = await ChatAgentSetup.get_connections(
            db=self.db, team_id=self.chat_metadata.team_id, user_id=self.prompt_metadata.user_id
        )
        self.connections = []
        for conn in connections:
            assert isinstance(conn, models.DBConn)
            conn_db = await ConnectDB.get_db_conn(db_conn=conn, db_params=conn.database_params)
            conn_schema = sch.SQLConnSchema(
                conn_params=conn_db.conn_params,
                conn_id=conn.conn_id,
                conn_uuid=str(conn.conn_uuid),
                db_id=conn.db_id,
                vector_id=conn.database_params.vector_id,
                db_uuid=str(conn.database_params.db_uuid),
            )
            self.connections.append(conn_schema)
        await self.db.commit()  # NOTE: Closing transaction to avoid idle in transaction
        for connection in self.connections:
            self.sql_tool = sql.SQLTool(
                agent=self,
                db=self.db,
                db_conn_params=self.db_conn_params,
                client_conn_params=connection.conn_params,
                conn_id=connection.conn_id,
                conn_uuid=connection.conn_uuid,
                db_id=connection.db_id,
                db_uuid=connection.db_uuid,
                vector_id=connection.vector_id,
                prompt_metadata=self.prompt_metadata,
                redis_client_async=self.redis_client_async,
                large_model_info=self.large_model_info,
                small_model_info=self.small_model_info,
                embedding_model_info=self.embedding_model_info,
                sql_engine=self.sql_engine,
            )
            await self.sql_tool.post_init()
            tools += self.sql_tool.tools
        vis_tool = visualize.VisTool(
            db=self.db,
            agent=self,
            llm=self.agent_llm,
            small_model_info=self.small_model_info,
            embedding_model_info=self.embedding_model_info,
        )
        tools.append(vis_tool.get_plot_tool())
        return tools

    async def check_semcache(self, prompt) -> Optional[sch.Message]:
        try:
            # TODO: Determine why the semantic cache has issues initializing sometimes
            semcache_init_timeout = 10
            async with asyncio.timeout(semcache_init_timeout):
                llmcache = await init_semcache(
                    client_id=self.prompt_metadata.client_id, redis_client_async=self.redis_client_async
                )
        except TimeoutError:
            logger.warning(f"Connection to the semcache timed out after {semcache_init_timeout} seconds")
            return None
        client_id_filter = Tag("client_id") == str(self.prompt_metadata.client_id)
        db_uuid_filter = Tag("db_uuid") == {str(connection.db_uuid) for connection in self.connections}
        complex_filter = db_uuid_filter & client_id_filter
        semcache_response = await llmcache.acheck(prompt=prompt, filter_expression=complex_filter)
        if semcache_response:
            # Get variables for the first result
            logger.info("Semantic similarity distance: %s", semcache_response[0]["vector_distance"])
            metadata = semcache_response[0]["metadata"]
            can_verify = db_auth.check_can_verify(
                required_role=enums.UserRoles(metadata["verified_user_role"]),
                user_role=enums.UserRoles(self.prompt_metadata.user_role),
            )
            semcache_response_obj = sch.SemCacheResponse(
                response=semcache_response[0]["response"],
                prompt=semcache_response[0]["prompt"],
                vector_dist=semcache_response[0]["vector_distance"],
                can_verify=can_verify,
                verified=True,
                **metadata,
            )
            self.chat_metadata.semcache_response = semcache_response_obj  # save for later use in SQL query tool
            # Convert timestamp to datetime obj
            # Check if the question is the same and within 1 day of the original result
            conn_uuids = {str(connection.conn_uuid) for connection in self.connections}
            if (
                semcache_response_obj.vector_dist <= constants.REDIS_SEMCACHE_EXACT_DISTANCE
                and metadata["conn_uuid"] in conn_uuids
            ):
                timestamp_obj = datetime.strptime(semcache_response_obj.timestamp, "%Y-%m-%d %H:%M:%S.%f%z")
                # Get the results
                result = await crud_result.get_result(
                    db=self.db, result_uuid=uuid.UUID(semcache_response_obj.result_uuid)
                )
                visual_result = await crud_result.get_visual_result_from_result(db=self.db, result_id=result.result_id)
                self.query_result = sch.MessageQueryResult.from_orm(result)
                if visual_result:
                    self.query_result.visual_result_uuid = visual_result.visual_result_uuid
                    self.query_result.visual_json = visual_result.visual_json
                    self.query_result.visual_explanation = visual_result.visual_explanation
                if timestamp_obj > (timestamp_obj - timedelta(days=1)):
                    # Create a Message to return
                    logger.info("Cached message found - returning cached message.")
                    # Update the prompt ID for token cost calcs to just use previous cost
                    prompt_hist = await crud_chat.get_prompt_history(
                        db=self.db, prompt_uuid=uuid.UUID(semcache_response_obj.prompt_uuid)
                    )
                    assert prompt_hist
                    self.prompt_metadata.prompt_id = prompt_hist.prompt_id
                    return await self._get_message(response=semcache_response_obj.response)
                else:
                    # Refresh the results
                    await service_utils.refresh_result(
                        db=self.db,
                        result=result,
                        commit=False,
                        client_id=self.prompt_metadata.client_id,
                        small_model_info=self.small_model_info,
                        db_conn_params=self.db_conn_params,
                    )
                    file_gen_func = service_utils.get_file_generator_func(result.result_file_path)
                    stream_gen = file_gen_func(result.result_file_path)
                    rows_base = next(stream_gen)
                    rows = [tuple(row.split(",")) for row in rows_base.decode("utf-8").splitlines()]
                    query_res = sch.QueryResult(
                        query_result=rows[: constants.AI_RESULT_PREVIEW_CT],
                        preview_row_ct=constants.AI_RESULT_PREVIEW_CT,
                        num_rows=result.row_num_total,
                        num_cols=1,  # just a placeholder since it isn't used in the prompt
                        result_type=result.result_type,
                        sql_query=result.sql_query,
                        result_uuid=str(result.result_uuid),
                        # TODO: Clean up schema objs with preview row ct (they are redundant)
                        ai_preview_row_ct=constants.AI_RESULT_PREVIEW_CT,
                        result_file_path=result.result_file_path,
                        preview_file_path=result.preview_file_path,
                    )
                    self.query_result = sch.MessageQueryResult.from_orm(result)
                    if visual_result:
                        client_user = sch.ClientUserInfo.parse_obj(self.prompt_metadata)
                        visual_result = await service_utils.refresh_visual_result(
                            db=self.db,
                            visual_result=visual_result,
                            client_user=client_user,
                            sql_engine=self.sql_engine,
                            small_model_info=self.small_model_info,
                            large_model_info=self.large_model_info,
                            embedding_model_info=self.embedding_model_info,
                            redis_client_async=self.redis_client_async,
                        )
                        self.query_result.visual_result_uuid = visual_result.visual_result_uuid
                        self.query_result.visual_json = visual_result.visual_json
                        self.query_result.visual_explanation = visual_result.visual_explanation
                    # Get the response using SQL query results
                    # Get the S3 object rows
                    new_prompt_base = sql_result_prompt_basic(query_result=query_res)
                    new_prompt = (
                        f"""The user asked this question: {self.prompt_metadata.initial_prompt}. \
        This SQL query has been ran for you: {self.query_result.sql_query}. """
                        + new_prompt_base
                    )
                    return await self._chat_base(prompt=new_prompt)
        return None

    async def _chat(self, prompt: str) -> sch.Message:
        """Prompt the AI"""
        intros = [
            "Thanks for your request, I'm on it!",
            "Let me dig up an answer. Searching company knowledge...",
            "Hmmm - let me think about this...",
            "I'm on it! Just a moment...",
            "Searching...",
            "You've come to the right place. Let me get an answer for you...",
        ]
        handler = ChatMessageHandler(
            prompt_metadata=self.prompt_metadata,
            chat_metadata=self.chat_metadata,
            redis_client_async=self.redis_client_async,
        )
        if self.chat_history:
            await handler.create_message(
                db=self.db, role=MessageRole.ASSISTANT, content=choice(intros), msg_type=enums.MessageType.THOUGHT
            )
            await handler.send_api_message()
        # Save the prompt right away in case the user asks another question before the AI answers the first question
        await handler.create_message(
            db=self.db,
            role=MessageRole.USER,
            content=prompt,
            msg_uuid=self.chat_metadata.parent_msg_uuid,
            initial_prompt=True,
        )
        await handler.save_message(message=handler.message)
        # Prompt the AI
        # Modify the prompt if needed
        if not self.connections:
            prompt = NO_DB_ACCESS_PROMPT.format(prompt=prompt)
        if semcache_response := await self.check_semcache(prompt=prompt):
            return semcache_response
        return await self._chat_base(prompt=prompt)
