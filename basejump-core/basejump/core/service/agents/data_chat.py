"""Defines the AI models and routers to use for text to SQL"""

import asyncio
import uuid
from datetime import datetime, timedelta
from random import choice
from typing import Optional, Sequence

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools.types import AsyncBaseTool
from redisvl.query.filter import Tag

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database import db_auth
from basejump.core.database.crud import crud_chat, crud_connection, crud_result
from basejump.core.database.db_connect import ConnectDB
from basejump.core.database.result import store
from basejump.core.database.vector_utils import init_semcache
from basejump.core.models import constants, enums, errors, models
from basejump.core.models import schemas as sch
from basejump.core.models.prompts import NO_DB_ACCESS_PROMPT, sql_result_prompt_basic
from basejump.core.service.agents import agent_utils
from basejump.core.service.base import BaseChatAgent, ChatAgentSetup, ChatMessageHandler
from basejump.core.service.tools import sql, visualize

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
        service_context: sch.ServiceContext,
        chat_history: Optional[list[ChatMessage]] = None,
        max_iterations: int = constants.MAX_ITERATIONS,
        agent_llm: Optional[FunctionCallingLLM] = None,
        select_sample_values: bool = False,
        check_if_prompt_is_cached: bool = False,
        result_store: Optional[store.ResultStore] = None,
        conn_id: Optional[int] = None,
        verbose: bool = False,
    ):
        super().__init__(
            prompt_metadata=prompt_metadata,
            chat_metadata=chat_metadata,
            chat_history=chat_history,
            max_iterations=max_iterations,
            agent_llm=agent_llm,
            sql_engine=service_context.sql_engine,
            redis_client_async=service_context.redis_client_async,
            large_model_info=service_context.large_model_info,
            verbose=verbose,
        )
        self.service_context = service_context
        self.db_conn_params = db_conn_params
        self.select_sample_values = select_sample_values
        self.check_if_prompt_is_cached = check_if_prompt_is_cached
        self.result_store = result_store or store.LocalResultStore(client_id=self.prompt_metadata.client_id)
        self.conn_id = conn_id
        logger.debug("Chat history: %s", chat_history)

    @staticmethod
    def get_llm_type() -> enums.LLMType:
        return enums.LLMType.DATA_AGENT

    async def setup_tools(self) -> Sequence[AsyncBaseTool]:
        """Setup tools for the AI Agent to use"""
        tools = []
        # Loop over the available connections and setup the various tools
        if self.conn_id:
            db_connection = await crud_connection.get_db_conn_from_id(db=self.db, conn_id=self.conn_id)
            if not db_connection:
                msg = "The connection does not exist based on the provided connection ID."
                logger.error(msg)
                raise errors.NotFoundError(msg)
            connections: list[models.Connection] = [db_connection]
        else:
            connections = await ChatAgentSetup.get_connections(
                db=self.db,
                team_id=self.chat_metadata.team_id,
                user_id=self.prompt_metadata.user_id,
            )
        if not connections:
            raise errors.NotFoundError("No connections found")
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
                service_context=self.service_context,
                select_sample_values=self.select_sample_values,
                verbose=self.verbose,
                result_store=self.result_store,
            )
            await self.sql_tool.post_init()
            tools += self.sql_tool.tools
        vis_tool = visualize.VisTool(
            db=self.db,
            agent=self,
            llm=self.agent_llm,
            small_model_info=self.service_context.small_model_info,
            embedding_model_info=self.service_context.embedding_model_info,
            result_store=self.result_store,
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
                    await agent_utils.refresh_result(
                        db=self.db,
                        result=result,
                        commit=False,
                        client_id=self.prompt_metadata.client_id,
                        small_model_info=self.service_context.small_model_info,
                        db_conn_params=self.db_conn_params,
                        result_store=self.result_store,
                    )
                    result_manager = self.result_store.get_result_manager(result.result_file_path)
                    file_gen_func = result_manager.get_stream_result_generator()
                    stream_gen = file_gen_func()
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
                        visual_result = await agent_utils.refresh_visual_result(
                            db=self.db,
                            visual_result=visual_result,
                            client_user=client_user,
                            sql_engine=self.sql_engine,
                            small_model_info=self.service_context.small_model_info,
                            large_model_info=self.service_context.large_model_info,
                            embedding_model_info=self.service_context.embedding_model_info,
                            redis_client_async=self.redis_client_async,
                            result_store=self.result_store,
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
            verbose=self.verbose,
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
        if self.check_if_prompt_is_cached:
            if semcache_response := await self.check_semcache(prompt=prompt):
                return semcache_response
        return await self._chat_base(prompt=prompt)
