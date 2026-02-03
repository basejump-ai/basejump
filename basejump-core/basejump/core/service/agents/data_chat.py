"""Defines the AI models and routers to use for text to SQL"""

import uuid
from datetime import datetime, timedelta
from random import choice
from typing import Optional, Sequence

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms import MessageRole
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools.types import AsyncBaseTool
from sqlalchemy.ext.asyncio import AsyncSession

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database import auth
from basejump.core.database.crud import crud_chat, crud_connection, crud_result
from basejump.core.database.result import store
from basejump.core.models import constants, enums, errors, models, prompts
from basejump.core.models import schemas as sch
from basejump.core.service.agents.base import ChatAgent
from basejump.core.service.agents.memory.agent import AgentMemory
from basejump.core.service.agents.memory.semantic import SemanticMemory
from basejump.core.service.agents.message import ChatMessageHandler
from basejump.core.service.agents.setup import ChatAgentSetup
from basejump.core.service.agents.tools import sql, tool_utils, visualize
from basejump.core.service.agents.tools.sql.base import SQLTool
from basejump.core.service.agents.tools.visualize import VisTool

logger = set_logging(handler_option="stream", name=__name__)


class DataChatAgent(ChatAgent):
    """
    An AI Agent used for chatting with data in relational or unstructured formats
    """

    def __init__(
        self,
        db_conn_params: sch.SQLDBSchema,
        prompt_metadata: sch.PromptMetadata,
        chat_metadata: sch.ChatMetadata,
        service_context: sch.ServiceContext,
        chat_history: Optional[list[ChatMessage]] = None,
        memory: Optional[AgentMemory] = None,
        max_iterations: int = constants.MAX_ITERATIONS,
        agent_llm: Optional[FunctionCallingLLM] = None,
        select_sample_values: bool = False,
        use_semantic_cache: bool = False,
        result_store: Optional[store.ResultStore] = None,
        conn_id: Optional[int] = None,
        verbose: bool = False,
    ):
        if memory and chat_history:
            logger.warning(
                """Both 'memory' and 'chat_history' were provided. The 'chat_history' parameter \
will be ignored; using memory's chat history instead."""
            )
        if not memory:
            memory = AgentMemory(
                service_context=service_context,
                prompt_metadata=prompt_metadata,
                chat_metadata=chat_metadata,
                conn_params=db_conn_params,
                chat_history=chat_history,
            )
        super().__init__(
            prompt_metadata=prompt_metadata,
            chat_metadata=chat_metadata,
            memory=memory,
            max_iterations=max_iterations,
            agent_llm=agent_llm,
            service_context=service_context,
            verbose=verbose,
            result_store=result_store,
        )
        self.service_context = service_context
        self.db_conn_params = db_conn_params
        self.select_sample_values = select_sample_values
        self.use_semantic_cache = use_semantic_cache
        self.conn_id = conn_id
        if self.verbose:
            logger.debug("Chat history: %s", self.memory.chat_history)

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
        self.sql_tool_contexts = []
        for conn in connections:
            assert isinstance(conn, models.DBConn)
            sql_tool_context = await tool_utils.get_sql_tool_context(service_context=self.service_context, conn=conn)
            self.connections.append(sql_tool_context)
        await self.db.commit()  # NOTE: Closing transaction to avoid idle in transaction
        for sql_tool_context in self.sql_tool_contexts:
            self.sql_tool = sql.SQLTool(
                agent=self,
                db=self.db,
                db_conn_params=self.db_conn_params,
                sql_tool_context=sql_tool_context,
                select_sample_values=self.select_sample_values,
                result_store=self.result_store,
            )
            tools += await self.sql_tool.get_tools()
        self.vis_tool = visualize.VisTool(
            agent=self,
            llm=self.agent_llm,
        )
        tools += await self.vis_tool.get_tools()
        return tools

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
        if self.memory.chat_history:
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
            prompt = prompts.NO_DB_ACCESS_PROMPT.format(prompt=prompt)
        elif self.use_semantic_cache:
            if message_pair := await self.check_semantic_memory(prompt=prompt):
                if message_pair.response:
                    return await self._get_message(response=message_pair.response.response)
                prompt = message_pair.prompt.prompt

        return await self._chat_base(prompt=prompt)

    async def check_semantic_memory(self, prompt: str) -> Optional[sch.MessagePair]:
        conn_uuids = {str(connection.conn_uuid) for connection in self.connections}
        db_uuids = {str(connection.db_uuid) for connection in self.connections}
        if message_pair := await self.check_cache(
            db=self.db,
            sql_tool=self.sql_tool,
            vis_tool=self.vis_tool,
            prompt=prompt,
            conn_uuids=conn_uuids,
            db_uuids=db_uuids,
        ):
            if message_pair.response:
                # Update the prompt ID for token cost calcs to just use previous cost
                assert message_pair.response.metadata, "Missing ChatResponse metadata"
                prompt_hist = await crud_chat.get_prompt_history(
                    db=self.db, prompt_uuid=uuid.UUID(message_pair.response.metadata.prompt_uuid)
                )
                assert prompt_hist
                self.prompt_metadata.prompt_id = prompt_hist.prompt_id
        return message_pair

    async def _get_cached_response(self, semcache_response: sch.SemCacheResponse) -> sch.MessagePair:
        # Get query result
        result = await crud_result.get_result(db=self.db, result_uuid=uuid.UUID(semcache_response.result_uuid))
        visual_result = await crud_result.get_visual_result_from_result(db=self.db, result_id=result.result_id)
        self.sql_tool.agent.query_result = sch.MessageQueryResult.from_orm(result)
        if visual_result:
            self.sql_tool.agent.query_result.visual_result_uuid = visual_result.visual_result_uuid
            self.sql_tool.agent.query_result.visual_json = visual_result.visual_json
            self.sql_tool.agent.query_result.visual_explanation = visual_result.visual_explanation

        # Return recent response
        cached_response_timestamp = datetime.strptime(semcache_response.timestamp, "%Y-%m-%d %H:%M:%S.%f%z")
        refresh_not_needed = datetime.now(cached_response_timestamp.tzinfo) - cached_response_timestamp < timedelta(
            days=1
        )
        if refresh_not_needed:
            # Create a message to return
            logger.info("Cached message found - returning cached message.")
            return sch.MessagePair(
                prompt=sch.ChatPrompt(prompt=semcache_response.prompt),
                response=sch.ChatResponse(
                    response=semcache_response.response,
                    query_result=self.sql_tool.agent.query_result,
                    metadata=sch.ResponseMetadata.parse_obj(semcache_response),
                ),
            )
        else:
            # Refresh the results
            query_result = await tool_utils.refresh_results(
                sql_tool=self.sql_tool, vis_tool=self.vis_tool, result=result, visual_result=visual_result
            )
            # Get the response using SQL query results
            new_prompt_base = prompts.sql_result_prompt_basic(query_result=query_result)
            prompt = (
                f"""The user asked this question: {self.sql_tool.agent.prompt_metadata.initial_prompt}. \
    This SQL query has been ran for you: {self.sql_tool.agent.query_result.sql_query}. """
                + new_prompt_base
            )
            return sch.MessagePair(prompt=sch.ChatPrompt(prompt=prompt))

    async def check_cache(
        self, sql_tool: SQLTool, vis_tool: VisTool, db: AsyncSession, prompt, conn_uuids: set[str], db_uuids: set[str]
    ) -> Optional[sch.MessagePair]:
        # See if a similar prompt has been cached
        semantic_memory = SemanticMemory(
            redis_client_async=self.service_context.redis_client_async,
        )
        semcache_response = await semantic_memory.get_cached_prompt(
            prompt=prompt,
            client_id=sql_tool.agent.prompt_metadata.client_id,
            distance_threshold=constants.REDIS_SEMCACHE_EXACT_DISTANCE,
            db_uuids=db_uuids,
        )
        if not semcache_response:
            return None

        # Load cached response
        logger.info("Semantic similarity distance: %s", semcache_response.vector_dist)
        can_verify = auth.check_can_verify(
            required_role=enums.UserRoles(semcache_response.verified_user_role),
            user_role=enums.UserRoles(sql_tool.agent.prompt_metadata.user_role),
        )
        semcache_response.can_verify = can_verify
        sql_tool.agent.chat_metadata.semcache_response = semcache_response  # save for later use in SQL query tool

        # Confirm permissions
        if semcache_response.conn_uuid not in conn_uuids:
            return None

        # Get the response
        return await self._get_cached_response(semcache_response=semcache_response)
