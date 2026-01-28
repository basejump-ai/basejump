"""Defines the AI models and routers to use for text to SQL"""

import uuid
from random import choice
from typing import Optional, Sequence

from llama_index.core.llms import MessageRole
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools.types import AsyncBaseTool

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.connector import Connector
from basejump.core.database.crud import crud_chat, crud_connection
from basejump.core.database.result import store
from basejump.core.models import constants, enums, errors, models, prompts
from basejump.core.models import schemas as sch
from basejump.core.service.agents.base import BaseChatAgent
from basejump.core.service.agents.memory.agent import AgentMemory
from basejump.core.service.agents.memory.semantic import SemanticMemory
from basejump.core.service.agents.message import ChatMessageHandler
from basejump.core.service.agents.setup import ChatAgentSetup
from basejump.core.service.agents.tools import sql, visualize

logger = set_logging(handler_option="stream", name=__name__)


class DataChatAgent(BaseChatAgent):
    """
    An AI Agent used for chatting with data in relational or unstructured formats
    """

    def __init__(
        self,
        db_conn_params: sch.SQLDBSchema,
        prompt_metadata: sch.PromptMetadata,
        chat_metadata: sch.ChatMetadata,
        service_context: sch.ServiceContext,
        memory: Optional[AgentMemory] = None,
        max_iterations: int = constants.MAX_ITERATIONS,
        agent_llm: Optional[FunctionCallingLLM] = None,
        select_sample_values: bool = False,
        use_semantic_cache: bool = False,
        result_store: Optional[store.ResultStore] = None,
        conn_id: Optional[int] = None,
        verbose: bool = False,
    ):
        if not memory:
            memory = AgentMemory(
                service_context=service_context,
                prompt_metadata=prompt_metadata,
                chat_metadata=chat_metadata,
                conn_params=db_conn_params,
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
        self.connections = []
        for conn in connections:
            assert isinstance(conn, models.DBConn)
            conn_db = await Connector.get_db_conn(db_conn=conn, db_params=conn.database_params)
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
            sql_tool_context = sch.SQLToolContext(
                client_conn_params=connection.conn_params,
                conn_id=connection.conn_id,
                conn_uuid=connection.conn_uuid,
                db_id=connection.db_id,
                db_uuid=connection.db_uuid,
                vector_id=connection.vector_id,
                prompt_metadata=self.prompt_metadata,
                service_context=self.service_context,
            )
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

    async def check_semantic_memory(self, prompt: str) -> Optional[sch.MessagePair]:
        conn_uuids = {str(connection.conn_uuid) for connection in self.connections}
        db_uuids = {str(connection.db_uuid) for connection in self.connections}
        semantic_memory = SemanticMemory(
            service_context=self.service_context,
            prompt_metadata=self.prompt_metadata,
            chat_metadata=self.chat_metadata,
            conn_params=self.db_conn_params,
            result_store=self.result_store,
            query_result=self.query_result,
        )
        if message_pair := await semantic_memory.check_cache(
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
