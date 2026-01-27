"""Defines the AI models and routers to use for text to SQL"""

from random import choice
from typing import Optional, Sequence

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools.types import AsyncBaseTool

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.connector import Connector
from basejump.core.database.crud import crud_connection
from basejump.core.database.result import store
from basejump.core.models import constants, enums, errors, models, prompts
from basejump.core.models import schemas as sch
from basejump.core.service.agents.base import BaseChatAgent
from basejump.core.service.agents.memory import AgentMemory
from basejump.core.service.agents.message import ChatMessageHandler
from basejump.core.service.agents.setup import ChatAgentSetup
from basejump.core.service.agents.tools import sql, visualize

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
        if self.verbose:
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
            prompt = prompts.NO_DB_ACCESS_PROMPT.format(prompt=prompt)
        if self.check_if_prompt_is_cached:
            conn_uuids = {str(connection.conn_uuid) for connection in self.connections}
            db_uuids = {str(connection.db_uuid) for connection in self.connections}
            agent_memory = AgentMemory(
                db=self.db,
                service_context=self.service_context,
                prompt_metadata=self.prompt_metadata,
                prompt_metadata=self.chat_metadata,
                conn_params=self.db_conn_params,
                result_store=self.result_store,
                query_result=self.query_result,
            )
            message_pair = await agent_memory.check_semcache(prompt=prompt, conn_uuids=conn_uuids, db_uuids=db_uuids)
            if not message_pair.response:
                prompt = message_pair.prompt.prompt
            elif message_pair.response:
                # Update the prompt ID for token cost calcs to just use previous cost
                prompt_hist = await crud_chat.get_prompt_history(
                    db=self.db, prompt_uuid=uuid.UUID(message_pair.response.metadata.prompt_uuid)
                )
                assert prompt_hist
                self.prompt_metadata.prompt_id = prompt_hist.prompt_id
                return await self._get_message(response=message_pair.response)
        return await self._chat_base(prompt=prompt)
