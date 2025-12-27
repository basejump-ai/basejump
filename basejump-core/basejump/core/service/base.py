"""Contains parent classes for the service directory"""

import asyncio
import json
import re
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Union, Sequence
from zoneinfo import ZoneInfo

import aiohttp
import redis
from basejump.core.common.config.logconfig import set_logging
from basejump.core.database import db_utils
from basejump.core.database.aicatalog import AICatalog
from basejump.core.database.crud import crud_chat, crud_connection
from basejump.core.database.crud.crud_utils import create_callback_mgrs
from basejump.core.database.db_connect import LocalSession
from basejump.core.models import constants, enums, errors, models
from basejump.core.models import schemas as sch
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.agent.react.output_parser import (
    COULD_NOT_PARSE_TXT,
    EXPECTED_OUTPUT_INSTRUCTIONS,
)
from llama_index.core.agent.types import Task, TaskStep
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import VectorMemory
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.tools.types import AsyncBaseTool
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.vector_stores.redis.base import NO_DOCS
from redis.asyncio import Redis as RedisAsync
from redisvl.schema import IndexSchema
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

logger = set_logging(handler_option="stream", name=__name__)


class MessageHandler:
    def __init__(
        self,
        prompt_metadata: Union[sch.PromptMetadataBase, sch.PromptMetadata],
        query_result: Optional[sch.MessageQueryResult] = None,
    ):
        self.prompt_metadata = prompt_metadata
        self.query_result = query_result or sch.MessageQueryResult()

    def create_message(
        self,
        role: MessageRole,
        content: str = "",
        msg_type: enums.MessageType = enums.MessageType.RESPONSE,
        msg_uuid: Optional[uuid.UUID] = None,
    ):
        if not msg_uuid:
            msg_uuid = uuid.uuid4()
        self.message = sch.Message(
            msg_uuid=msg_uuid,
            role=role,
            content=db_utils.remove_message_context(content=content),
            msg_type=msg_type,
            query_result=self.query_result,
            timestamp=datetime.now(ZoneInfo("UTC")).isoformat(),
        )


class ChatMessageHandler(MessageHandler):
    def __init__(
        self,
        prompt_metadata: Union[sch.PromptMetadataBase, sch.PromptMetadata],
        chat_metadata: sch.ChatMetadata,
        redis_client_async: RedisAsync,
        query_result: Optional[sch.MessageQueryResult] = None,
    ):
        super().__init__(prompt_metadata=prompt_metadata, query_result=query_result)
        self.chat_metadata = chat_metadata
        self.redis_client_async = redis_client_async

    @property
    def api_message(self):
        return self.create_api_message()

    def _log_thought_message(self, content: str):
        thought = sch.ThoughtMessage(timestamp=datetime.now(ZoneInfo("UTC")), thought=content)
        self.chat_metadata.curr_thought_history.append(thought)

    def create_thought_message(self, content: str) -> None:
        super().create_message(role=sch.MessageRole.ASSISTANT, content=content, msg_type=enums.MessageType.THOUGHT)
        self._log_thought_message(content=content)

    async def create_message(  # type: ignore
        self,
        db: AsyncSession,
        role: MessageRole,
        content: str = "",
        msg_type: enums.MessageType = enums.MessageType.RESPONSE,
        msg_uuid: Optional[uuid.UUID] = None,
        initial_prompt: bool = False,
    ):
        super().create_message(role=role, content=content, msg_type=msg_type, msg_uuid=msg_uuid)
        if msg_type == enums.MessageType.THOUGHT:
            self._log_thought_message(content=content)
        if initial_prompt:
            # Save the user prompt
            self.chat_metadata.curr_chat_history.append(self.api_message)
            await crud_chat.save_message(
                db=db,
                message=self.api_message,
                prompt_metadata=self.prompt_metadata,
                chat_metadata=self.chat_metadata,
                query_result=self.query_result,
            )
            # Save the assistant response placeholder
            super().create_message(role=MessageRole.ASSISTANT, msg_type=enums.MessageType.INIT)
            self.chat_metadata.curr_chat_history.append(self.api_message)
            await crud_chat.save_message(
                db=db,
                message=self.api_message,
                prompt_metadata=self.prompt_metadata,
                chat_metadata=self.chat_metadata,
                query_result=self.query_result,
            )
        if self.chat_metadata.reset_parent_msg_uuid:
            # Sending an extra message if a new parent msg UUID needs to be reset
            # TODO: Create message here with blanks

            logger.debug("Webhook message: %s", "Sending solution status to indicate AI has finalized reply")
            await self._send_solution_message(db=db)
            self.chat_metadata.parent_msg_uuid = self.message.msg_uuid
            self.chat_metadata.reset_parent_msg_uuid = False

    def format_message(self) -> str:
        self.api_message.timestamp = self.api_message.timestamp.isoformat()
        self.api_message.prompt_time = self.api_message.prompt_time.isoformat()
        logger.debug("Here is the timestamp: %s", str(self.api_message.timestamp))
        return self.api_message.model_dump_json()

    async def save_message(self, message: sch.Message) -> None:
        # Add running chat history for the VectorMemory
        found_match = False
        for hist_message in self.chat_metadata.curr_chat_history:
            if hist_message.role == message.role and str(hist_message.parent_msg_uuid) == str(
                self.chat_metadata.parent_msg_uuid
            ):
                logger.debug("Found chat hist match for role: %s", hist_message.role)
                found_match = True
                # Update the message
                hist_message.content = message.content
                hist_message.msg_type = message.msg_type
                if message.query_result:
                    query_res_dict = self.process_query_result(query_result=message.query_result)
                    for key, value in query_res_dict.items():
                        setattr(hist_message, key, value)
        if not found_match:
            self.chat_metadata.curr_chat_history.append(self.api_message)
        # TODO: Use websockets so the DB doesn't have to be saved to until the user
        # disconnects

    async def send_api_message(self, send_solution: Optional[sch.SendSolution] = None):
        """Send messages to the API"""

        if not self.message:
            raise ValueError("Create a message first using create_message")
        if not self.chat_metadata:
            # Need a webhook url for anything to be sent
            return
        logger.debug("Webhook message: %s", self.message.content)
        api_message = self.format_message()
        await self._send_api_message(api_message=api_message)
        # Make sure to send a solution message after the error message
        if send_solution or self.message.msg_type == enums.MessageType.ERROR:
            # NOTE: This is so the initial message from the endpoint has time to resolve before an error
            # is sent. This avoids the messages getting out of order.
            if self.message.msg_type == enums.MessageType.ERROR:
                await asyncio.sleep(1)
            if send_solution:
                await self._send_solution_message(db=send_solution.db)

    async def _send_api_message(self, api_message: str):
        logger.debug("Webhook API message: %s", api_message)
        try:
            assert self.chat_metadata.webhook_url
            webhook_url = self.chat_metadata.webhook_url
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    headers={**self.chat_metadata.webhook_headers, "Content-Type": "application/json"},  # type: ignore
                    url=webhook_url,
                    data=api_message,
                ) as response:
                    if response.status != 200:
                        logger.warning("Webhook status response: %s", response.status)
                        logger.warning("Webhook status text: %s", response.text)
        except AssertionError:
            logger.debug("No webhook URL found")
            logger.debug("Webhook header values %s", str(self.chat_metadata.webhook_headers))
            # If no webhook URL, then skip sending the API message
            pass

    def process_query_result(self, query_result) -> dict:
        if self.prompt_metadata.return_visual_json and query_result:
            if isinstance(query_result.visual_json, str):
                visual_json = json.loads(query_result.visual_json)
            else:
                visual_json = query_result.visual_json
        else:
            visual_json = None
        query_result = sch.MessageQueryResult(
            result_uuid=query_result.result_uuid if query_result.result_uuid else None,
            sql_query=query_result.sql_query,
            result_type=query_result.result_type,
            visual_result_uuid=(query_result.visual_result_uuid if query_result.visual_result_uuid else None),
            visual_json=visual_json,
            visual_explanation=query_result.visual_explanation,
        )
        query_result_dict = query_result.model_dump()
        return query_result_dict

    def create_api_message(self) -> sch.APIMessage:
        query_result_dict = self.process_query_result(query_result=self.query_result)
        if self.chat_metadata.semcache_response:
            verified_user_role = self.chat_metadata.semcache_response.verified_user_role
            can_verify = self.chat_metadata.semcache_response.can_verify
            verified_user_uuid = self.chat_metadata.semcache_response.verified_user_uuid
            verified = self.chat_metadata.semcache_response.verified
        else:
            verified_user_role = None
            can_verify = None
            verified_user_uuid = None
            verified = False
        # HACK: Need to use a constant instead: https://github.com/Basejump-AI/Basejump/issues/1441
        if self.message.content.strip() == "Reached max iterations.":
            raise Exception("Reached max iterations.")
        api_message = sch.APIMessage(
            # vars from ChatMessage
            role=self.message.role,
            msg_type=self.message.msg_type,
            # content=self.message.content,
            # TODO: Move this into be passed in the body instead
            # special characters in the content can cause issues if sent via header
            content=self.message.content,
            timestamp=self.message.timestamp,
            msg_uuid=self.message.msg_uuid,
            # vars from PromptMetadata
            prompt_uuid=self.prompt_metadata.prompt_uuid,
            initial_prompt=self.prompt_metadata.initial_prompt,
            prompt_time=self.prompt_metadata.prompt_time,
            parent_msg_uuid=self.chat_metadata.parent_msg_uuid,
            verified=verified,
            verified_user_role=verified_user_role,
            verified_user_uuid=verified_user_uuid,
            can_verify=can_verify,
            # vars from QueryResult
            **query_result_dict,
        )
        return api_message

    async def save_messages(self, db: AsyncSession):
        assert isinstance(self.prompt_metadata, sch.PromptMetadata)
        # TODO: Performance could possibly be improved to not update the vector DB table every time
        # for the index_created flg
        await crud_chat.index_chat_history(
            db=db,
            client_uuid=self.prompt_metadata.client_uuid,
            chat_id=self.chat_metadata.chat_id,
            chat_uuid=self.chat_metadata.chat_uuid,
            vector_id=self.chat_metadata.vector_id,
            chat_history=self.chat_metadata.curr_chat_history,
            callback_manager=self.prompt_metadata.callback_manager,
            vector_store=self.chat_metadata.vector_store,
            embedding_model_info=self.chat_metadata.embedding_model_info,
        )
        for api_message in self.chat_metadata.curr_chat_history:
            await crud_chat.save_message(
                db=db,
                message=api_message,
                prompt_metadata=self.prompt_metadata,
                chat_metadata=self.chat_metadata,
                query_result=self.query_result,
                msg_in_index=True,
            )
        # Remove everything from current chat history
        self.chat_metadata.curr_chat_history = []
        # Remove extra chat history to prevent vector DB from getting too large
        msg_uuids = await crud_chat.get_chat_history_for_chats(db=db, chat_ids=[self.chat_metadata.chat_id])
        len_chats = len(msg_uuids)
        if len_chats > constants.MAX_CHAT_HISTORY:
            # Find the number of chat messages to remove
            chat_num_to_remove = len_chats - constants.MAX_CHAT_HISTORY
            await crud_chat.delete_chat_msgs_from_vector(
                db=db,
                client_id=self.prompt_metadata.client_id,
                msg_uuids=msg_uuids[:chat_num_to_remove],
                redis_client_async=self.redis_client_async,
            )

    async def _send_solution_message(self, db: AsyncSession):
        try:
            assert isinstance(self.prompt_metadata, sch.PromptMetadata)
        except AssertionError:
            raise AssertionError(
                "send_solution_message can only be used within the background task after helper_run_chat has been ran"
            )
        await self.save_messages(db=db)
        super().create_message(
            role=MessageRole.ASSISTANT,
            content="",
            msg_type=enums.MessageType.SOLUTION,
        )
        api_message = self.format_message()
        await self._send_api_message(api_message=api_message)


class AgentSetup:
    def __init__(self, prompt_metadata: sch.PromptMetadata):
        """
        Setup methods for agents
        """
        self.prompt_metadata = prompt_metadata

    @staticmethod
    def _load_from_prompt_metadata(prompt_metadata_base: sch.PromptMetadataBase):
        callback_managers = create_callback_mgrs()

        # NOTE: Re-instantiating prompt metadata here since this is background submitted
        prompt_metadata = sch.PromptMetadata(
            **prompt_metadata_base.dict(),
            token_counter=callback_managers.token_counter,
            llama_debug=callback_managers.llama_debug,
            callback_manager=callback_managers.callback_manager,
        )
        return prompt_metadata

    @classmethod
    def load_from_prompt_metadata(cls, prompt_metadata_base: sch.PromptMetadataBase):
        prompt_metadata = cls._load_from_prompt_metadata(prompt_metadata_base=prompt_metadata_base)
        return cls(prompt_metadata=prompt_metadata)


class ChatAgentSetup(AgentSetup):
    def __init__(
        self,
        db: AsyncSession,
        embedding_model_info: sch.AzureModelInfo,
        prompt_metadata: sch.PromptMetadata,
        chat_metadata: sch.ChatMetadata,
        redis_client_async: RedisAsync,
        team_info: sch.TeamFields,
        system_prompt: Optional[str] = None,
    ):
        """
        Setup methods for chat agents
        """
        super().__init__(prompt_metadata=prompt_metadata)
        self.db = db
        self.chat_metadata = chat_metadata
        self.redis_client_async = redis_client_async
        self.embedding_model_info = embedding_model_info
        self.team_info = team_info
        # TODO: Move into the prompts module
        self.system_prompt = (
            system_prompt
            or f"""\
You are used to help company employees (called users) answer data related questions by creating SQL queries to query \
their internal database or by answering question directly if you already have enough information. \
If you feel the user's prompt is ambiguous or needs clarification, ask the user follow-up questions to ensure \
you have enough context. Don't ask the user for column or table names for a query since it is your responsibility \
to help them explore and understand tables and columns in the database. \
Your responses are based on being provided to the {team_info.team_name} team. This is a quick description \
of their team to inform your responses: {team_info.team_desc} \
To create charts, the {constants.VIS_TOOL_NM} must be used. \
The chat history may have certain context after certain keywords/keyphrases at the end of a given message. \
Here are the keywords/keyphrases:
'{constants.SQL_QUERY_TXT}' - Information following this key phrase was the SQL you generated to provide your answer. \
When a user asks a follow up question \
for a question you answered, you can take the prior SQL and build off of it as a starting point if desired.
'{constants.TIMESTAMP_TXT}' - Information following this keyword is the time that this particular message \
was sent to the user.
'{constants.VISUAL_RESULT_UUID}' - Information following this key phrase is the result UUID that is related \
to the dataset generated for the chat response.
'{constants.VISUAL_CONFIG}' - Information following this key phrase contains some metadata of the visualization \
that was generated for this chat response.
Don't structure your output with the keywords and keyphrases since they're only meant to provide you with more context
"""
        )

    # TODO: Review if this should be an instance method instead
    @classmethod
    async def load_from_metadata(
        cls,
        db: AsyncSession,
        prompt_metadata_base: sch.PromptMetadataBase,
        chat_metadata: sch.ChatMetadata,
        redis_client_async: RedisAsync,
        embedding_model_info: sch.AzureModelInfo,
        team_info: sch.TeamFields,
    ):
        prompt_metadata = cls._load_from_prompt_metadata(prompt_metadata_base=prompt_metadata_base)
        cls.chat_metadata = chat_metadata
        return cls(
            prompt_metadata=prompt_metadata,
            chat_metadata=chat_metadata,
            db=db,
            redis_client_async=redis_client_async,
            embedding_model_info=embedding_model_info,
            team_info=team_info,
        )

    @staticmethod
    async def get_connections(db: AsyncSession, team_id: int, user_id: int) -> list[models.Connection]:
        # Get the connections available for the AI
        try:
            connections = await crud_connection.get_connections(db=db, team_id=team_id, user_id=user_id)
        except Exception as e:
            logger.error(e)
            raise errors.GetTeamConnError
        return connections

    async def get_chat(self) -> models.Chat:
        chat = await crud_chat.get_chat(
            db=self.db, chat_uuid=self.chat_metadata.chat_uuid, user_id=self.prompt_metadata.user_id
        )
        try:
            assert chat
        except AssertionError:
            raise errors.ChatUUIDNotFound
        if not chat.chat_name:
            chat.chat_name = self.prompt_metadata.initial_prompt
        return chat

    async def get_chat_history(self, chat: models.Chat) -> list[ChatMessage]:
        try:
            full_chat_history = await crud_chat.get_chat_history_for_ai(db=self.db, chat_id=chat.chat_id)
        except Exception as e:
            logger.error("Error retrieving get chat history %s", str(e))
            raise errors.GetChatHistoryError
        return await self.update_history_from_index(full_chat_history=full_chat_history)

    async def update_history_from_index(self, full_chat_history: list[ChatMessage]) -> list[ChatMessage]:
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
        vector_store = RedisVectorStore(redis_client_async=self.redis_client_async, schema=schema, legacy_filters=True)
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
        embed_model = ai_catalog.get_embedding_model(model_info=self.embedding_model_info)
        vector_memory = VectorMemory.from_defaults(
            vector_store=vector_store,
            embed_model=embed_model,
            retriever_kwargs={"similarity_top_k": TOP_K, "filters": filters},
        )

        if not full_chat_history:
            return [ChatMessage(role=MessageRole.SYSTEM, content=self.system_prompt, timestamp=datetime.now())]
        try:
            msgs = await vector_memory.aget(input=self.prompt_metadata.initial_prompt)
            logger.info("Retrieved the following messages for chat history: %s", msgs)
            if not all([message.timestamp for message in msgs]):
                return [ChatMessage(role=MessageRole.SYSTEM, content=self.system_prompt, timestamp=datetime.now())]
            min_timestamp = min([message.timestamp for message in msgs]) - timedelta(seconds=1)  # type: ignore
            # Always add the system prompt as the earliest timestamped message
            chat_history = [ChatMessage(role=MessageRole.SYSTEM, content=self.system_prompt, timestamp=min_timestamp)]
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


class BaseAgent(ABC):
    """Concrete class for agents

    See Also
    --------
    BaseChatAgent
        Use the BaseChatAgent if you want to track the chat history of the agent with a human in the loop
    """

    def __init__(
        self,
        prompt_metadata: sch.PromptMetadata,
        chat_history: Optional[list[ChatMessage]],
        redis_client_async: RedisAsync,
        sql_engine: AsyncEngine,
        large_model_info: sch.ModelInfo,
        agent_llm: Optional[FunctionCallingLLM] = None,
        max_iterations: int = constants.MAX_ITERATIONS,
    ):
        self.prompt_metadata = prompt_metadata
        self.query_result: Optional[sch.MessageQueryResult] = None
        ai_catalog = AICatalog(callback_manager=prompt_metadata.callback_manager)
        self.agent_llm: FunctionCallingLLM = agent_llm or ai_catalog.get_llm(model_info=large_model_info)
        self.memory = ChatMemoryBuffer.from_defaults(chat_history=chat_history, llm=self.agent_llm)
        self.chat_history = chat_history or []
        self.max_iterations = max_iterations  # NOTE: This only works with streaming off
        self.sql_engine = sql_engine

    @abstractmethod
    def get_llm_type() -> enums.LLMType:  # type: ignore
        pass

    @abstractmethod
    async def setup_tools(self) -> Sequence[AsyncBaseTool]:
        pass

    @abstractmethod
    def _chat(self, prompt: str):
        """Pass in the user question to the AI

        Parameters
        ----------
        prompt
            The user question written in natural language

        """
        pass

    async def handle_malformed_llm_output(self, exception: Exception):
        error_text = str(exception)
        if COULD_NOT_PARSE_TXT in error_text:
            logger.warning("Incorrect output format - attempting to re-prompt to self-heal")
            input = f"""Incorrect output format. Here was your response with the incorrect format:\n\n\
{error_text.split(COULD_NOT_PARSE_TXT)[1]}. \n\nHere is a reminder of \
instructions for the expected output format: \n{EXPECTED_OUTPUT_INSTRUCTIONS}\
\nCorrect the message to use the correct format."""
            message = await self.provide_input(input=input)
            return message
        elif "() missing 1 required positional argument:" in error_text:
            logger.warning("Incorrect output format - attempting to re-prompt to self-heal")
            input = f"""Incorrect format for tool use. Here was the error:\n\n\
{error_text}. Make sure to include all required fields for 'Action Input'. Try again and fix your error."""
            message = await self.provide_input(input=input)
            return message
        else:
            raise exception

    def _get_response_hook(self):
        return None

    async def setup_agent(self) -> BaseChatEngine:
        """Setting up the chat agent"""
        tools = await self.setup_tools()
        # If no tools, then create a simple chat engine
        if not tools:
            agent = SimpleChatEngine.from_defaults(
                llm=self.agent_llm,
                memory=self.memory,
                callback_manager=self.agent_llm.callback_manager,
            )
        else:
            logger.debug("Here are the tools: %s", tools)
            agent = FunctionCallingAgent.from_tools(  # type: ignore
                tools,  # type: ignore
                llm=self.agent_llm,
                verbose=True,
                memory=self.memory,
                max_function_calls=self.max_iterations,
                callback_manager=self.agent_llm.callback_manager,
                response_hook=self._get_response_hook(),
            )
        return agent

    async def _prompt_agent(self) -> sch.Message:
        """This sets up the AI agent and inputs the prompt"""
        try:
            self.agent = await self.setup_agent()
            message = await self._chat(prompt=self.prompt_metadata.initial_prompt)
            await crud_chat.save_token_counts(db=self.db, prompt_metadata=self.prompt_metadata)
        except Exception as e:
            try:
                message = await self.handle_malformed_llm_output(exception=e)
            # TODO: Be more specific with the exception here
            except Exception as e:
                error_text = str(e)
                logger.error(error_text)
                raise e
        return message

    async def prompt_agent(self) -> sch.Message:
        """This does initial setup and tear down of the database for the chat"""
        # NOTE: Background tasks need their own session so the DB is created here and not in init
        try:
            local_session = LocalSession(client_id=self.prompt_metadata.client_id, engine=self.sql_engine)
            Session = await local_session.get_session()
            async with Session() as session:
                self.db = session
                message = await self._prompt_agent()
            return message
        except (errors.GetTeamConnError, errors.SQLIndexError) as e:
            raise e
        except Exception:
            raise errors.PromptingAIError

    async def _get_message(self, response: str) -> sch.Message:
        handler = MessageHandler(prompt_metadata=self.prompt_metadata, query_result=self.query_result)
        try:
            handler.create_message(
                role=MessageRole.ASSISTANT,
                msg_type=enums.MessageType.RESPONSE,
                content=response,
            )
            return handler.message
        except Exception as e:
            logger.error(str(e))
            return sch.Message(
                role=MessageRole.ASSISTANT,
                msg_type=enums.MessageType.RESPONSE,
                content=response,
                timestamp=datetime.now(),
            )

    async def _chat_base(
        self,
        prompt: str,
        task: Optional[Task] = None,
        step: Optional[TaskStep] = None,
        chat_history: Optional[list[ChatMessage]] = None,
        input: Optional[str] = None,
    ) -> sch.Message:
        """Chat with the AI

        Parameters
        ----------
        prompt
            The prompt to chat with the AI
        """
        if isinstance(self.agent, FunctionCallingAgent):
            agent_output = await self.agent.achat(
                message=prompt, task=task, chat_history=chat_history, input=input, step=step
            )
        else:
            agent_output = await self.agent.achat(message=prompt)
        response_list = []
        for sentence in agent_output.response.split("."):
            if "Option 1:" in sentence:
                continue
            else:
                response_list.append(sentence)
        response = ".".join(response_list).replace("Answer:", "").replace("Thought:", "")
        return await self._get_message(response=response)

    async def provide_input(self, input: str, chat_message: Optional[ChatMessage] = None) -> sch.Message:
        message = await self._chat_base(
            prompt=self.prompt_metadata.initial_prompt,
            task=self.agent.current_task,  # type: ignore
            step=self.agent.current_step,  # type: ignore
            input=input,
        )
        return message


class BaseChatAgent(BaseAgent):
    """An agent intended for chat with a human in the loop"""

    def __init__(
        self,
        prompt_metadata: sch.PromptMetadata,
        chat_history: Optional[list[ChatMessage]],
        chat_metadata: sch.ChatMetadata,
        redis_client_async: RedisAsync,
        sql_engine: AsyncEngine,
        large_model_info: sch.ModelInfo,
        agent_llm: Optional[FunctionCallingLLM] = None,
        max_iterations: int = constants.MAX_ITERATIONS,
    ):
        super().__init__(
            prompt_metadata=prompt_metadata,
            chat_history=chat_history,
            agent_llm=agent_llm,
            max_iterations=max_iterations,
            redis_client_async=redis_client_async,
            sql_engine=sql_engine,
            large_model_info=large_model_info,
        )
        self.chat_metadata = chat_metadata
        self.redis_client_async = redis_client_async

    async def _prompt_agent(self) -> sch.Message:
        try:
            message = await super()._prompt_agent()
            await self.redis_client_async.publish(str(self.chat_metadata.parent_msg_uuid), constants.PUBSUB_DONEWORD)
            if message.content == "Reached max iterations.":
                raise Exception("Reached max iterations.")
            return message
        except Exception as e:
            error_text = str(e)
            # NOTE: This will be sent to the user
            if (
                isinstance(e, sch.SQLTimeoutError)
                or isinstance(e, errors.LowConfidenceResponse)
                or isinstance(e, errors.StrictModeFlagged)
            ):
                error_msg = error_text
            elif error_text == "Reached max iterations.":
                error_msg = (
                    "Sorry I can't seem to find an answer to that question. "
                    "I've reached my max attempts. "
                    "Please try re-phrasing the prompt and ask again."
                )
            elif error_text in [
                constants.NO_PERMITTED_TABLES,
                constants.NO_TABLES,
                constants.REINDEXING_DB_ERROR_MSG,
                constants.INDEX_DB_ERROR_MSG,
                constants.UNRESOLVED_JINJA,
            ]:
                error_msg = error_text
            elif constants.CONTENT_MGMT_POLICY in error_text:
                error_msg = """Your prompt triggered a responsible AI policy violation. \
For more information about our content management policy please refer to this link: \
https://go.microsoft.com/fwlink/?linkid=2198766"""
            else:
                logger.error(str(e))
                error_msg = errors.PROMPTING_AI_ERROR
            handler = ChatMessageHandler(
                prompt_metadata=self.prompt_metadata,
                chat_metadata=self.chat_metadata,
                redis_client_async=self.redis_client_async,
            )
            if self.chat_metadata.semcache_response:
                self.chat_metadata.semcache_response.verified = False
            await handler.create_message(
                db=self.db,
                role=sch.MessageRole.ASSISTANT,
                content=error_msg,
                msg_type=enums.MessageType.ERROR,
            )
            await handler.save_message(message=handler.message)
            await handler.save_messages(db=self.db)
            await handler.send_api_message()
            raise e

    async def _get_message(self, response: str) -> sch.Message:
        handler = ChatMessageHandler(
            prompt_metadata=self.prompt_metadata,
            chat_metadata=self.chat_metadata,
            query_result=self.query_result,
            redis_client_async=self.redis_client_async,
        )
        await handler.create_message(
            db=self.db,
            role=MessageRole.ASSISTANT,
            msg_type=enums.MessageType.RESPONSE,
            content=response,
        )
        if self.chat_metadata.send_message:
            await handler.save_message(message=handler.message)
            await handler.send_api_message(send_solution=sch.SendSolution(db=self.db))
        else:
            # Need to save messages if not sending them
            # otherwise send_api_message saves the messages so no need to put it above
            await handler.save_message(message=handler.message)
            await handler.save_messages(db=self.db)
        return handler.message

    async def response_hook(self, text):
        # If already logged, then create a new message UUID since we are logging per SQL query execution
        # logger.info("Webhook messages: %s", text)
        # If webhook is set, then post the thoughts to the webhook
        thoughts = []
        sentence_ls_base = re.split(r"\.\s|\.\n", text)
        # Recombine sentences if they don't start capitalized (e.g. table names)
        sentence_ls: list = []
        for idx, sentence in enumerate(sentence_ls_base):
            if sentence[0].islower() and idx > 0:
                # Add to the prior sentence
                sentence_ls[-1] += f".{sentence}"
            else:
                sentence_ls.append(sentence)
        for sentence in sentence_ls:
            logger.info("Here is a pre-filtered thought: %s", sentence)
            # TODO: Make this more robust
            # TODO: Fix the hard reference to structured_sql_generation_tool
            if not sentence:
                continue
            if (
                constants.SQL_TABLES_TOOL_NM_PREFIX in sentence
                or constants.SQL_EXEC_TOOL_NM_PREFIX in sentence
                or constants.VIS_TOOL_NM in sentence
                or constants.INTERNAL_DOCS_TOOL_NM in sentence
            ):
                continue
            if "The current language" in sentence:
                continue
            if "I need to use a tool" in sentence:
                continue
            if "Option 1:" in sentence:
                continue
            if "UUID" in sentence or "uuid" in sentence:
                continue
            if "== Plan" in sentence:
                continue
            if "Use the '" in sentence:
                continue
            if "prefix for the plan" in sentence:
                continue
            # if SQL_OPTION_1 in sentence or SQL_OPTION_2_SUFFIX in sentence or SQL_OPTION_3_SUFFIX:
            #     continue
            else:
                thoughts.append(sentence.strip())
        for thought in thoughts:
            if not thought:
                continue
            logger.debug("Webhook message: %s", thought)
            handler = ChatMessageHandler(
                prompt_metadata=self.prompt_metadata,
                chat_metadata=self.chat_metadata,
                redis_client_async=self.redis_client_async,
            )
            await handler.create_message(
                db=self.db, role=MessageRole.ASSISTANT, content=thought, msg_type=enums.MessageType.THOUGHT
            )
            await handler.send_api_message()

    def _get_response_hook(self):
        return self.response_hook


class SimpleAgent(BaseAgent):
    """An AI Agent with the bare minimum"""

    def __init__(
        self,
        prompt_metadata: sch.PromptMetadata,
        sql_engine: AsyncEngine,
        large_model_info: sch.ModelInfo,
        redis_client_async: RedisAsync,
        chat_history: Optional[list[ChatMessage]] = None,
        agent_llm: Optional[FunctionCallingLLM] = None,
        max_iterations: int = 10,
    ):
        super().__init__(
            prompt_metadata=prompt_metadata,
            chat_history=chat_history,
            max_iterations=max_iterations,
            agent_llm=agent_llm,
            sql_engine=sql_engine,
            large_model_info=large_model_info,
            redis_client_async=redis_client_async,
        )

    @staticmethod
    def get_llm_type() -> enums.LLMType:
        return enums.LLMType.SIMPLE_AGENT

    async def setup_tools(self) -> list[AsyncBaseTool]:
        return []

    async def _chat(self, prompt: str) -> sch.Message:
        logger.debug("Here is the prompt sent to the simple agent: %s", prompt)
        return await self._chat_base(prompt=prompt)
