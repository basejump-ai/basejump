"""Contains parent classes for the service directory"""

import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Sequence

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
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.tools.types import AsyncBaseTool
from redis.asyncio import Redis as RedisAsync
from sqlalchemy.ext.asyncio import AsyncEngine

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.crud import crud_chat
from basejump.core.database.session import LocalSession
from basejump.core.models import constants, enums, errors
from basejump.core.models import schemas as sch
from basejump.core.models.ai.catalog import AICatalog
from basejump.core.service.agents.memory.base import BaseAgentMemory, SimpleAgentMemory
from basejump.core.service.agents.message import ChatMessageHandler, MessageHandler

logger = set_logging(handler_option="stream", name=__name__)


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
        memory: BaseAgentMemory,
        redis_client_async: RedisAsync,
        sql_engine: AsyncEngine,
        large_model_info: sch.ModelInfo,
        agent_llm: Optional[FunctionCallingLLM] = None,
        max_iterations: int = constants.MAX_ITERATIONS,
        verbose: bool = False,
    ):
        self.prompt_metadata = prompt_metadata
        self.query_result: Optional[sch.MessageQueryResult] = None
        ai_catalog = AICatalog(callback_manager=prompt_metadata.callback_manager)
        self.agent_llm: FunctionCallingLLM = agent_llm or ai_catalog.get_llm(model_info=large_model_info)
        self.initial_memory = ChatMemoryBuffer.from_defaults(chat_history=memory.chat_history, llm=self.agent_llm)
        self.memory = memory
        self.max_iterations = max_iterations  # NOTE: This only works with streaming off
        self.sql_engine = sql_engine
        self.verbose = verbose

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
                memory=self.initial_memory,
                callback_manager=self.agent_llm.callback_manager,
            )
        else:
            logger.debug("Here are the tools: %s", tools)
            agent = FunctionCallingAgent.from_tools(  # type: ignore
                tools,  # type: ignore
                llm=self.agent_llm,
                verbose=self.verbose,
                memory=self.initial_memory,
                max_function_calls=self.max_iterations,
                callback_manager=self.agent_llm.callback_manager,
                response_hook=self._get_response_hook(),
                # NOTE: If not set to False, then the same result UUID can be applied to 2 SQL results
                allow_parallel_tool_calls=False,
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
        memory: BaseAgentMemory,
        chat_metadata: sch.ChatMetadata,
        redis_client_async: RedisAsync,
        sql_engine: AsyncEngine,
        large_model_info: sch.ModelInfo,
        agent_llm: Optional[FunctionCallingLLM] = None,
        max_iterations: int = constants.MAX_ITERATIONS,
        verbose: bool = False,
    ):
        super().__init__(
            prompt_metadata=prompt_metadata,
            memory=memory,
            agent_llm=agent_llm,
            max_iterations=max_iterations,
            redis_client_async=redis_client_async,
            sql_engine=sql_engine,
            large_model_info=large_model_info,
            verbose=verbose,
        )
        self.chat_metadata = chat_metadata
        self.redis_client_async = redis_client_async

    async def _prompt_agent(self) -> sch.Message:
        try:
            message = await super()._prompt_agent()
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
                verbose=self.verbose,
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
            verbose=self.verbose,
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
        sentence_ls_base = re.split(r"(?<=[a-zA-Z])\.(?=\s)", text)
        # Recombine sentences if they don't start capitalized (e.g. table names)
        sentence_ls: list = []
        for idx, sentence in enumerate(sentence_ls_base):
            if sentence[0].islower() and idx > 0:
                # Add to the prior sentence
                sentence_ls[-1] += f".{sentence}"
            else:
                sentence_ls.append(sentence)
        for sentence in sentence_ls:
            logger.debug("Initial LLM thought: %s", sentence)
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
            if ">>" in sentence:
                continue
            if "Use the '" in sentence:
                continue
            if "prefix for the plan" in sentence:
                continue
            # if SQL_OPTION_1 in sentence or SQL_OPTION_2_SUFFIX in sentence or SQL_OPTION_3_SUFFIX:
            #     continue
            else:
                logger.info("LLM Thought: %s", sentence)
                thoughts.append(sentence.strip())
        for thought in thoughts:
            if not thought:
                continue
            if self.verbose:
                logger.debug("Webhook message: %s", thought)
            handler = ChatMessageHandler(
                prompt_metadata=self.prompt_metadata,
                chat_metadata=self.chat_metadata,
                redis_client_async=self.redis_client_async,
                verbose=self.verbose,
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
        memory: SimpleAgentMemory,
        agent_llm: Optional[FunctionCallingLLM] = None,
        max_iterations: int = 10,
        verbose: bool = False,
    ):
        super().__init__(
            prompt_metadata=prompt_metadata,
            memory=memory,
            max_iterations=max_iterations,
            agent_llm=agent_llm,
            sql_engine=sql_engine,
            large_model_info=large_model_info,
            redis_client_async=redis_client_async,
            verbose=verbose,
        )

    @staticmethod
    def get_llm_type() -> enums.LLMType:
        return enums.LLMType.SIMPLE_AGENT

    async def setup_tools(self) -> list[AsyncBaseTool]:
        return []

    async def _chat(self, prompt: str) -> sch.Message:
        logger.debug("Here is the prompt sent to the simple agent: %s", prompt)
        return await self._chat_base(prompt=prompt)
