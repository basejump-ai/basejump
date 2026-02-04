"""Contains concrete base classes for the agents package"""

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
from llama_index.core.tools.types import AsyncBaseTool

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.crud import crud_chat
from basejump.core.database.result import store
from basejump.core.database.session import LocalSession
from basejump.core.models import constants, enums, errors
from basejump.core.models import schemas as sch
from basejump.core.models.ai.catalog import AICatalog
from basejump.core.service.agents.message import MessageHandler

logger = set_logging(handler_option="stream", name=__name__)


class BaseAgentMemory(ABC):
    def __init__(
        self,
        service_context: sch.ServiceContext,
        prompt_metadata: sch.PromptMetadata,
        chat_history: Optional[list[ChatMessage]] = None,
        query_result: Optional[sch.MessageQueryResult] = None,
        result_store: Optional[store.ResultStore] = None,
    ):
        self.service_context = service_context
        self.prompt_metadata = prompt_metadata
        self.chat_history = chat_history or []
        self.query_result = query_result or sch.MessageQueryResult()
        self.result_store = result_store or store.LocalResultStore(client_id=self.prompt_metadata.client_id)


class BaseAgent(ABC):
    """Concrete class for agents

    See Also
    --------
    ChatAgent
        Use the ChatAgent if you want to track the chat history of the agent with a human in the loop
    """

    def __init__(
        self,
        prompt_metadata: sch.PromptMetadata,
        service_context: sch.ServiceContext,
        memory: BaseAgentMemory,
        llm: Optional[FunctionCallingLLM] = None,
        max_iterations: int = constants.MAX_ITERATIONS,
        verbose: bool = False,
    ):
        self.prompt_metadata = prompt_metadata
        self.service_context = service_context
        ai_catalog = AICatalog(callback_manager=prompt_metadata.callback_manager)
        self.llm: FunctionCallingLLM = llm or ai_catalog.get_llm(model_info=self.service_context.large_model_info)
        self.memory = memory
        self.max_iterations = max_iterations  # NOTE: This only works with streaming off
        self.sql_engine = self.service_context.sql_engine
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
                llm=self.llm,
                callback_manager=self.llm.callback_manager,
            )
        else:
            logger.debug("Here are the tools: %s", tools)
            agent = FunctionCallingAgent.from_tools(  # type: ignore
                tools,  # type: ignore
                llm=self.llm,
                verbose=self.verbose,
                max_function_calls=self.max_iterations,
                callback_manager=self.llm.callback_manager,
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
        handler = MessageHandler(prompt_metadata=self.prompt_metadata, query_result=self.memory.query_result)
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
