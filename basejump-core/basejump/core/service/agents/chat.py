import re
from typing import Optional

from llama_index.core.llms import MessageRole
from llama_index.core.llms.function_calling import FunctionCallingLLM

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.result import store
from basejump.core.models import constants, enums, errors
from basejump.core.models import schemas as sch
from basejump.core.service.agents.memory.agent import AgentMemory
from basejump.core.service.agents.message import ChatMessageHandler
from basejump.core.service.agents.simple import SimpleAgent

logger = set_logging(handler_option="stream", name=__name__)


class ChatAgent(SimpleAgent):
    """An agent intended for chat with a human in the loop"""

    def __init__(
        self,
        prompt_metadata: sch.PromptMetadata,
        chat_metadata: sch.ChatMetadata,
        service_context: sch.ServiceContext,
        memory: AgentMemory,
        result_store: Optional[store.ResultStore] = None,
        agent_llm: Optional[FunctionCallingLLM] = None,
        max_iterations: int = constants.MAX_ITERATIONS,
        verbose: bool = False,
    ):
        super().__init__(
            prompt_metadata=prompt_metadata,
            memory=memory,
            agent_llm=agent_llm,
            max_iterations=max_iterations,
            service_context=service_context,
            result_store=result_store,
            verbose=verbose,
        )
        self.chat_metadata = chat_metadata
        self.redis_client_async = service_context.redis_client_async

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
                role=MessageRole.ASSISTANT,
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
