import copy
from typing import Optional, Type

from llama_index.core import ChatPromptTemplate
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import LLM, ChatMessage
from llama_index.core.program import FunctionCallingProgram
from pydantic import BaseModel

from basejump.core.common.config.logconfig import set_logging
from basejump.core.models import schemas as sch
from basejump.core.models.ai.catalog import AICatalog

logger = set_logging(handler_option="stream", name=__name__)


class JSONResponseFormatter:
    def __init__(
        self,
        response: str,
        pydantic_format: Type[BaseModel],
        small_model_info: sch.ModelInfo,
        max_tokens: int = 500,
        llm: Optional[LLM] = None,
        callback_manager: Optional[CallbackManager] = None,
    ):
        """
        Parameters
        ----------
        callback_manager
            The conversation's callback manager, so the formatting LLM's token usage is recorded. Only used when
            no llm is passed in; a passed in llm reports to its own callback manager.
        """
        self.response = response
        self.pydantic_format = pydantic_format
        self.max_tokens = max_tokens
        self.llm = llm
        self.small_model_info = small_model_info
        self.callback_manager = callback_manager

    @property
    def feedback_template(self):
        return ChatPromptTemplate(
            message_templates=[
                ChatMessage(
                    role="system",
                    content=("You are an expert assistant for formatting a response into JSON."),
                ),
                ChatMessage(
                    role="user",
                    content=("Here is the response: \n" "------\n" "{response}\n" "------"),
                ),
            ]
        )

    def _format_json_response(self):
        if not self.llm:
            my_small_model_info = copy.deepcopy(self.small_model_info)
            my_small_model_info.max_tokens = self.max_tokens
            ai_catalog = AICatalog(callback_manager=self.callback_manager)
            self.llm = ai_catalog.get_llm(model_info=my_small_model_info)
        program = FunctionCallingProgram.from_defaults(
            output_cls=self.pydantic_format,
            llm=self.llm,
            prompt=self.feedback_template,
        )
        return program

    def format_sync(self):
        program = self._format_json_response()
        return program(response=self.response)

    async def format(self):
        program = self._format_json_response()
        return await program.acall(response=self.response)

    # TODO: Some of these params can be simplified into classes
    # TODO: Replace this with composition - pass in an agent


class DateFormatter(JSONResponseFormatter):
    def __init__(
        self,
        response: str,
        pydantic_format: Type[BaseModel],
        small_model_info: sch.ModelInfo,
        max_tokens: int = 500,
        llm: Optional[LLM] = None,
        callback_manager: Optional[CallbackManager] = None,
    ):
        super().__init__(
            response=response,
            pydantic_format=pydantic_format,
            max_tokens=max_tokens,
            small_model_info=small_model_info,
            llm=llm,
            callback_manager=callback_manager,
        )

    @property
    def feedback_template(self):
        return ChatPromptTemplate(
            message_templates=[
                ChatMessage(
                    role="system",
                    content=("You are an expert assistant for formatting a list of dates into the YYYY-MM-DD format."),
                ),
                ChatMessage(
                    role="user",
                    content=("Here is the response: \n" "------\n" "{response}\n" "------"),
                ),
            ]
        )
