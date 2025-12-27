import copy
from typing import Optional, Type

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.aicatalog import AICatalog
from basejump.core.models import pydantic_ai_formats as fmt
from basejump.core.models import schemas as sch
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import LLM, ChatMessage
from llama_index.program.openai import OpenAIPydanticProgram
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

logger = set_logging(handler_option="stream", name=__name__)


class JSONResponseFormatter:
    def __init__(
        self,
        response: str,
        pydantic_format: Type[BaseModel],
        small_model_info: sch.ModelInfo,
        max_tokens: int = 500,
        llm: Optional[LLM] = None,
    ):
        self.response = response
        self.pydantic_format = pydantic_format
        self.max_tokens = max_tokens
        self.llm = llm
        self.small_model_info = small_model_info

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
            ai_catalog = AICatalog()
            self.llm = ai_catalog.get_llm(model_info=my_small_model_info)
        program = OpenAIPydanticProgram.from_defaults(
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
    ):
        super().__init__(
            response=response,
            pydantic_format=pydantic_format,
            max_tokens=max_tokens,
            small_model_info=small_model_info,
            llm=llm,
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


async def get_title_description(
    db: AsyncSession,
    prompt_metadata: sch.PromptMetadata,
    sql_query: str,
    query_result: str,
    small_model_info: sch.ModelInfo,
) -> fmt.DescriptionFormat:
    prompt = f"""\
Summarize the following query results into a title and description. \
You will be given the original user prompt, the SQL query to answer the prompt, \
and the query results. DO NOT use any numbers or specific values in the title or description. \n
Prompt: {prompt_metadata.initial_prompt}\n
SQL Query: {sql_query}\n
SQL Results: {query_result}\n
    """
    format_json_response = JSONResponseFormatter(
        response=prompt, pydantic_format=fmt.DescriptionFormat, small_model_info=small_model_info
    )
    return await format_json_response.format()
