import uuid

from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import FunctionTool
from llama_index.core.tools.function_tool import create_tool_metadata
from sqlalchemy.ext.asyncio import AsyncSession

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.vector import utils as vector_utils
from basejump.core.models import constants, enums
from basejump.core.models import schemas as sch
from basejump.core.models.ai.catalog import AICatalog
from basejump.core.service.agents.message import ChatMessageHandler
from basejump.core.service.agents.tools import tool_utils
from basejump.core.service.agents.tools.base import BaseTool

SCORE_THRESHOLD = 0.3
TOP_K = 8
INDEX_NAME = "basejump_internal_docs"
DELIMITER = "<--->"
logger = set_logging(handler_option="stream", name=__name__)


class DocsTool(BaseTool):
    def __init__(
        self,
        db: AsyncSession,
        client_id: int,
        db_id: int,
        db_uuid: uuid.UUID,
        llm: FunctionCallingLLM,
        service_context: sch.ServiceContext,
        prompt_metadata: sch.PromptMetadata,
        chat_metadata: sch.ChatMetadata,
    ):
        self.db = db
        self.client_id = client_id
        self.db_id = db_id
        self.db_uuid = db_uuid
        self.llm = llm
        self.service_context = service_context
        self.prompt_metadata = prompt_metadata
        self.chat_metadata = chat_metadata

    def get_tool(self) -> FunctionTool:
        func = self.get_docs
        tool_metadata = create_tool_metadata(
            fn=func,
            name=constants.get_docs_tool_nm(db_id=self.db_id),
            description="""This tool returns additional information \
about the type of information available in the database based on your prompt. \
This can be useful for supplemental information beyond table and column definitions such as \
definitions of values not found in the database metadata, or meaning of specific terms unique \
to the organization in charge of the database.""",
        )
        tool = FunctionTool.from_defaults(fn=func, async_fn=func, tool_metadata=tool_metadata)
        return tool

    async def get_tools(self) -> list[FunctionTool]:
        return [self.get_tool()]

    async def get_docs(self, prompt: str, max_docs: int = 10):
        # Increase the max tokens for a longer response
        logger.debug("Searching documentation...")
        await tool_utils.update_llm_tokens(llm=self.llm, max_tokens=5000)
        handler = ChatMessageHandler(
            prompt_metadata=self.prompt_metadata,
            chat_metadata=self.chat_metadata,
            redis_client_async=self.service_context.redis_client_async,
        )
        await handler.create_message(
            db=self.db,
            role=sch.MessageRole.ASSISTANT,
            content="Searching documentation...",
            msg_type=enums.MessageType.THOUGHT,
        )
        await handler.send_api_message()
        ai_catalog = AICatalog()
        embed_model = ai_catalog.get_embedding_model(model_info=self.service_context.embedding_model_info)
        index_name = vector_utils.get_docs_index_name(client_id=self.client_id, db_uuid=self.db_uuid)
        vector_index = vector_utils.get_redis_index(
            index_name=index_name,
            embed_model=embed_model,
            redis_client_async=self.service_context.redis_client_async,
            redis_client=self.service_context.redis_client,
        )
        my_retriever = vector_index.as_retriever(similarity_top_k=TOP_K)
        nodes = await my_retriever.aretrieve(prompt)
        response_text = f"""\
Here is a list of relevant documentation snippets based on your prompt. Each snippet is separated using a \
{DELIMITER} delimiter. Here are the retrieved documentation snippets: \n\n"""
        context = [node.text for node in nodes if node.score > SCORE_THRESHOLD]  # type: ignore
        if not context:
            return "There was no documentation found to answer your specific prompt."
        return response_text + DELIMITER.join(context[:max_docs])
