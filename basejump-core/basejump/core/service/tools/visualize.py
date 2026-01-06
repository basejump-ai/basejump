import json
import uuid
from typing import Optional

import pandas as pd
from chat2plot import chat2plot as cp
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.llms import LLM
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.tools import FunctionTool
from llama_index.core.tools.function_tool import create_tool_metadata
from sqlalchemy.ext.asyncio import AsyncSession

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.crud import crud_result
from basejump.core.database.result import store
from basejump.core.models import constants, enums, errors, models
from basejump.core.models import schemas as sch
from basejump.core.models.ai import formats as fmt
from basejump.core.models.ai import formatter
from basejump.core.models.ai.catalog import AICatalog
from basejump.core.service.base import BaseAgent, BaseChatAgent
from basejump.core.service.tools import tool_utils

bucket_name = "datasetsfromchat"


logger = set_logging(handler_option="stream", name=__name__)
TIMEOUT = 60 * 3


class VisTool:
    def __init__(
        self,
        db: AsyncSession,
        agent,
        small_model_info: sch.ModelInfo,
        embedding_model_info: sch.AzureModelInfo,
        result_store: store.ResultStore,
        llm: Optional[LLM] = None,
    ):
        self.db = db
        self.agent: BaseAgent = agent
        self.small_model_info = small_model_info
        self.embedding_model_info = embedding_model_info
        self.result_store = result_store

    def get_plot_tool(self) -> FunctionTool:
        func = self.get_plot
        tool_metadata = create_tool_metadata(
            fn=func,
            name=constants.VIS_TOOL_NM,
            description="""This tool returns a visualization of the data that can be \
shown to the user to provide more insight into their data.""",
        )
        plot_tool = FunctionTool.from_defaults(fn=func, async_fn=func, tool_metadata=tool_metadata)
        return plot_tool

    async def select_date_cols(self, cols: list[str]) -> list[str]:
        date_cols = []

        documents = [
            Document(text="date"),
            Document(text="time"),
            Document(text="month"),
            Document(text="year"),
            Document(text="week"),
            Document(text="quarter"),
            Document(text="yearmo"),
        ]

        # Build index
        # TODO: Add a callback manager to track token usage
        ai_catalog = AICatalog()
        embed_model = ai_catalog.get_embedding_model(model_info=self.embedding_model_info)
        index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

        # Configure retriever
        retriever = VectorIndexRetriever(index=index, similarity_top_k=1)  # Set to 1 to get the most similar result

        # Perform similarity search
        for col in cols:
            nodes = await retriever.aretrieve(col)
            # Get similarity score
            similarity_score = nodes[0].score
            logger.debug(f"Cosine similarity: {similarity_score}")
            if similarity_score > 0.6:  # type: ignore
                date_cols.append(col)
        return date_cols

    async def format_date(self, cols) -> pd.DataFrame:
        date_prompt = f"""
        dates:{cols}\n"""
        f = formatter.DateFormatter(
            response=date_prompt,
            pydantic_format=fmt.DateData,
            small_model_info=self.small_model_info,
        )
        return await f.format()

    async def get_plot(self, result_uuid: uuid.UUID, prompt: str):
        await tool_utils.update_agent_tokens(agent=self.agent)
        # Get the result
        result = await crud_result.get_result_filtered(
            db=self.db, result_uuid=result_uuid, user_uuid=self.agent.prompt_metadata.user_uuid
        )
        if not result:
            logger.error(errors.RESULT_UUID_NOT_FOUND)
            return f"""result_uuid {result_uuid} was not found. Unable to create a visualization since either the \
result_uuid is incorrect or the originally created data has been deleted."""

        # Retrieve the result
        try:
            result_manager = self.result_store.get_result_manager(result_file_path=result.result_file_path)
            df = await result_manager.aget_result(max_file_size=5)
        except errors.FileSizeError:
            return """File size is larger than 5 MB. Make sure to aggregate the data using SQL \
before attempting to visualize."""
        dates = await self.select_date_cols(df.columns.to_list())
        if dates:
            formatted = await self.format_date(cols=df[dates])
            df[dates] = pd.DataFrame(formatted.dates)
        c2p = cp(df, chat=self.agent.agent_llm)
        visual = c2p(prompt)
        # Save and send back to the user
        # TODO: Sometimes visual is None
        # Add some error handling for this
        visual_json = visual.figure.to_json()
        visual_result_uuid = uuid.uuid4()
        if not self.agent.query_result:
            self.agent.query_result = sch.MessageQueryResult()
        self.agent.query_result.visual_result_uuid = visual_result_uuid
        self.agent.query_result.visual_json = json.loads(visual_json)
        self.agent.query_result.visual_explanation = visual.explanation
        if not self.agent.query_result.result_uuid:
            self.agent.query_result.result_uuid = result.result_uuid
            self.agent.query_result.sql_query = result.sql_query
            self.agent.query_result.result_type = enums.ResultType(result.result_type)
        # Create VisualResultHistory table
        visual_result_hist = models.VisualResultHistory(
            client_id=result.client_id,
            visual_result_uuid=visual_result_uuid,
            parent_msg_uuid=(
                self.agent.chat_metadata.parent_msg_uuid if isinstance(self.agent, BaseChatAgent) else None
            ),
            result_id=result.result_id,
            result_uuid=result.result_uuid,
            visual_json=visual_json,
            visual_explanation=visual.explanation,
        )
        self.db.add(visual_result_hist)
        await self.db.commit()
        prompt = """
Either use another tool or complete your current line of thinking by responding to the user. \
If you decide to respond to the user, follow these instructions:
The visual result will be displayed to the user after your comment. \
Respond to the user letting them know about the visual. For example, if the user asked, "I want to see a bar chart" \
then you would respond "Here is the bar chart you requested." \
Do not mention anything about the results being displayed to the user. \
Talk as if you are showing them the chart in person."""
        return prompt
