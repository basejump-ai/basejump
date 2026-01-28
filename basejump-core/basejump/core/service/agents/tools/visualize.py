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

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.crud import crud_result
from basejump.core.database.db_utils import extract_visual_info
from basejump.core.models import constants, enums, errors, models
from basejump.core.models import schemas as sch
from basejump.core.models.ai import formats as fmt
from basejump.core.models.ai import formatter
from basejump.core.models.ai.catalog import AICatalog
from basejump.core.service.agents.base import BaseChatAgent
from basejump.core.service.agents.tools import tool_utils
from basejump.core.service.agents.tools.base import ResultTool

bucket_name = "datasetsfromchat"


logger = set_logging(handler_option="stream", name=__name__)
TIMEOUT = 60 * 3


class VisTool(ResultTool):
    def __init__(
        self,
        agent: BaseChatAgent,
        llm: Optional[LLM] = None,
    ):
        self.agent = agent

    async def get_tools(self) -> list[FunctionTool]:
        return [self.get_plot_tool()]

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
        embed_model = ai_catalog.get_embedding_model(model_info=self.agent.service_context.embedding_model_info)
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
            small_model_info=self.agent.service_context.small_model_info,
        )
        return await f.format()

    async def get_plot(self, result_uuid: uuid.UUID, prompt: str):
        await tool_utils.update_agent_tokens(agent=self.agent)
        # Get the result
        result = await crud_result.get_result_filtered(
            db=self.agent.db, result_uuid=result_uuid, user_uuid=self.agent.prompt_metadata.user_uuid
        )
        if not result:
            logger.error(errors.RESULT_UUID_NOT_FOUND)
            return f"""result_uuid {result_uuid} was not found. Unable to create a visualization since either the \
result_uuid is incorrect or the originally created data has been deleted."""

        # Retrieve the result
        try:
            result_manager = self.agent.result_store.get_result_manager(result_file_path=result.result_file_path)
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
        self.agent.db.add(visual_result_hist)
        await self.agent.db.commit()
        prompt = """
Either use another tool or complete your current line of thinking by responding to the user. \
If you decide to respond to the user, follow these instructions:
The visual result will be displayed to the user after your comment. \
Respond to the user letting them know about the visual. For example, if the user asked, "I want to see a bar chart" \
then you would respond "Here is the bar chart you requested." \
Do not mention anything about the results being displayed to the user. \
Talk as if you are showing them the chart in person."""
        return prompt

    async def refresh(
        self,
        result: models.VisualResultHistory,
    ) -> models.VisualResultHistory:
        """Refresh the visualization result"""
        # Create the prompt that includes the axis from the prior chart
        visual_info = extract_visual_info(visual_json=json.loads(result.visual_json))  # type: ignore
        # TODO: This was part of the logic for inferring the chart type to provide to the LLM to improve charting
        # Need to revisit this
        # match = re.search(r"type\s*=\s*(\w+)", visual_info)
        # if match:
        #     chart_type_base = match.group(1)
        #     try:
        #         chart_type_obj = sch.ChartType(chart_type=chart_type_base)
        #         chart_type = chart_type_obj.chart_type
        #     except Exception as e:
        #         logger.error(f"{chart_type_base} is not a valid chart type. Here is the error: {str(e)}")
        #     logger.info(f"Chart type: {chart_type}")
        # else:
        #     msg = "No chart type found"
        #     logger.error(msg)
        #     raise Exception(msg)
        prompt = f"""You are refreshing a plot you previously created. You need to use the same axis titles as \
    well as the same/similar axis ranges and/or format. Here is the visual information from the previous plot:
    {visual_info}
    """
        logger.debug("Refresh visual result prompt: %s", visual_info)
        # Query the VisTool
        result_uuid = result.result_uuid
        await self.get_plot(result_uuid=result_uuid, prompt=prompt)
        # Return the new visual result
        assert self.agent.query_result, "There should be a query result - check your code"
        assert self.agent.query_result.visual_result_uuid
        return await crud_result.get_visual_result(
            db=self.agent.db, visual_result_uuid=self.agent.query_result.visual_result_uuid
        )
