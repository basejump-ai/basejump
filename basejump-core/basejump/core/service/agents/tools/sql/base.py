"""An agent tool for running SQL queries as well as other functions for managing the query results"""

from llama_index.core.llms.function_calling import FunctionCallingLLM
from sqlalchemy.ext.asyncio import AsyncSession

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.result import store
from basejump.core.models import schemas as sch
from basejump.core.service.agents.tools.base import BaseTool
from basejump.core.service.agents.tools.sql import retriever, runner

logger = set_logging(handler_option="stream", name=__name__)


class SQLTool(BaseTool):
    def __init__(
        self,
        db: AsyncSession,
        llm: FunctionCallingLLM,
        sql_tool_context: sch.SQLToolContext,
        db_conn_params: sch.SQLDBSchema,
        result_store: store.ResultStore,
        prompt_metadata: sch.PromptMetadata,
        chat_metadata: sch.ChatMetadata,
        query_result: sch.MessageQueryResult,
        select_sample_values: bool = False,
    ):
        # Set variables
        self.db = db
        self.llm = llm
        self.sql_tool_context = sql_tool_context
        self.db_conn_params = db_conn_params
        self.select_sample_values = select_sample_values
        self.result_store = result_store
        self.query_result = query_result

        # Create tools
        self.table_retriever_tool = retriever.TableRetrieverTool(
            db=self.db, llm=self.llm, sql_tool_context=self.sql_tool_context, prompt_metadata=prompt_metadata
        )
        self.runner_tool = runner.SQLRunnerTool(
            db=self.db,
            llm=self.llm,
            sql_tool_context=self.sql_tool_context,
            result_store=self.result_store,
            db_conn_params=self.db_conn_params,
            select_sample_values=self.select_sample_values,
            prompt_metadata=prompt_metadata,
            chat_metadata=chat_metadata,
            query_result=self.query_result,
        )

    async def get_tools(self):
        runner_tools = await self.runner_tool.get_tools()
        retriever_tools = await self.table_retriever_tool.get_tools()
        return runner_tools + retriever_tools
