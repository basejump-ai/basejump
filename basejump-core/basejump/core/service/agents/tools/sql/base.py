"""Configure the SQL tool"""

from sqlalchemy.ext.asyncio import AsyncSession

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.result import store
from basejump.core.models import schemas as sch
from basejump.core.service.agents.tools.base import BaseTool
from basejump.core.service.agents.tools.sql import retriever, runner
from basejump.core.service.base import BaseChatAgent

logger = set_logging(handler_option="stream", name=__name__)


class SQLTool(BaseTool):
    def __init__(
        self,
        db: AsyncSession,
        agent: BaseChatAgent,
        sql_tool_context: sch.SQLToolContext,
        db_conn_params: sch.SQLDBSchema,
        result_store: store.ResultStore,
        select_sample_values: bool = False,
    ):
        self.db = db
        self.agent = agent
        self.sql_tool_context = sql_tool_context
        self.db_conn_params = db_conn_params
        self.select_sample_values = select_sample_values
        self.result_store = result_store
        self.table_retriever_tool = retriever.TableRetrieverTool(
            db=self.db,
            agent=self.agent,
            sql_tool_context=self.sql_tool_context,
        )
        self.runner_tool = runner.SQLRunnerTool(
            db=self.db,
            agent=self.agent,
            sql_tool_context=self.sql_tool_context,
            result_store=self.result_store,
            db_conn_params=self.db_conn_params,
            select_sample_values=self.select_sample_values,
        )

    async def get_tools(self):
        runner_tools = await self.runner_tool.get_tools()
        retriever_tools = await self.table_retriever_tool.get_tools()
        return runner_tools + retriever_tools
