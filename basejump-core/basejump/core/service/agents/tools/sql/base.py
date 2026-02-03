"""An agent tool for running SQL queries as well as other functions for managing the query results"""

from datetime import datetime
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.client import query
from basejump.core.database.connector import Connector
from basejump.core.database.crud import crud_chat, crud_connection
from basejump.core.database.result import store
from basejump.core.models import models
from basejump.core.models import schemas as sch
from basejump.core.service.agents.chat import ChatAgent
from basejump.core.service.agents.tools.base import ResultTool
from basejump.core.service.agents.tools.sql import retriever, runner

logger = set_logging(handler_option="stream", name=__name__)


class SQLTool(ResultTool):
    def __init__(
        self,
        db: AsyncSession,
        agent: ChatAgent,
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

    async def refresh(
        self,
        result: models.ResultHistory,
        # HACK: Fix this
        commit: bool = True,
    ) -> Optional[models.ResultHistory]:
        db_conn = await crud_connection.get_db_conn_from_id(db=self.db, conn_id=result.result_conn_id)
        if not db_conn:
            logger.warning("Missing db conn")
            return None
        db_params = await db_conn.awaitable_attrs.database_params
        conn_db = await Connector.get_db_conn(db_conn=db_conn, db_params=db_params)
        # Get the initial prompt
        initial_prompt = await crud_chat.get_initial_prompt_for_result(db=self.db, result_uuid=result.result_uuid)
        assert initial_prompt, "Missing chat history"
        self.result_store.result_uuid = result.result_uuid
        async with query.ClientQueryRecorder(
            client_conn_params=conn_db.conn_params,
            sql_query=result.sql_query,
            initial_prompt=initial_prompt,
            client_id=self.agent.prompt_metadata.client_id,
            small_model_info=self.agent.service_context.small_model_info,
            result_store=self.result_store,
        ) as query_recorder:
            query_result = await query_recorder.astore_query_result()
        # Update record
        # TODO: Update this to use schemas instead
        # HACK: Fix this
        result.refresh_result = False
        result.row_num_preview = query_result.preview_row_ct
        result.row_num_total = query_result.num_rows
        result.result_type = query_result.result_type.value
        result.result_exp_time = query_result.result_exp_time
        result.aborted_upload = query_result.aborted_upload
        result.metric_value = query_result.metric_value
        result.metric_value_formatted = query_result.metric_value_formatted
        result.result_file_path = query_result.result_file_path
        result.preview_file_path = query_result.preview_file_path
        result.timestamp = datetime.now()
        if commit:
            await self.db.commit()
            await self.db.refresh(result)
            return result
        return None
