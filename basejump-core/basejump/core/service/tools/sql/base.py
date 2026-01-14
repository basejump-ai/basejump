"""Configure the SQL tool"""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.result import store
from basejump.core.models import enums
from basejump.core.models import schemas as sch
from basejump.core.service.base import BaseChatAgent
from basejump.core.service.tools import BaseTool
from basejump.core.service.tools.sql import retriever, runner

logger = set_logging(handler_option="stream", name=__name__)


class SQLTool(BaseTool):
    def __init__(
        self,
        agent: BaseChatAgent,
        db: AsyncSession,
        conn_id: int,
        conn_uuid: uuid.UUID,
        db_id: int,
        db_uuid: uuid.UUID,
        vector_id: int,
        prompt_metadata: sch.PromptMetadata,
        client_conn_params: sch.SQLDBSchema,
        db_conn_params: sch.SQLDBSchema,
        service_context: sch.ServiceContext,
        result_store: store.ResultStore,
        select_sample_values: bool = False,
        verbose: bool = False,
    ):
        self.agent = agent
        self.db = db
        self.conn_id = conn_id
        self.conn_uuid = conn_uuid
        self.db_id = db_id
        self.db_uuid = db_uuid
        self.prompt_metadata = prompt_metadata
        self.db_conn_params = db_conn_params
        self.client_conn_params = client_conn_params
        self.vector_id = vector_id
        self.sqlglot_dialect = enums.DB_TYPE_TO_SQLGLOT_DIALECT_LKUP.get(self.client_conn_params.database_type)
        self.sql_engine = service_context.sql_engine
        self.select_sample_values = select_sample_values
        self.result_store = result_store
        self.verbose = verbose
        self.schemas = self.client_conn_params.schemas or []

    async def get_tools(self):
        runner_tool = runner.SQLRunnerTool(
            db=self.db,
            prompt_metadata=self.prompt_metadata,
            service_context=self.service_context,
            result_store=self.result_store,
            agent=self.agent,
            sqlglot_dialect=self.sqlglot_dialect,
            client_conn_params=self.client_conn_params,
            conn_id=self.conn_id,
            schemas=self.schemas,
            select_sample_values=self.select_sample_values,
            verbose=self.verbose,
        )
        runner_tools = await runner_tool.get_tools()
        table_retriever_tools = retriever.TableRetrieverTool(
            db=self.db,
            service_context=self.service_context,
            agent=self.agent,
            conn_id=self.conn_id,
            vector_id=self.vector_id,
            prompt_metadata=self.prompt_metadata,
            db_uuid=self.db_uuid,
        )
        retriever_tools = await table_retriever_tools.get_tools()
        return runner_tools + retriever_tools
