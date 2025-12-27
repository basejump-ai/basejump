import asyncio
import uuid
from typing import Callable, Optional

import pandas as pd
import sqlalchemy as sa
from basejump.core.common.config.logconfig import set_logging
from basejump.core.database import upload
from basejump.core.database.db_connect import ConnectDB, TableManager
from basejump.core.models import schemas as sch
from sqlalchemy.engine import Engine, Row
from sqlglot import exp, parse_one

logger = set_logging(handler_option="stream", name=__name__)


def get_output_df(query_result: list[Row], sql_query: str) -> sch.QueryResultDF:
    # TODO: Have some handling in case this gets too big
    output_df = pd.DataFrame(query_result)
    result_row_ct = len(output_df)
    preview_row_ct = upload.RESULT_PREVIEW_CT if result_row_ct > upload.RESULT_PREVIEW_CT else result_row_ct
    preview_output_df = output_df.head(preview_row_ct)
    num_rows = output_df.shape[0]
    num_cols = output_df.shape[1]
    result_type = upload.get_result_type(num_rows=num_rows, num_cols=num_cols)
    return sch.QueryResultDF(
        output_df=output_df,
        query_result=query_result,
        preview_output_df=preview_output_df,
        preview_row_ct=preview_row_ct,
        num_rows=num_rows,
        num_cols=num_cols,
        result_type=result_type,
        sql_query=sql_query,
    )


class ClientQueryManager:
    def __init__(
        self,
        db_conn_params: sch.SQLDBSchema,
        client_conn_params: sch.SQLDBSchema,
        sql_query: str,
        result_uuid: Optional[uuid.UUID] = None,
    ):
        self.sql_query_base = sql_query
        self.db_conn_params = db_conn_params
        self.client_conn_params = client_conn_params
        self.result_uuid = result_uuid

    async def get_sql_query(self):
        logger.info("Getting SQL query for %s", self.sql_query_base)
        return await TableManager.arender_query_jinja(
            jinja_str=self.sql_query_base, schemas=self.client_conn_params.schemas
        )

    def quote_identifiers(self, sql: str, dialect: str) -> str:
        def _quote_identifiers(node):
            if isinstance(node, exp.Identifier):
                node.set("quoted", True)
            return node

        return parse_one(sql, dialect=dialect).transform(_quote_identifiers).sql(dialect=dialect)

    # NOTE: Could be made into a decorator
    async def _run_query_func(self, func: Callable, **kwargs) -> sch.QueryResultBase:
        conn_db = ConnectDB(conn_params=self.client_conn_params)
        client_engine = conn_db.connect_db()
        sql_query = await self.get_sql_query()
        logger.info("Running query: %s", sql_query)

        result = await asyncio.to_thread(
            func, client_engine=client_engine, sql_query=sql_query, result_uuid=self.result_uuid, **kwargs
        )
        client_engine.dispose()
        return result

    async def run_client_query_and_upload(
        self, initial_prompt: str, small_model_info: sch.ModelInfo, client_id: int
    ) -> sch.QueryResult:
        """Function to run queries against client databases.
        Needs to be synchronous queries since not all drivers
        support async"""
        return await self._run_query_func(
            run_client_query_sync_and_upload,
            db_conn_params=self.db_conn_params,
            initial_prompt=initial_prompt,
            client_id=client_id,
            small_model_info=small_model_info,
        )  # type: ignore

    async def run_client_query(self) -> sch.QueryResultDF:
        """Function to run queries against client databases.
        Needs to be synchronous queries since not all drivers
        support async"""
        return await self._run_query_func(run_client_query_sync)  # type: ignore


# NOTE: run_client_query_sync needs to use a synchronous engine
# since not all drivers support SQLAlchemy 2 or async drivers
def run_client_query_sync(
    client_engine: Engine, sql_query: str, result_uuid: Optional[str] = None
) -> sch.QueryResultDF:
    if result_uuid:
        raise NotImplementedError(
            """This function does not require a result_uuid - \
you were likely trying to use run_client_query_sync_and_upload instead"""
        )
    # NOTE: This needs to stay as connect so no DDL statements get committed
    with client_engine.connect() as client_db:
        try:
            result = client_db.execute(sa.text(sql_query))
        except Exception as e:
            client_db.rollback()
            raise e
        query_result = result.all()
    query_result_df = get_output_df(query_result=list(query_result), sql_query=sql_query)
    return query_result_df


def run_client_query_sync_and_upload(
    db_conn_params: sch.SQLDBSchema,
    client_engine: Engine,
    sql_query: str,
    initial_prompt: str,
    small_model_info: sch.ModelInfo,
    client_id: int,
    result_uuid: Optional[uuid.UUID] = None,
) -> sch.QueryResult:
    # TODO: Parse and parameterize this SQL query
    # NOTE: This needs to stay as connect so no DDL statements get committed
    with client_engine.connect() as client_db:
        try:
            query_result = upload.upload_sql_to_s3(
                db_conn_params=db_conn_params,
                conn=client_db,
                sql_query=sql_query,
                result_uuid=result_uuid,
                initial_prompt=initial_prompt,
                client_id=client_id,
                small_model_info=small_model_info,
            )
        except Exception as e:
            logger.error("Error in run_client_query_sync_and_upload %s", str(e))
            client_db.rollback()
            raise e
    return query_result
