import asyncio
from functools import cached_property
from typing import Optional

import sqlalchemy as sa

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.connector import POOL_TIMEOUT, Connector
from basejump.core.database.manager import TableManager
from basejump.core.database.result import result_utils, store
from basejump.core.database.ssl import SSLEngine
from basejump.core.models import constants, errors
from basejump.core.models import schemas as sch

logger = set_logging(handler_option="stream", name=__name__)

TIMEOUT = 60 * 30


class ClientQueryRunner:
    def __init__(
        self,
        client_conn_params: sch.SQLDBSchema,
        sql_query: str,
    ):
        self.client_db = Connector.get_database_to_connect(conn_params=client_conn_params)
        self.client_conn_params = client_conn_params
        self._sql_query = sql_query
        self._client_engine: Optional[SSLEngine] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @cached_property
    def sql_query(self):
        """Render the SQL query if any jinja is included"""
        return TableManager.render_query_jinja(jinja_str=self._sql_query, schemas=self.client_conn_params.schemas)

    @property
    def client_engine(self) -> SSLEngine:
        """Lazily create and reuse the engine."""
        if self._client_engine is None:
            self._client_engine = self.client_db.connect_db()
        return self._client_engine

    async def close(self):
        """Call this when done with the manager."""
        if self._client_engine:
            self._client_engine.dispose()
            self._client_engine = None

    # NOTE: run_client_query needs to use a synchronous engine
    # since not all drivers support SQLAlchemy 2 or async drivers
    # TODO: Implement a timeout here
    def run_client_query(self) -> sch.QueryResultDF:
        """Run a SQL query against the client database."""
        try:
            logger.debug("Running client query: %s", self.sql_query)
            # NOTE: This needs to stay as connect so no DDL statements get committed
            with self.client_engine.connect() as client_db:
                try:
                    result = client_db.execute(sa.text(self.sql_query))
                except Exception as e:
                    client_db.rollback()
                    raise e
                query_result = result.all()
            query_result_df = result_utils.get_output_df(query_result=list(query_result), sql_query=self.sql_query)
            return query_result_df
        except Exception as e:
            # TODO: Improve the debugging
            if constants.SQLALCHEMY_TIMEOUT in str(e):
                error_msg = f"""Failed to connect to the database after {POOL_TIMEOUT/60} minutes. \
Connection timed out. Please try again."""
                raise sch.SQLTimeoutError(error_msg)
            logger.warning("Error running this SQL query: %s", self.sql_query)
            logger.warning("Here is the error: %s", str(e))
            raise errors.SQLRunError("Error running SQL query") from e
        logger.info("Completed running client query")
        return query_result_df

    async def arun_client_query(self) -> sch.QueryResultDF:
        """Function to run queries against the client database."""
        try:
            async with asyncio.timeout(TIMEOUT):
                return await asyncio.to_thread(self.run_client_query)
        except TimeoutError:
            error_msg = f"SQL query took longer to execute than the max {TIMEOUT/60} minute time out limit."
            logger.error(error_msg)
            raise sch.SQLTimeoutError(error_msg)


class ClientQueryRecorder(ClientQueryRunner):
    def __init__(
        self,
        client_id: int,
        initial_prompt: str,
        small_model_info: sch.ModelInfo,
        result_store: store.ResultStore,
        client_conn_params: sch.SQLDBSchema,
        sql_query: str,
    ):
        super().__init__(sql_query=sql_query, client_conn_params=client_conn_params)
        self.client_id = client_id
        self.initial_prompt = initial_prompt
        self.small_model_info = small_model_info
        self.result_store = result_store

    def store_query_result(self) -> sch.QueryResult:
        """Run a SQL query against a client database and store the results."""
        logger.debug("Running client query and storing results for SQL query: \n\n%s", self.sql_query)
        # TODO: Parse and parameterize this SQL query
        # NOTE: This needs to stay as connect so no DDL statements get committed

        with self.client_engine.connect() as client_db:
            try:
                with client_db.execute(sa.text(self.sql_query)) as result:
                    query_result = self.result_store.store(
                        result=result,
                        small_model_info=self.small_model_info,
                        initial_prompt=self.initial_prompt,
                        sql_query=self.sql_query,
                    )
            except Exception as e:
                logger.error("Error running client sql query and storing results: %s", str(e))
                client_db.rollback()
                raise e
        return query_result

    async def astore_query_result(self) -> sch.QueryResult:
        """Function to run queries against client databases.
        Needs to be synchronous queries since not all drivers
        support async"""
        return await asyncio.to_thread(self.store_query_result)
