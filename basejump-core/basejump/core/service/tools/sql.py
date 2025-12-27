"""Configure the SQL tool"""

import asyncio
import copy
import re
import uuid
from typing import Optional

import redis
from basejump.core.common.config.logconfig import set_logging
from basejump.core.database import db_utils, query
from basejump.core.database.aicatalog import AICatalog
from basejump.core.database.crud import crud_connection, crud_table
from basejump.core.database.db_connect import POOL_TIMEOUT, TableManager
from basejump.core.database.format_response import JSONResponseFormatter
from basejump.core.database.vector_utils import get_vector_idx
from basejump.core.models import constants, enums, errors
from basejump.core.models import pydantic_ai_formats as fmt
from basejump.core.models import schemas as sch
from basejump.core.models.prompts import DB_METADATA_PROMPT, ZERO_ROW_PROMPT
from basejump.core.service import service_utils
from basejump.core.service.base import BaseChatAgent, ChatMessageHandler
from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.indices.struct_store.sql_retriever import SQLTableRetriever
from llama_index.core.objects import SQLTableNodeMapping, base
from llama_index.core.schema import QueryBundle
from llama_index.core.tools import FunctionTool
from llama_index.core.tools.function_tool import create_tool_metadata
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.vector_stores.redis.base import NO_DOCS
from redis.asyncio import Redis as RedisAsync
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlglot import errors as sqlglot_errors
from sqlglot import exp, parse_one
from sqlglot.dialects.dialect import Dialects

logger = set_logging(handler_option="stream", name=__name__)
TIMEOUT = 60 * 15
RELEVANCE_THRESHOLD = 0.1
STUCK_IN_LOOP_MAX_CT = 3


class SQLTool:
    TABLES_TO_RETRIEVE: int = 12
    MAX_SQL_ITER = 5

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
        redis_client_async: RedisAsync,
        sql_engine: AsyncEngine,
        large_model_info: sch.ModelInfo,
        small_model_info: sch.ModelInfo,
        embedding_model_info: sch.AzureModelInfo,
    ):
        self.agent = agent
        self.db = db
        self.conn_id = conn_id
        self.conn_uuid = conn_uuid
        self.db_id = db_id
        self.db_uuid = db_uuid
        self.prompt_metadata = prompt_metadata
        self.confirm_tool_retrieval = False
        self.db_conn_params = db_conn_params
        self.client_conn_params = client_conn_params
        self.vector_id = vector_id
        self.tools: list[FunctionTool] = []
        self.is_demo = False
        self.sql_query_created = False
        self.sqlglot_dialect = enums.DB_TYPE_TO_SQLGLOT_DIALECT_LKUP.get(self.client_conn_params.database_type)
        self.prior_sql_query: Optional[str] = None
        self.db_columns: list = []
        self.col_check_ct = 0
        self.provided_sample_vals = False
        self.db_cols: list = []
        self.large_model_info = large_model_info
        self.small_model_info = small_model_info
        self.embedding_model_info = embedding_model_info
        self.sql_engine = sql_engine
        self.redis_client_async = redis_client_async
        self.stuck_in_loop_ct = 0

    async def post_init(self):
        loaded_sql_tool = await self._get_sql_tables_tool()
        self.tools.append(loaded_sql_tool)
        self.tools.append(self._sql_execution_tool())

    # TODO: This would change to 'get sql' once we have a SQL specific model and
    # would take no input args
    async def _get_sql_tables_tool(self) -> FunctionTool:
        # SQL Table Vector Index setup
        vector_conn = await crud_connection.get_vector_connection_from_id(db=self.db, vector_id=self.vector_id)
        self.vector_uuid = copy.copy(vector_conn.vector_uuid)
        self.index_name = str(copy.copy(vector_conn.index_name))
        # Check if the table is a demo table
        demo_tbl_info = await crud_connection.get_demo_tbl_info(db=self.db, vector_id=self.vector_id)
        if demo_tbl_info:
            vector_db_uuid = demo_tbl_info.demo_db_uuid
            vector_client_id = str(demo_tbl_info.demo_client_id)
            vector_client_uuid = demo_tbl_info.demo_client_uuid
            self.is_demo = True
        else:
            vector_db_uuid = self.db_uuid
            vector_client_id = str(self.prompt_metadata.client_id)
            vector_client_uuid = self.prompt_metadata.client_uuid
        logger.debug(
            f"""Using the following for vector indexes:
vector_client_id: {vector_client_id}
vector_client_uuid: {str(vector_client_uuid)}
vector_db_uuid: {str(vector_db_uuid)}
        """
        )
        self.table_index = await self.setup_sql_table_vector_index(
            vector_id=self.vector_id, client_id=int(vector_client_id)
        )
        self.schemas = self.client_conn_params.schemas or []
        all_tables = await crud_table.get_all_tables(db=self.db)
        self.all_tables: list = []
        self.ignored_tables = []
        for tbl in all_tables:
            self.all_tables.append(
                await TableManager.arender_query_jinja(jinja_str=tbl.table_name, schemas=self.schemas)
            )
            if tbl.ignore:
                self.ignored_tables.append(
                    await TableManager.arender_query_jinja(jinja_str=tbl.table_name, schemas=self.schemas)
                )

        db_cols = await crud_table.get_all_columns(db=self.db, conn_id=self.conn_id)
        for col in db_cols:
            table_name = await TableManager.arender_query_jinja(jinja_str=col.table_name, schemas=self.schemas)
            col_obj = sch.DBColumn(
                column_name=col.column_name,
                table_name=db_utils.get_table_name(table_name=table_name),
                schema_name=db_utils.get_table_schema(table_name=table_name),
                quoted=col.quoted,
            )
            self.ignored_cols = []
            if col.ignore:
                self.ignored_cols.append(col_obj)
            self.db_cols.append(col_obj)
        self.filters = await self.get_table_metadata_filters(
            conn_id=self.conn_id, db_uuid=vector_db_uuid, client_uuid=vector_client_uuid
        )
        # Setup the SQL Retriever
        self.sql_retriever = self.setup_sql_retriever(top_k=self.TABLES_TO_RETRIEVE)
        self.sub_prompt_sql_retriever = self.setup_sql_retriever(top_k=self.TABLES_TO_RETRIEVE)
        # TODO: See if I need varying names for different databases
        func = self.get_sql_tables
        name = constants.get_sql_tables_tool_nm(conn_id=self.conn_id)
        assert func.__name__ in name
        tool_metadata = create_tool_metadata(
            fn=func,
            name=name,
            description="""This tool returns a list of database tables that are relevant \
to your prompt that can be used in SQL queries. \
Here is a description of the SQL database connection: """
            + self.client_conn_params.data_source_desc,
        )
        sql_tool = FunctionTool.from_defaults(fn=func, async_fn=func, tool_metadata=tool_metadata)
        await self.db.commit()  # NOTE: Closing transaction to avoid idle in transaction
        return sql_tool

    def _sql_execution_tool(self) -> FunctionTool:
        func = self.run_sql

        name = constants.get_sql_execution_tool_nm(conn_id=self.conn_id)
        assert func.__name__ in name
        tool_metadata = create_tool_metadata(
            fn=func,
            name=name,
            description="Run this function to execute a SQL query",
        )
        sql_exec_tool = FunctionTool.from_defaults(fn=func, async_fn=func, tool_metadata=tool_metadata)

        return sql_exec_tool

    async def setup_sql_table_vector_index(self, vector_id: int, client_id: int) -> VectorStoreIndex:
        """Load the vector index"""
        # Get the vector DB
        vector_db = await crud_connection.get_vector_connection_from_id(db=self.db, vector_id=vector_id)
        # Initialize the environment
        vector_schema = sch.VectorDBSchema.model_validate(vector_db)
        ai_catalog = AICatalog()
        settings = ai_catalog.get_settings(llm=self.agent.agent_llm, embedding_model_info=self.embedding_model_info)
        table_index = get_vector_idx(
            client_id=client_id,
            vector_schema=vector_schema,
            settings=settings,
            redis_client_async=self.redis_client_async,
        )

        return table_index

    async def check_all_tables(self, sql_query: str) -> Optional[str]:
        try:
            parsed_query = parse_one(sql_query, dialect=self.sqlglot_dialect)
            parsed_query_tbls = parsed_query.find_all(exp.Table)
            cte_tbls = parsed_query.find_all(exp.CTE)
            # Get the schema + table name
            cleaned_tbl_names = []
            for tbl in parsed_query_tbls:
                if tbl.db:
                    cleaned_tbl_names.append(f"{tbl.db}.{tbl.name}")
                else:
                    cleaned_tbl_names.append(tbl.name)
            query_tbls_no_cte = set(cleaned_tbl_names) - {tbl.alias for tbl in cte_tbls}
            query_tbls_lowered = {table.lower() for table in query_tbls_no_cte}
            all_tables_lowered = {table.lower() for table in self.all_tables}
            # Find the ignored tables
            ignored_tables_lowered = {table.lower() for table in self.ignored_tables}
            tbl_overlap = ignored_tables_lowered & query_tbls_lowered
            # Check for hallucinated tables
            if not query_tbls_lowered.issubset(all_tables_lowered) or tbl_overlap:
                ai_msg = f'The following tables do not exist: {", ".join(query_tbls_lowered-all_tables_lowered)}'
                logger.info(ai_msg)
                return ai_msg
            # logger.debug("Here are the tables from the sql query: %s", query_tbls)
            # logger.debug("Here are the tables from the ignored tables: %s", self.ignored_tbls)
        except Exception as e:
            logger.warning("SQLglot failed parsing: %s", str(e))
            logger.traceback()
            return None
        else:
            return None

    async def check_ignored_columns(self, sql_query: str) -> Optional[str]:
        try:
            star_exists_msg = db_utils.check_for_star(sql_query=sql_query, dialect=self.sqlglot_dialect)
            if star_exists_msg:
                return star_exists_msg
            query_cols_base = db_utils.get_fully_qualified_col_names(sql_query=sql_query, dialect=self.sqlglot_dialect)
            query_cols = {db_utils.get_column_str(column) for column in query_cols_base}
            # logger.debug("Here are the columns from the sql query: %s", query_cols)
            # logger.debug("Here are the columns from the ignored columns: %s", self.ignored_cols)
            ignored_cols = {db_utils.get_column_str(column) for column in self.ignored_cols}
            col_overlap = set(ignored_cols) & query_cols
            if col_overlap:
                ai_msg = f'You do not have access to query the following columns: {", ".join(col_overlap)}'
                logger.info(ai_msg)
                return ai_msg
        except (errors.SQLParseError, sqlglot_errors.ParseError) as e:
            logger.warning("SQLglot failed parsing: %s", str(e))
            return None
        else:
            return None

    async def check_all_columns(self, columns: list[sch.DBColumn]):
        """Check columns for hallucinations and capitalization errors"""
        try:
            db_cols = {db_utils.get_column_str(column) for column in self.db_cols}
            ignored_cols = {db_utils.get_column_str(column) for column in self.ignored_cols}
            valid_cols = db_cols - ignored_cols
            valid_cols_lowered = {column.lower() for column in valid_cols}
            query_cols = {db_utils.get_column_str(column) for column in columns}
            query_cols_lowered = {column.lower() for column in query_cols}
            # logger.warning("Here are the valid columns: %s", valid_cols)
            # logger.warning("Here are the query columns: %s", query_cols)
            # logger.warning("Here are the all columns: %s", self.db_cols)
            if not query_cols_lowered.issubset(valid_cols_lowered):
                ai_msg = f'The following column does not exist in the \
table. Do not use these column(s): {", ".join(query_cols_lowered-valid_cols_lowered)}'
                logger.info(ai_msg)
                raise errors.HallucinatedColumnError(ai_msg)
            elif not query_cols.issubset(valid_cols):
                # TODO: does not need all valid cols, just need ones that are miscapitalized in the query
                ai_msg = f'The following column(s) do exist in the schema but you have the capitalization wrong:\
                       {", ".join(query_cols-valid_cols)}. Try using one of these instead: {", ".join(valid_cols)}'
                logger.info(ai_msg)
                raise errors.ColumnCapitalizationError(ai_msg)
        except (errors.SQLParseError, sqlglot_errors.ParseError) as e:
            logger.warning("SQLglot failed parsing: %s", str(e))
            return None
        else:
            return None

    def quote_case_sensitive_cols(self, sql_query: str, columns: list[sch.DBColumn]):
        """Quote case sensitive columns in the SQL query"""
        # logger.info("here are the columns to quote: %s", columns)
        table_aliases = {}
        table_dbs = {}
        logger.info("Qualifying names in quote_case_sensitive_cols")
        try:
            qualified_ast = db_utils.qualify_names(sql_query, dialect=self.sqlglot_dialect)
            logger.info("Unquoting identifiers")
            sql_query_unquoted = db_utils.unquote_identifiers(
                qualified_ast.sql(dialect=self.sqlglot_dialect), dialect=self.sqlglot_dialect
            )
            logger.info("Completed unquote")
        except Exception as e:
            logger.warning("Here is the error: %s", str(e))
            raise e
        logger.info("Here is the SQL query unquoted: %s", sql_query_unquoted)

        for node in parse_one(sql_query_unquoted).find_all(exp.Table):
            # If the table has an alias, store it
            if node.alias:
                table_aliases[node.alias] = node.name
            if node.db:
                table_dbs[node.name] = node.db

        def _quote_identifiers(node):
            try:
                if isinstance(node, exp.Column):
                    for col in columns:
                        if (
                            col.column_name == node.name
                            and (col.table_name == node.table or col.table_name == table_aliases[node.table])
                            if node.table
                            else None
                        ):
                            if col.schema_name:
                                try:
                                    if col.schema_name != table_dbs[table_aliases[node.table]]:
                                        continue
                                except Exception:
                                    logger.debug(
                                        f"Skipping quoting for column {col.column_name} and schema {col.schema_name}"
                                    )
                                    continue
                                # Check if the column is case sensitive
                            if col.quoted:
                                # Quote the identr
                                node.this.set("quoted", True)
                                return node
            except Exception as e:
                logger.error("There was a quote error: %s", str(e))
                raise e
            return node

        parsed_ast = parse_one(sql_query_unquoted)
        transformed_ast = parsed_ast.transform(_quote_identifiers)
        sql_str = transformed_ast.sql(dialect=self.sqlglot_dialect)
        return sql_str

    async def validate_all_columns(self, sql_query: str) -> str:
        """Validate all columns in the SQL query for capitalization errors and hallucinations and quote if needed"""
        logger.info("Validating all columns in the SQL query: %s", sql_query)
        try:
            columns = db_utils.get_fully_qualified_col_names(sql_query=sql_query, dialect=self.sqlglot_dialect)
            await self.check_all_columns(columns=columns)
            logger.info("All cols checked")
            try:
                sql_query = self.quote_case_sensitive_cols(sql_query=sql_query, columns=self.db_cols)
            except Exception as e:
                logger.warning(str(e))
                logger.traceback()
            logger.info("Here is the SQL query after quoting: %s", sql_query)
            return sql_query
        except (errors.StarQueryError, errors.ColumnCapitalizationError, errors.HallucinatedColumnError) as e:
            logger.warning("Error in validating columns: %s", str(e))
            raise e

    def check_strict_mode(self):
        user_role = enums.USER_ROLES_LVL_LKUP[self.prompt_metadata.user_role]
        admin_role = enums.USER_ROLES_LVL_LKUP[enums.UserRoles.ADMIN.value]
        if self.agent.chat_metadata.verify_mode == enums.VerifyMode.STRICT and user_role < admin_role:
            raise errors.StrictModeFlagged

    async def check_query_where_clause(self, query1: str, query2: str) -> None:
        """If there is a semantically cached response, this function checks if that the difference \
between that semantically similar query and the new SQL query is only the WHERE clause. If it is, \
then it can still be considered verified.
        """
        comparison = db_utils.compare_sql_queries_no_where_clause(
            query1=query1, query2=query2, dialect=self.sqlglot_dialect
        )
        if comparison == enums.SQLSimilarityLabel.IDENTICAL:
            logger.info("Found verified similar SQL Query from semantic cache")
        else:
            # If it's not identical after checking the WHERE clause, then it is not verified
            assert self.agent.chat_metadata.semcache_response
            self.agent.chat_metadata.semcache_response.verified = False
            self.check_strict_mode()

    async def get_where_clause_columns(self, sql_query: str) -> Optional[list[sch.DBColumn]]:
        try:
            ast = parse_one(sql_query, dialect=self.sqlglot_dialect)
            where_clause_exists = ast.find(exp.Where)
        except Exception as e:
            logger.warning("SQLglot failed parsing for where clause example values: %s", str(e))
            return None
        if not where_clause_exists:
            return None
        # Get the where clause columns
        try:
            columns = db_utils.get_fully_qualified_col_names(
                sql_query=sql_query, dialect=self.sqlglot_dialect, ancestor_to_filter=exp.Where
            )
        except (errors.SQLParseError, sqlglot_errors.ParseError) as e:
            logger.warning("SQLglot failed parsing for where clause example values: %s", str(e))
            return None
        return columns

    async def extend_db_columns(self, columns: list[sch.DBColumn]) -> None:
        # Check for any columns that already have been retrieved from the DB
        logger.warning("Extending DB Cols")
        columns_to_retrieve = []
        for column in columns:
            match = False
            column_str = db_utils.get_column_str(column=column)
            for db_column in self.db_columns:
                db_column_str = db_utils.get_column_str(column=db_column)
                if column_str == db_column_str:  # If the columns are the same
                    match = True
                    break
            if not match:
                # If not already retrieved, then append
                columns_to_retrieve.append(column)
        if columns_to_retrieve:
            new_db_columns = await crud_table.get_columns_by_name(
                db=self.db, columns=columns_to_retrieve, conn_id=self.conn_id, schemas=self.schemas
            )
            self.db_columns.extend(new_db_columns)

    async def get_db_column_filters(self, column: sch.DBColumn, db_column: sch.DBColumn):
        # Do a fuzzy match to find similar values
        tbl_name = db_utils.get_table_name_from_column(column=column)
        assert column.column_w_func, "This should be populated. Check your code and fix."
        # Get distinct values
        ast = exp.select(column.column_w_func).from_(tbl_name)
        # Loop through filters and create a like
        if "lower(" in column.column_w_func:  # avoiding using lower twice
            col_name = exp.column(column.column_w_func)
        else:
            col_name = exp.func(
                "lower", exp.column(column.column_w_func), dialect=self.sqlglot_dialect
            )  # type: ignore
        ast_filters = [col_name.like(db_utils.fuzzify_filter_value(value=filter_)) for filter_ in column.filters]
        # Use an OR if filters is over 1 since that indicates an IN operator was used
        if len(column.filters) > 1:
            filter_condition = exp.or_(*ast_filters, dialect=self.sqlglot_dialect)
            ast = ast.where(filter_condition, dialect=self.sqlglot_dialect)
        else:
            ast = ast.where(ast_filters[0], dialect=self.sqlglot_dialect)
        # Run the SQL query
        fuzzy_sql_base = ast.sql(dialect=self.sqlglot_dialect)
        logger.info("Running fuzzy sql base %s", fuzzy_sql_base)
        # HACK: SQLGlot isn't transpiling correctly, so doing it manually
        if self.sqlglot_dialect == Dialects.TSQL.value:
            # TODO: This performs distinct and then limits, need a subquery to limit first
            fuzzy_sql = "SELECT DISTINCT TOP 100000" + fuzzy_sql_base.lower().split("select")[1]
        else:
            # TODO: this does not limit as intended will need to be fixed
            fuzzy_sql = (
                "SELECT DISTINCT " + re.split("select", fuzzy_sql_base, flags=re.IGNORECASE)[1] + " LIMIT 100000"
            )
        logger.info("Running fuzzy sql %s", fuzzy_sql)
        query_result = await self.run_client_query(sql_query=fuzzy_sql)
        if query_result.output_df.empty:
            logger.warning("The fuzzy sql returned no results running distinct without filter")
            sql = "SELECT DISTINCT " + re.split("select", fuzzy_sql_base, flags=re.IGNORECASE)[1]
            distinct_ast = parse_one(sql, dialect=self.sqlglot_dialect)
            distinct_ast.args["where"] = None
            logger.warning("Here is the AST select: %s", distinct_ast)
            sql = distinct_ast.sql(dialect=self.sqlglot_dialect) + " LIMIT 100000"
            logger.info("Running unfuzzy sql %s", sql)
            query_result = await self.run_client_query(sql_query=sql)
        # Add results to db_column.filters
        db_column.filters = db_utils.get_query_column_values(query_result=query_result)

    def compare_column_filters(self, llm_feedback: str, column: sch.DBColumn, db_column: sch.DBColumn):
        # Compare the filters - verify that it choose one of the columns in the table or used fuzzy match
        db_filters_ct = len(db_column.filters)
        logger.info("Here are the number of filters: %s", db_filters_ct)
        if len(db_column.filters) == 0:
            column_str = db_utils.get_column_str(column=column)
            return (
                f"""- {column_str}: The filter value used for this column did not match any values in the database. """
                + ZERO_ROW_PROMPT
            )
        if db_filters_ct > 15:
            logger.info("DB Filters > 15")
            # If more than 15 choices, then allow the LLM to fuzzy match,
            # otherwise require an exact match
            try:
                for filter_ in column.filters:
                    logger.info("Verifying this filter: %s", filter_)
                    attempted_fuzzy_match = False
                    if "%" in filter_:
                        attempted_fuzzy_match = True
                    match = False
                    for db_filter in db_column.filters:
                        # Replace any % with a .* greedy search
                        regexed_filter = filter_.replace("%", ".*")
                        if re.fullmatch(regexed_filter, db_filter):
                            match = True
                    assert match
            except AssertionError:
                column_str = db_utils.get_column_str(column=column)
                # TODO: Clean up, not very DRY
                if attempted_fuzzy_match:
                    if db_filters_ct <= 100:
                        db_col_filters = ", ".join([f"'{str(val)}'" for val in db_column.filters])
                        llm_feedback += f"""- {column_str}: The fuzzy filter value used for this column \
in the WHERE clause did not match any values in the database. Here are the available values in the database, please \
update your filter value to match one or multiple of these instead: {db_col_filters}\n"""
                    else:
                        sample_ct = 50
                        db_col_filters = ", ".join([f"'{str(val)}'" for val in db_column.filters[:sample_ct]])
                        llm_feedback += f"""- {column_str}: The fuzzy filter value used for this column\
in the WHERE clause did not match any values in the database. Here is a sample of the available \
values in the database. Please update your filter using the samples as reference for the correct \
format: {db_col_filters}\n"""
                else:
                    if db_filters_ct <= 100:
                        db_col_filters = ", ".join([f"'{str(val)}'" for val in db_column.filters])
                        llm_feedback += f"""- {column_str}: The filter value used for this column \
in the WHERE clause did not match any values in the database. Here are the available values in the database, please \
update your filter value to match one or multiple of these instead: {db_col_filters}\n"""
                    else:
                        sample_ct = 50
                        db_col_filters = ", ".join([f"'{str(val)}'" for val in db_column.filters[:sample_ct]])
                        llm_feedback += f"""- {column_str}: The filter value used for this column in the WHERE \
clause did not exactly match any values in the database. Here is a sample of the available values in the database. \
Please update your filter to either use an exact or fuzzy match using the samples as reference for the correct \
format: {db_col_filters}\n"""
        else:
            logger.info("DB Filters < 15")
            # Values must match exactly
            try:
                for filter_ in column.filters:
                    assert filter_ in db_column.filters
            except AssertionError:
                logger.error("The column that failed was %s", filter_)
                logger.error("Here are the column filters %s", column.filters)
                logger.error("Here are the DB column filters %s", db_column.filters)
                column_str = db_utils.get_column_str(column=column)
                db_col_filters = ", ".join([f"'{str(val)}'" for val in db_column.filters])
                llm_feedback += f"""- {column_str}: The filter value used for this column in the WHERE \
clause did not exactly match any values in the database. Here are the available values in the \
database, please update your filter value to one or multiple of these instead: {db_col_filters}\n"""
        return llm_feedback

    async def verify_column_filters(self, columns: list[sch.DBColumn]) -> Optional[str]:
        # Retrieve the columns from the database
        llm_feedback = ""
        if self.db_columns:
            await self.extend_db_columns(columns=columns)
        else:
            self.db_columns = await crud_table.get_columns_by_name(
                db=self.db, columns=columns, conn_id=self.conn_id, schemas=self.schemas
            )
            if not self.db_columns:
                col_names = ", ".join([column.column_name for column in columns])
                logger.warning("Matching columns not found for these columns: %s", col_names)
                raise errors.UnverifiedColumns("Matching columns not found")
        # Compare every column and its filters to the db columns
        logger.debug("Checking the following columns: %s", columns)
        logger.debug("Here are the DB Columns being compared against columns: %s", self.db_columns)
        db_cols_str = [db_utils.get_column_str(column=db_column) for db_column in self.db_columns]
        for column in columns:
            found_db_match = False
            skipped = False
            column_str = db_utils.get_column_str(column=column)
            if not column.filters:
                logger.warning("Skipping column since it has no filters to verify: %s", column.column_name)
                logger.warning("Likely due to parsing error considering all filters should be in the where clause")
                skipped = True
                continue
            for db_column in self.db_columns:
                if not db_column.column_type:
                    # TODO: Look into updating the optional None on column_type
                    logger.warning("Missing column type")
                elif "char" not in db_column.column_type.lower():
                    # Only checking columns with a character type
                    logger.debug("Skipping db_column %s since it is not a character", db_column.column_name)
                    skipped = True
                    continue
                db_column_str = db_utils.get_column_str(column=db_column)
                if column_str.lower() == db_column_str.lower():  # If the columns are the same
                    found_db_match = True
                    if not db_column.filters:
                        await self.get_db_column_filters(column=column, db_column=db_column)
                        if not db_column.filters:
                            logger.warning(
                                "No DB column filters found, skipping column verify: %s", column.column_name
                            )
                            raise errors.UnverifiedColumns("No filters found to very, skipping")
                    llm_feedback = self.compare_column_filters(
                        llm_feedback=llm_feedback, column=column, db_column=db_column
                    )
                    break  # Found the match, don't need to loop over remaining db column for this particular column
            if not found_db_match and not skipped:
                logger.warning(f"No DB match for column: {column_str}. Here are the db columns: {db_cols_str}")
        return llm_feedback

    # TODO: Need to make tests for verifying the where clause
    async def verify_where_clause_distinct_values(self, sql_query: str) -> Optional[str]:
        columns = await self.get_where_clause_columns(sql_query=sql_query)
        if not columns:
            logger.info("No where clause columns to verify")
            return None
        handler = ChatMessageHandler(
            prompt_metadata=self.prompt_metadata,
            chat_metadata=self.agent.chat_metadata,
            redis_client_async=self.redis_client_async,
        )
        await handler.create_message(
            db=self.db,
            role=sch.MessageRole.ASSISTANT,
            content="Verifying query filters...",
            msg_type=enums.MessageType.THOUGHT,
        )
        await handler.send_api_message()
        columns_to_verify: list = []
        for column in columns:
            logger.debug("Here are the columns to verify: %s", columns_to_verify)
            # TODO: Use an StrEnum here and label these as SQLGlot datatypes
            if column.cast_type:
                # Only include columns casted if they are casted to a string
                logger.debug("Here is the cast type: %s", column.cast_type)
                if column.cast_type in ["TEXT", "VARCHAR", "BPCHAR", "NVARCHAR", "NCHAR"]:
                    columns_to_verify.append(column)
                else:
                    logger.debug("Skipping %s column since it has a cast type != strings", column.column_name)
            else:
                columns_to_verify.append(column)
        try:
            llm_feedback = await self.verify_column_filters(columns=columns_to_verify)
        except Exception as e:
            logger.warning("Verifying the where clause failed")
            logger.error("Here is the exception %s", str(e))
            raise errors.UnverifiedColumns("Column verification failed")
        return llm_feedback

    async def get_where_clause_sample_values(self, sql_query: str) -> Optional[str]:
        """Get sample values for the LLM

        Notes
        -----
        This has been replaced by get_where_clause_distinct_values since both are providing
        guidance on how to correct the SQL query, but the distinct values approach is preferred
        since it will result in enforced correct filters as opposed to suggesting them like the
        sample values does. However, the sample values is more performant since it's not running
        a distinct to get values in the database.
        """
        columns = await self.get_where_clause_columns(sql_query=sql_query)
        if not columns:
            return None
        col_samples = await self.get_column_sample_values(columns=columns)
        col_samples_prefix = "Here are a few values from the first few rows for columns used in the WHERE clause \
in your SQL query:\n\n"
        return col_samples_prefix + col_samples

    async def get_select_sample_values(self, sql_query: str) -> tuple[Optional[list[str]], Optional[str]]:
        """Get sample values for the LLM"""
        try:
            logger.info("Here is the SQL query to parse: %s", sql_query)
            columns_base = db_utils.get_fully_qualified_col_names(
                sql_query=sql_query, dialect=self.sqlglot_dialect, ancestor_to_filter=exp.Select
            )
            columns = [db_utils.get_column_str(column) for column in columns_base]
        except (errors.SQLParseError, sqlglot_errors.ParseError) as e:
            logger.warning("SQLglot failed parsing for select statement example values: %s", str(e))
            return None, None

        if not columns:
            return None, None
        col_samples = await self.get_column_sample_values(columns=columns_base)
        return columns, col_samples

    async def get_column_sample_values(self, columns: list[sch.DBColumn]) -> str:
        """Find example values from the where clause to improve accuracy"""
        # TODO: This is a temporary placeholder until the sample values can be added as part of the initial indexing
        # Determine if there is a where clause in the SQL query
        # Get all tables for each column in a dictionary and the column names
        cols_by_table: dict = {}
        for column in columns:
            table_name = db_utils.get_table_name_from_column(column=column)
            if not cols_by_table.get(table_name):
                cols_by_table[table_name] = [column.column_name]
            else:
                # Assumes fully_qualified_col_names is returning a set of distinct cols
                cols_by_table[table_name].append(column.column_name)
        # Construct and run the queries
        col_examples = {}
        for table, cols in cols_by_table.items():
            cols_str = ",".join(cols)
            query = f"SELECT {cols_str} FROM {table}"
            logger.info("SQL query for samples: %s", query)
            # Add a LIMIT using SQLglot
            ast = parse_one(query, dialect=self.sqlglot_dialect)
            limited_ast = ast.limit(5)  # type: ignore
            limited_query = limited_ast.sql(dialect=self.sqlglot_dialect)
            # Run the SQL query
            query_result = await self.run_client_query(sql_query=limited_query)
            # Update the examples list
            for column_name in query_result.output_df.columns:
                col_values = db_utils.get_query_column_values(query_result=query_result)
                stringified_values = ", ".join([str(val) for val in col_values])
                col_examples[f"{table}.{column_name}"] = stringified_values
        final_example_str = ""
        for column_str, values in col_examples.items():
            final_example_str += f"""\
Column: {column_str}
Values: {values}\n\n"""
        return final_example_str

    async def create_sql_query(self, initial_sql_query: str):
        """This function is used to create a plan to create a correct SQL query."""
        logger.info("Here is the initial SQL query: %s", initial_sql_query)
        self.sql_query_created = True
        # Explain plan
        columns, sample_values = await self.get_select_sample_values(sql_query=initial_sql_query)
        initial_instructions = f"""
Before executing a SQL query, you need to make a plan. Do the following:
- Identify the filters for the query based on the initial user prompt: {self.prompt_metadata.initial_prompt}. \
A filter is anything that is going to be put into the where clause. List each filter using a dash instead of \
numbering them.
- Determine if you have enough information or if you need to ask the user clarifying questions. This means that for \
every filter the user has given enough context and defined it clearly. If you are unsure what column the filter \
may be referring to, ask the user a clarifying question before proceeding. Do not ask the user for the column name.
- The plan should be formatted as == Plan ==, followed by plan bullet points."""
        intermediate_instructions = ""
        if sample_values and columns:
            intermediate_instructions = f"""\n- Here are some sample values for the columns selected \
in your query: {sample_values}\n"""
        final_instructions = """\n
After stating your plan, do one of the following:
- Option 1: Ask the user a clarifying question.
- Option 2: Run this tool again to run your original or updated SQL query.
"""
        return initial_instructions + intermediate_instructions + final_instructions

    # TODO: Refactor this query to be shorter
    async def run_sql(self, sql_query: str) -> str:
        logger.info("Here is the SQL query trying to be ran: %s", sql_query)
        # Clean the SQL query format
        format_json_response = JSONResponseFormatter(
            response=sql_query,
            pydantic_format=fmt.CleanSQLFormat,
            max_tokens=1000,
            small_model_info=self.small_model_info,
        )
        extract = await format_json_response.format()
        sql_query = extract.sql_query
        logger.info("Here is the cleaned SQL query: %s", sql_query)
        # Check for any hallucinated tables
        msg = await self.check_all_tables(sql_query=sql_query)
        if msg:
            return msg
        logger.info("No hallucinated tables")
        # Check for any hallucinated columns
        try:
            logger.info("Validating all cols for run_sql")
            sql_query = await self.validate_all_columns(sql_query=sql_query)
            logger.info("Validated sql query: %s", sql_query)
        except (
            Exception,
            errors.StarQueryError,
            errors.ColumnCapitalizationError,
            errors.HallucinatedColumnError,
        ) as e:
            logger.error("Here is the error from validate_all_columns: %s", str(e))
            return str(e)
        logger.info("No hallucinated columns")
        await service_utils.update_agent_tokens(agent=self.agent, max_tokens=1000)
        if self.prior_sql_query:
            if self.prior_sql_query == sql_query:
                self.stuck_in_loop_ct += 1
                if self.stuck_in_loop_ct > STUCK_IN_LOOP_MAX_CT:
                    raise Exception("Reached max iterations.")
            else:
                self.stuck_in_loop_ct = 0
            logger.warning("Stuck in loop ct: %s", self.stuck_in_loop_ct)
            try:
                sql_similarity = db_utils.compare_sql_queries(
                    sql_source=self.prior_sql_query, sql_target=sql_query, dialect=self.sqlglot_dialect
                )
                if sql_similarity not in [enums.SQLSimilarityLabel.IDENTICAL, enums.SQLSimilarityLabel.EQUIVALENT]:
                    self.sql_query_created = False  # Check query again if using different tables
                    self.prior_sql_query = sql_query
            except Exception as e:
                logger.warning("Failed comparing sql queries: %s", str(e))
        if not self.sql_query_created:
            logger.info("Planning SQL query")
            llm_prompt = await self.create_sql_query(initial_sql_query=sql_query)
            logger.info(
                "Causing the AI to self-reflect on the SQL query with the following prompt: \n\n %s", llm_prompt
            )
            return llm_prompt
        logger.info("SQL query plan made and SQL query created")
        logger.info("Verifying column values")
        try:
            llm_feedback = await self.verify_where_clause_distinct_values(sql_query=sql_query)
            if llm_feedback:
                logger.info("Here is the llm feedback for the where clause: %s", llm_feedback)
                self.col_check_ct += 1
                logger.info("Column check run number: %s", self.col_check_ct)
                return llm_feedback
        except errors.UnverifiedColumns as e:
            logger.error(str(e))
            if self.provided_sample_vals:
                # Get where clause sample values as a backup if column check fails
                try:
                    where_clause_sample_vals = await self.get_where_clause_sample_values(sql_query=sql_query)
                    if where_clause_sample_vals:
                        self.provided_sample_vals = True
                        return f"""Review the following sample values and adjust your query WHERE clause if \
needed based on examples from the database. An example of needing to update would be if you are using an \
incorrect \
format (for example, instead of abbreviations using the full spelling or vice-versa). You can update your query \
to either fuzzy match using LIKE or exact matches. Here are the WHERE clause \
columns with sample values from the database - review and update your SQL query if necessary:

{where_clause_sample_vals}

After reviewing, run this tool again to run your original or updated SQL query."""
                except Exception as e:
                    logger.warning("where clause sample values failed with this error: %s", str(e))
        logger.info("Column filter values successfully verified")
        if self.agent.chat_metadata.semcache_response:
            await self.check_query_where_clause(self.agent.chat_metadata.semcache_response.sql_query, query2=sql_query)
        else:
            self.check_strict_mode()
        # TODO: Ensure only select statements are used
        # NOTE: Need to save the chat history at this point so the report history has a reference

        try:
            async with asyncio.timeout(TIMEOUT):
                logger.info("Running AI SQL query: %s", sql_query)
                query_result_str = await service_utils.run_ai_sql_query(
                    db=self.db,
                    conn_id=self.conn_id,
                    sql_query=sql_query,
                    db_conn_params=self.db_conn_params,
                    client_conn_params=self.client_conn_params,
                    prompt_metadata=self.prompt_metadata,
                    chat_metadata=self.agent.chat_metadata,
                    agent=self.agent,
                    client_id=self.prompt_metadata.client_id,
                    small_model_info=self.small_model_info,
                    redis_client_async=self.redis_client_async,
                )
        except TimeoutError:
            error_msg = f"SQL query took longer to execute than the max {TIMEOUT/60} minute time out limit."
            logger.error(error_msg)
            await self.db.rollback()
            raise sch.SQLTimeoutError(error_msg)
        except Exception as e:
            # TODO: Improve the debugging
            # TODO: Use a manual retriever and then pass that to the AI only after filling in with the prompt template
            if constants.SQLALCHEMY_TIMEOUT in str(e):
                error_msg = f"""Failed to connect to the database after {POOL_TIMEOUT/60} minutes. \
Connection timed out. Please try again."""
                raise sch.SQLTimeoutError(error_msg)

            msg = f"Error running SQL query. Let's verify step by step. Try rewriting your SQL query using only the tables in the provided context. Here was the error: {str(e)}"  # noqa
            logger.error(msg)
            await self.db.rollback()
            self.sql_query_created = False  # Reset so it checks it again
            return msg
        self.prior_sql_query = sql_query
        logger.info("Message sent to LLM from run_sql: %s", query_result_str)
        return query_result_str

    async def get_table_metadata_filters(
        self, conn_id: int, db_uuid: uuid.UUID, client_uuid: uuid.UUID
    ) -> MetadataFilters:
        """Get the tables for the connection based on the metadata filter

        Returns
        -------
        filters
            Metadata filters for the index
        """
        tables = await crud_table.get_conn_tables(db=self.db, conn_id=conn_id)
        if not tables:
            # Check if the DB is still indexing
            running_db_index_binary = await self.redis_client_async.hget(  # type: ignore
                str(self.vector_uuid), enums.RedisHashKeys.DB_INDEX_STATUS_KEY.value
            )
            logger.warning("Here is the vector UUID to use to debug: %s", str(self.vector_uuid))
            if running_db_index_binary:
                running_db_index = running_db_index_binary.decode("utf-8")
                if running_db_index == enums.RedisValues.NO_TABLES_ERR.value:
                    logger.error(enums.RedisValues.NO_TABLES_ERR.value)
                    raise Exception(constants.NO_TABLES)
                elif running_db_index == enums.RedisValues.NO_PERMITTED_TABLES_ERR.value:
                    logger.error(enums.RedisValues.NO_PERMITTED_TABLES_ERR.value)
                    raise Exception(constants.NO_PERMITTED_TABLES)
                elif running_db_index == enums.RedisValues.ERROR_RUNNING_DB_INDEX.value:
                    logger.error(enums.RedisValues.ERROR_RUNNING_DB_INDEX.value)
                    raise Exception(enums.RedisValues.ERROR_RUNNING_DB_INDEX.value)
                elif running_db_index == enums.RedisValues.RUNNING_DB_INDEX.value:
                    raise Exception(constants.INDEX_DB_ERROR_MSG)
                else:
                    raise ValueError(constants.NO_TABLES)
            else:
                raise ValueError(constants.NO_TABLES)
        metadata_filters = []
        for table in tables:
            metadata_filters.append(MetadataFilter(key="name", value=table.table_name, operator=FilterOperator.IN))
        metadata_filters += [
            MetadataFilter(key="db_uuid", value=str(db_uuid), operator=FilterOperator.EQ),
            MetadataFilter(key="client_uuid", value=str(client_uuid), operator=FilterOperator.EQ),
            MetadataFilter(key="vector_type", value=enums.VectorSourceType.TABLE.value, operator=FilterOperator.EQ),
        ]
        return MetadataFilters(filters=metadata_filters)

    def setup_sql_retriever(self, top_k: int) -> SQLTableRetriever:
        """Return the SQL engine"""
        index_table_retriever = self.table_index.as_retriever(similarity_top_k=top_k, filters=self.filters)
        table_retriever = base.ObjectRetriever(
            retriever=index_table_retriever,
            object_node_mapping=SQLTableNodeMapping(),
        )
        sql_retriever = SQLTableRetriever(
            table_retriever=table_retriever,
        )
        return sql_retriever

    async def use_sub_questions(self, prompt) -> list:
        # Ask the agent to classify the prompt
        # TODO: Add a callback manager to track token usage here
        ai_catalog = AICatalog()
        agent_llm = ai_catalog.get_llm(model_info=self.large_model_info)
        agent = SimpleChatEngine.from_defaults(llm=agent_llm)
        agent_prompt = f"""\
Return True if the following is True, otherwise return False. If you consider the following prompt to be \
multiple questions in one, uses many commas, requests many things which likely will require using multiple tables, or \
is in general considered to be complex, return True. Otherwise return False. Here is the prompt: \
{prompt}"""
        agent_output = await agent.achat(message=agent_prompt)
        # Extract the answer
        format_json_response = JSONResponseFormatter(
            response=agent_output.response,
            pydantic_format=fmt.TrueFalseBool,
            llm=agent_llm,  # NOTE: GPT 4o-mini selects sub-questions too often
            small_model_info=self.small_model_info,
        )
        extract = await format_json_response.format()
        logger.debug("Decision to use sub-question tool: %s", extract.true_false_bool)
        if not extract.true_false_bool:
            return []
        logger.debug("Agent decided to use sub-questions to retrieve tables")
        # Ask the agent for the sub prompts
        agent_prompt = f"""\
Take the following prompt and break it out into 2-3 more distinct sub-prompts. \
Each sub-prompt should be a component of the original prompt with additional keywords \
and synonyms added to make the topic clear. Here is an example: \
Original prompt: Get me a report with users, teams, and clients. \n\
New sub-prompts: \n\
1. A report of users (i.e. purchaser and person)\n\
2. A report of teams (i.e. groups and crew)\n\
3. A report of clients (i.e. customers) \n\
Use a numbered list when answering. There should be no overlap in the subjects of the sub-prompts.\
Here is the prompt that needs to be broken out: \n\n\
{prompt}
"""
        agent_output = await agent.achat(message=agent_prompt)
        # Extract the sub prompts
        format_json_response = JSONResponseFormatter(
            response=agent_output.response,
            pydantic_format=fmt.SubPrompts,
            llm=agent_llm,  # NOTE: GPT 4o-mini selects sub-questions too often
            small_model_info=self.small_model_info,
        )
        extract = await format_json_response.format()
        # For each sub-prompt, get related tables
        final_tables = set()
        logger.debug("Here are the sub_questions: \n-%s", "\n- ".join(extract.sub_prompts))
        for sub_prompt in extract.sub_prompts:
            retrieved_tables = await self.get_sql_tables_helper(
                inquiry=sub_prompt, sql_retriever=self.sub_prompt_sql_retriever
            )
            final_tables.update(retrieved_tables)
        return list(final_tables)

    async def get_sql_tables(self, inquiry):
        """Retrieve SQL tables to use in the SQL query"""
        # Need more tokens for large SQL queries
        await service_utils.update_agent_tokens(agent=self.agent, max_tokens=1000)
        try:
            tables = await self.use_sub_questions(prompt=inquiry)
            if not tables:
                tables = await self.get_sql_tables_helper(inquiry=inquiry, sql_retriever=self.sql_retriever)
            tables_str = "\n\n".join(tables)
        except errors.NoRelevantTables as e:
            logger.warning("The AI was unable to find any relevant tables")
            return str(e)
        logger.debug("Here are the retrieved tables: %s", tables_str)
        # Resolve jinja
        tables_str = await TableManager.arender_query_jinja(jinja_str=tables_str, schemas=self.schemas)
        # If there is unresolved Jinja, then throw an error
        pattern = r"\{\{\s*.+?\s*\}\}"
        jinja_detected = re.findall(pattern, tables_str)
        if jinja_detected:
            # If there is jinja, then halt and send error to the user
            raise Exception(constants.UNRESOLVED_JINJA)
        logger.debug("Here are the schemas: %s", self.schemas)
        formatted_prompt = DB_METADATA_PROMPT.format(
            inquiry=inquiry,
            schema=tables_str,
            db_type=self.client_conn_params.database_type.value,
            run_sql_query_tool=constants.get_sql_execution_tool_nm(conn_id=self.conn_id),
        )
        return formatted_prompt
        # TODO: Use async task group or async for here to quickly get all tables
        # (this is referring to within the _aget_table_context method)

    async def get_sql_tables_helper(self, inquiry: str, sql_retriever: SQLTableRetriever) -> list:
        query_bundle = QueryBundle(inquiry)
        try:
            # TODO: See if there is something more efficient than checking this every time
            index_update_error = await self.redis_client_async.hget(  # type: ignore
                str(self.vector_uuid), enums.RedisHashKeys.DB_INDEX_UPDATE_STATUS_KEY.value
            )
            if index_update_error:
                logger.error("Index update error: %s", index_update_error)
                # TODO: This isn't resolving, but once triggered it is perpetually broken
                # HACK: Commenting out for now
                # raise Exception("Index update error")
            tables = await sql_retriever._aget_table_context(
                query_bundle=query_bundle, relevance_threshold=RELEVANCE_THRESHOLD
            )
        except Exception as e:
            logger.error(e)
            if isinstance(e, redis.exceptions.ResponseError) or NO_DOCS in str(e) or index_update_error:
                if isinstance(e, redis.exceptions.ResponseError):
                    logger.warning("Index not found: %s", str(e))
                elif NO_DOCS in str(e):
                    logger.warning("No docs found in index: %s", str(e))
                elif index_update_error:
                    logger.warning("Error found when updating DB index: %s", str(e))
                raise errors.SQLIndexError(constants.REINDEXING_DB_ERROR_MSG)
            else:
                raise e
        if not tables:
            raise errors.NoRelevantTables(
                """No relevant tables found for this question. Please rephrase your question and try again \
or check the underlying SQL database connection for misconfiguration."""
            )
        return tables

    async def run_client_query(self, sql_query: str) -> sch.QueryResultDF:
        """Run a query against the client database"""
        try:
            sql_query = await self.validate_all_columns(sql_query=sql_query)
        except (errors.StarQueryError, errors.ColumnCapitalizationError, errors.HallucinatedColumnError) as e:
            logger.error("Error validating SQL query: %s", str(e))
            raise e
        logger.info("Running client query... %s", sql_query)
        try:
            async with asyncio.timeout(TIMEOUT):
                mng_query = query.ClientQueryManager(
                    db_conn_params=self.db_conn_params, client_conn_params=self.client_conn_params, sql_query=sql_query
                )
                query_result = await mng_query.run_client_query()
                logger.info("Completed running client query")
        except TimeoutError:
            error_msg = f"SQL query took longer to execute than the max {TIMEOUT/60} minute time out limit."
            logger.error(error_msg)
            await self.db.rollback()
            raise sch.SQLTimeoutError(error_msg)
        except Exception as e:
            # TODO: Improve the debugging
            if constants.SQLALCHEMY_TIMEOUT in str(e):
                error_msg = f"""Failed to connect to the database after {POOL_TIMEOUT/60} minutes. \
Connection timed out. Please try again."""
                raise sch.SQLTimeoutError(error_msg)
            logger.warning("Error running this SQL query: %s", sql_query)
            logger.warning("Here is the error: %s", str(e))
            await self.db.rollback()
            raise errors.SQLRunError("Error running SQL query") from e
        return query_result
