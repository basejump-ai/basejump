import re
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlglot import errors as sqlglot_errors
from sqlglot import exp, parse_one
from sqlglot.dialects.dialect import Dialects

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database import db_utils
from basejump.core.database.client import query
from basejump.core.database.crud import crud_table
from basejump.core.database.manager import TableManager
from basejump.core.models import enums, errors
from basejump.core.models import schemas as sch
from basejump.core.models.prompts import ZERO_ROW_PROMPT
from basejump.core.service.agents.tools.sql.parser import SQLParser
from basejump.core.service.base import BaseChatAgent, ChatMessageHandler

logger = set_logging(handler_option="stream", name=__name__)


class SQLValidator:
    def __init__(
        self,
        db: AsyncSession,
        sqlglot_dialect: str,
        conn_id: int,
        schemas: list[sch.DBSchema],
        verbose: bool,
        conn_params: sch.SQLDBSchema,
        agent: BaseChatAgent,
        service_context: sch.ServiceContext,
    ):
        self.db = db
        self.sqlglot_dialect = sqlglot_dialect
        self.conn_id = conn_id
        self.schemas = schemas
        self.verbose = verbose
        self.conn_params = conn_params
        self.agent = agent
        self.service_context = service_context
        self.parser = SQLParser(sqlglot_dialect=sqlglot_dialect, verbose=verbose)
        self.db_columns: list = []

    async def check_all_tables(self, sql_query: str) -> Optional[str]:
        try:
            all_tables = await self._get_db_tables()
            logger.info("Dialect: %s", self.sqlglot_dialect)
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
            all_full_tables_lowered = {table.table_name.lower() for table in all_tables}
            all_tables_lowered = {table.table_name.lower().split(".")[-1] for table in all_tables}
            # Find the ignored tables
            ignored_tables_lowered = {table.table_name.lower() for table in all_tables if table.ignore}
            tbl_overlap = ignored_tables_lowered & query_tbls_lowered
            # Check for hallucinated tables
            if (
                not query_tbls_lowered.issubset(all_full_tables_lowered)
                and not query_tbls_lowered.issubset(all_tables_lowered)
            ) or tbl_overlap:
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

    def get_hallucinated_columns(self, query_columns: set, database_columns: set) -> list[str]:
        hallucinated_columns = []
        for query_column in query_columns:
            # NOTE: If the same column exists in different schemas, it will not be caught
            # TODO: Enforce schemas being specified for tables to avoid same table + column names in different schemas
            # getting through
            column_exists = any(query_column in database_column for database_column in database_columns)
            if not column_exists:
                logger.warning(f"Query column: {query_column} was not found in {database_columns}.")
                hallucinated_columns.append(query_column)
        return hallucinated_columns

    async def check_all_columns(self, sql_columns: list[sch.DBColumn], db_columns: list[sch.DBColumn]):
        """Check columns for hallucinations and capitalization errors"""
        try:
            # Get column sets for the SQL query and the database
            distinct_db_columns = {db_utils.get_column_str(column) for column in db_columns}
            ignored_columns = {db_utils.get_column_str(column) for column in db_columns if column.ignore}
            valid_database_columns = distinct_db_columns - ignored_columns
            valid_database_columns_lowered = {column.lower() for column in valid_database_columns}
            query_columns = {db_utils.get_column_str(column) for column in sql_columns}
            query_columns_lowered = {column.lower() for column in query_columns}

            # Check for hallucinated lowered columns
            hallucinated_columns = self.get_hallucinated_columns(
                query_columns=query_columns_lowered, database_columns=valid_database_columns_lowered
            )
            if hallucinated_columns:
                ai_msg = f'The following column(s) does not exist in the \
table. Do not use these column(s): {", ".join(hallucinated_columns)}'
                logger.warning(ai_msg)
                raise errors.HallucinatedColumnError(ai_msg)

            # Check for hallucination due to miscapitalization
            miscapitalized_columns = self.get_hallucinated_columns(
                query_columns=query_columns, database_columns=valid_database_columns
            )
            if miscapitalized_columns:
                ai_msg = f'The following column(s) exists in the \
table, but you miscapitalized it/them: {", ".join(miscapitalized_columns)}'
                logger.warning(ai_msg)
                raise errors.ColumnCapitalizationError(ai_msg)

        except (errors.SQLParseError, sqlglot_errors.ParseError) as e:
            logger.warning("SQLglot failed parsing: %s", str(e))
            return None
        else:
            return None

    async def validate_all_columns(self, sql_query: str) -> str:
        """Validate all columns in the SQL query for capitalization errors and hallucinations and quote if needed"""
        logger.info("Validating all columns in the SQL query: %s", sql_query)
        db_columns = await self._get_db_columns()
        try:
            sql_columns = self.parser.get_fully_qualified_col_names(sql_query=sql_query, dialect=self.sqlglot_dialect)
            await self.check_all_columns(sql_columns=sql_columns, db_columns=db_columns)
            logger.info("All cols checked")
            try:
                sql_query = self.parser.quote_case_sensitive_cols(sql_query=sql_query, columns=db_columns)
            except Exception as e:
                logger.warning(str(e))
                logger.traceback()
            if self.verbose:
                logger.info("Here is the SQL query after quoting: %s", sql_query)
            return sql_query
        except (errors.StarQueryError, errors.ColumnCapitalizationError, errors.HallucinatedColumnError) as e:
            logger.warning("Error in validating columns: %s", str(e))
            raise e

    async def check_query_where_clause(self, query1: str, query2: str) -> None:
        """If there is a semantically cached response, this function checks if that the difference \
    between that semantically similar query and the new SQL query is only the WHERE clause. If it is, \
    then it can still be considered verified.
        """
        comparison = self.parser.compare_sql_queries_no_where_clause(
            query1=query1, query2=query2, dialect=self.sqlglot_dialect
        )
        if comparison == enums.SQLSimilarityLabel.IDENTICAL:
            logger.info("Found verified similar SQL Query from semantic cache")
        else:
            # If it's not identical after checking the WHERE clause, then it is not verified
            assert self.agent.chat_metadata.semcache_response
            self.agent.chat_metadata.semcache_response.verified = False

    async def _extend_db_columns(self, columns: list[sch.DBColumn]) -> None:
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

    async def _get_db_column_filters(self, column: sch.DBColumn, db_column: sch.DBColumn):
        # Do a fuzzy match to find similar values
        tbl_name = db_utils.get_table_name_from_column(column=column)
        assert column.column_w_func, "This should be populated. Check your code and fix."
        # Get distinct values
        ast = exp.select(column.column_w_func).from_(tbl_name)
        # Loop through filters and create a like
        if "lower(" in column.column_w_func:  # avoiding using lower twice
            col_name = exp.column(column.column_w_func)
        else:
            col_name = exp.func("lower", exp.column(column.column_w_func), dialect=self.sqlglot_dialect)  # type: ignore # noqa
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
        async with query.ClientQueryRunner(client_conn_params=self.conn_params, sql_query=fuzzy_sql) as query_runner:
            query_result = await query_runner.arun_client_query()
        if query_result.output_df.empty:
            logger.warning("The fuzzy sql returned no results running distinct without filter")
            sql = "SELECT DISTINCT " + re.split("select", fuzzy_sql_base, flags=re.IGNORECASE)[1]
            distinct_ast = parse_one(sql, dialect=self.sqlglot_dialect)
            distinct_ast.args["where"] = None
            logger.warning("Here is the AST select: %s", distinct_ast)
            sql = distinct_ast.sql(dialect=self.sqlglot_dialect) + " LIMIT 100000"
            logger.info("Running unfuzzy sql %s", sql)
            async with query.ClientQueryRunner(client_conn_params=self.conn_params, sql_query=sql) as query_runner:
                query_result = await query_runner.arun_client_query()
        # Add results to db_column.filters
        db_column.filters = db_utils.get_query_column_values(query_result=query_result)

    def _compare_column_filters(self, llm_feedback: str, column: sch.DBColumn, db_column: sch.DBColumn):
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
in the WHERE clause did not match any values in the database. Here are the available values in the database,\
please update your filter value to match one or multiple of these instead: {db_col_filters}\n"""
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
in the WHERE clause did not match any values in the database. Here are the available values in the database,\
please update your filter value to match one or multiple of these instead: {db_col_filters}\n"""
                    else:
                        sample_ct = 50
                        db_col_filters = ", ".join([f"'{str(val)}'" for val in db_column.filters[:sample_ct]])
                        llm_feedback += f"""- {column_str}: The filter value used for this column in the WHERE \
clause did not exactly match any values in the database. Here is a sample of the available values in the database.\
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
            await self._extend_db_columns(columns=columns)
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
                        await self._get_db_column_filters(column=column, db_column=db_column)
                        if not db_column.filters:
                            logger.warning(
                                "No DB column filters found, skipping column verify: %s", column.column_name
                            )
                            raise errors.UnverifiedColumns("No filters found to very, skipping")
                    llm_feedback = self._compare_column_filters(
                        llm_feedback=llm_feedback, column=column, db_column=db_column
                    )
                    break  # Found the match, don't need to loop over remaining db column for this particular column
            if not found_db_match and not skipped:
                logger.warning(f"No DB match for column: {column_str}. Here are the db columns: {db_cols_str}")
        return llm_feedback

    # TODO: Need to make tests for verifying the where clause
    async def verify_where_clause_distinct_values(self, sql_query: str) -> Optional[str]:
        try:
            columns = await self.parser.get_where_clause_columns(sql_query=sql_query)
        except errors.StarQueryError as e:
            return str(e)
        if not columns:
            logger.info("No where clause columns to verify")
            return None
        handler = ChatMessageHandler(
            prompt_metadata=self.agent.prompt_metadata,
            chat_metadata=self.agent.chat_metadata,
            redis_client_async=self.service_context.redis_client_async,
            verbose=self.verbose,
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

    async def _get_db_tables(self) -> list[sch.SQLTableInfo]:
        all_tables = await crud_table.get_all_tables(db=self.db)
        retrieved_tables = []
        for tbl in all_tables:
            result = await TableManager.arender_query_jinja(jinja_str=tbl.table_name, schemas=self.schemas)
            retrieved_tables.append(sch.SQLTableInfo(table_name=result, ignore=tbl.ignore))
        if not retrieved_tables:
            raise ValueError("Missing table information in the database.")
        return retrieved_tables

    async def _get_db_columns(self) -> list[sch.DBColumn]:
        db_cols = await crud_table.get_all_columns(db=self.db, conn_id=self.conn_id)
        retrieved_columns = []
        for col in db_cols:
            table_name = await TableManager.arender_query_jinja(jinja_str=col.table_name, schemas=self.schemas)
            col_obj = sch.DBColumn(
                column_name=col.column_name,
                table_name=db_utils.get_table_name(table_name=table_name),
                schema_name=db_utils.get_table_schema(table_name=table_name),
                quoted=col.quoted,
                ignore=col.ignore,
            )
            retrieved_columns.append(col_obj)
        if not retrieved_columns:
            raise ValueError("Missing column information in the database.")
        return retrieved_columns
