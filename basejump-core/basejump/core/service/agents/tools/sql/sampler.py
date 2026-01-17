from typing import Optional

from sqlglot import errors as sqlglot_errors
from sqlglot import exp, parse_one

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database import db_utils
from basejump.core.database.client.query import ClientQueryRunner
from basejump.core.models import errors
from basejump.core.models import schemas as sch
from basejump.core.service.agents.tools.sql.parser import SQLParser

logger = set_logging(handler_option="stream", name=__name__)


class SQLSampler:
    def __init__(self, sqlglot_dialect: str, conn_params: sch.SQLDBSchema, verbose: bool = False):
        self.sqlglot_dialect = sqlglot_dialect
        self.conn_params = conn_params
        self.verbose = verbose

    async def get_select_sample_values(self, sql_query: str) -> tuple[Optional[list[str]], Optional[str]]:
        """Get sample values for the LLM"""
        try:
            logger.info("Here is the SQL query to parse: %s", sql_query)
            parser = SQLParser(sqlglot_dialect=self.sqlglot_dialect, verbose=self.verbose)
            columns_base = parser.get_fully_qualified_col_names(
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
            async with ClientQueryRunner(client_conn_params=self.conn_params, sql_query=limited_query) as query_runner:
                query_result = await query_runner.arun_client_query()
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
        parser = SQLParser(sqlglot_dialect=self.sqlglot_dialect, verbose=self.verbose)
        columns = await parser.get_where_clause_columns(sql_query=sql_query)
        if not columns:
            return None
        col_samples = await self.get_column_sample_values(columns=columns)
        col_samples_prefix = "Here are a few values from the first few rows for columns used in the WHERE clause \
    in your SQL query:\n\n"
        return col_samples_prefix + col_samples
