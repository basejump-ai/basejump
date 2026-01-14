import asyncio
from typing import Optional

from llama_index.core.tools import FunctionTool
from llama_index.core.tools.function_tool import create_tool_metadata
from sqlalchemy.ext.asyncio import AsyncSession

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.client import query
from basejump.core.database.db_connect import POOL_TIMEOUT
from basejump.core.database.result import store
from basejump.core.models import constants, enums, errors
from basejump.core.models import schemas as sch
from basejump.core.models.ai import formats as fmt
from basejump.core.models.ai import formatter
from basejump.core.service.base import BaseChatAgent
from basejump.core.service.tools import BaseTool, tool_utils
from basejump.core.service.tools.sql import parse, sample, validate

logger = set_logging(handler_option="stream", name=__name__)

STUCK_IN_LOOP_MAX_CT = 3


class SQLRunnerTool(BaseTool):
    def __int__(
        self,
        db: AsyncSession,
        prompt_metadata: sch.PromptMetadata,
        service_context: sch.ServiceContext,
        result_store: store.ResultStore,
        agent: BaseChatAgent,
        sqlglot_dialect: str,
        client_conn_params: sch.SQLDBSchema,
        db_conn_params: sch.SQLDBSchema,
        conn_id: int,
        schemas: list[sch.DBSchema],
        select_sample_values: bool = False,
        verbose: bool = False,
    ):
        self.db = db
        self.prompt_metadata = prompt_metadata
        self.service_context = service_context
        self.result_store = result_store
        self.agent = agent
        self.sqlglot_dialect = sqlglot_dialect
        self.client_conn_params = client_conn_params
        self.db_conn_params = db_conn_params
        self.conn_id = conn_id
        self.schemas = schemas
        self.select_sample_values = select_sample_values
        self.verbose = verbose
        self.sql_query_created = False
        self.prior_sql_query: Optional[str] = None
        self.stuck_in_loop_ct = 0
        self.col_check_ct = 0
        self.provided_sample_vals = False

    def get_tools(self) -> list[FunctionTool]:
        func = self.run_sql

        name = constants.get_sql_execution_tool_nm(conn_id=self.conn_id)
        assert func.__name__ in name
        tool_metadata = create_tool_metadata(
            fn=func,
            name=name,
            description="Run this function to execute a SQL query",
        )
        sql_exec_tool = FunctionTool.from_defaults(fn=func, async_fn=func, tool_metadata=tool_metadata)

        return [sql_exec_tool]

    def check_strict_mode(self):
        user_role = enums.USER_ROLES_LVL_LKUP[self.prompt_metadata.user_role]
        admin_role = enums.USER_ROLES_LVL_LKUP[enums.UserRoles.ADMIN.value]
        if self.agent.chat_metadata.verify_mode == enums.VerifyMode.STRICT and user_role < admin_role:
            raise errors.StrictModeFlagged

    async def create_sql_query(self, initial_sql_query: str):
        """This function is used to create a plan to create a correct SQL query."""
        logger.info("Here is the initial SQL query: %s", initial_sql_query)
        self.sql_query_created = True
        # Explain plan
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
        if self.select_sample_values:
            sampler = sample.SQLSampler(sqlglot_dialect=self.sqlglot_dialect, conn_params=self.client_conn_params)
            columns, sample_values = await sampler.get_select_sample_values(sql_query=initial_sql_query)
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
        validator = validate.SQLValidator(
            db=self.db,
            sqlglot_dialect=self.sqlglot_dialect,
            conn_id=self.conn_id,
            schemas=self.schemas,
            verbose=self.verbose,
            conn_params=self.client_conn_params,
            agent=self.agent,
            redis_client_async=self.service_context.redis_client_async,
        )

        # Get required info for SQL validation
        await validator.get_db_table_info()

        # Clean the SQL query format
        format_json_response = formatter.JSONResponseFormatter(
            response=sql_query,
            pydantic_format=fmt.CleanSQLFormat,
            max_tokens=1000,
            small_model_info=self.service_context.small_model_info,
        )
        extract = await format_json_response.format()
        sql_query = extract.sql_query
        logger.info("Here is the cleaned SQL query: %s", sql_query)
        # Check for any hallucinated tables
        msg = await validator.check_all_tables(sql_query=sql_query)
        if msg:
            return msg
        logger.info("No hallucinated tables")
        # Check for any hallucinated columns
        try:
            sql_query = await validator.validate_all_columns(sql_query=sql_query)
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
        await tool_utils.update_agent_tokens(agent=self.agent, max_tokens=1000)
        if self.prior_sql_query:
            if self.prior_sql_query == sql_query:
                self.stuck_in_loop_ct += 1
                if self.stuck_in_loop_ct > STUCK_IN_LOOP_MAX_CT:
                    raise Exception("Reached max iterations.")
            else:
                self.stuck_in_loop_ct = 0
            logger.warning("Stuck in loop ct: %s", self.stuck_in_loop_ct)
            try:
                parser = parse.SQLParser(sqlglot_dialect=self.sqlglot_dialect, verbose=self.verbose)
                sql_similarity = parser.compare_sql_queries(
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
            if self.verbose:
                logger.info(
                    "Causing the AI to self-reflect on the SQL query with the following prompt: \n\n %s", llm_prompt
                )
            return llm_prompt
        logger.info("SQL query plan made and SQL query created")
        logger.info("Verifying column values")
        try:
            llm_feedback = await validator.verify_where_clause_distinct_values(sql_query=sql_query)
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
                    sampler = sample.SQLSampler(
                        sqlglot_dialect=self.sqlglot_dialect, conn_params=self.client_conn_params
                    )
                    where_clause_sample_vals = await sampler.get_where_clause_sample_values(sql_query=sql_query)
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
            if self.agent.chat_metadata.semcache_response:  # check again after checking the query where clause
                self.check_strict_mode()
        else:
            self.check_strict_mode()
        # TODO: Ensure only select statements are used
        # NOTE: Need to save the chat history at this point so the report history has a reference

        try:
            async with asyncio.timeout(query.TIMEOUT):
                logger.info("Running AI SQL query: %s", sql_query)
                query_result_str = await tool_utils.run_ai_sql_query(
                    db=self.db,
                    conn_id=self.conn_id,
                    sql_query=sql_query,
                    db_conn_params=self.db_conn_params,
                    client_conn_params=self.client_conn_params,
                    prompt_metadata=self.prompt_metadata,
                    chat_metadata=self.agent.chat_metadata,
                    agent=self.agent,
                    client_id=self.prompt_metadata.client_id,
                    small_model_info=self.service_context.small_model_info,
                    redis_client_async=self.service_context.redis_client_async,
                    result_store=self.result_store,
                    verbose=self.verbose,
                )
        except TimeoutError:
            error_msg = f"SQL query took longer to execute than the max {query.TIMEOUT/60} minute time out limit."
            logger.error(error_msg)
            await self.db.rollback()
            raise sch.SQLTimeoutError(error_msg)
        except errors.AbortMultipartUpload as e:
            return str(e)
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
        if self.verbose:
            logger.info("Message sent to LLM: %s", query_result_str)
        return query_result_str
