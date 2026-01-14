"""Configure the SQL tool"""

import asyncio
import copy
import re
import uuid
from typing import Optional

import redis
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
from sqlalchemy.ext.asyncio import AsyncSession

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.client import query
from basejump.core.database.crud import crud_connection, crud_table
from basejump.core.database.db_connect import POOL_TIMEOUT, TableManager
from basejump.core.database.result import store
from basejump.core.database.vector_utils import get_vector_idx
from basejump.core.models import constants, enums, errors
from basejump.core.models import schemas as sch
from basejump.core.models.ai import formats as fmt
from basejump.core.models.ai import formatter
from basejump.core.models.ai.catalog import AICatalog
from basejump.core.models.prompts import DB_METADATA_PROMPT
from basejump.core.service.base import BaseChatAgent
from basejump.core.service.tools import tool_utils
from basejump.core.service.tools.sql import parse, sample, validate

logger = set_logging(handler_option="stream", name=__name__)
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
        self.confirm_tool_retrieval = False
        self.db_conn_params = db_conn_params
        self.client_conn_params = client_conn_params
        self.vector_id = vector_id
        self.tools: list[FunctionTool] = []
        self.is_demo = False
        self.sql_query_created = False
        self.sqlglot_dialect = enums.DB_TYPE_TO_SQLGLOT_DIALECT_LKUP.get(self.client_conn_params.database_type)
        self.prior_sql_query: Optional[str] = None
        self.col_check_ct = 0
        self.provided_sample_vals = False
        self.large_model_info = service_context.large_model_info
        self.small_model_info = service_context.small_model_info
        self.embedding_model_info = service_context.embedding_model_info
        self.sql_engine = service_context.sql_engine
        self.redis_client_async = service_context.redis_client_async
        self.stuck_in_loop_ct = 0
        self.select_sample_values = select_sample_values
        self.result_store = result_store
        self.verbose = verbose
        self.retrieved_sql_tables = False

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
        self.retrieved_sql_tables = True
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
            redis_client_async=self.redis_client_async,
        )

        # Get required info for SQL validation
        await validator.get_db_table_info()

        # Clean the SQL query format
        format_json_response = formatter.JSONResponseFormatter(
            response=sql_query,
            pydantic_format=fmt.CleanSQLFormat,
            max_tokens=1000,
            small_model_info=self.small_model_info,
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
                    small_model_info=self.small_model_info,
                    redis_client_async=self.redis_client_async,
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
        format_json_response = formatter.JSONResponseFormatter(
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
        format_json_response = formatter.JSONResponseFormatter(
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
        await tool_utils.update_agent_tokens(agent=self.agent, max_tokens=1000)
        try:
            tables = await self.use_sub_questions(prompt=inquiry)
            if not tables:
                tables = await self.get_sql_tables_helper(inquiry=inquiry, sql_retriever=self.sql_retriever)
            tables_str = "\n\n".join(tables)
        except errors.NoRelevantTables as e:
            logger.warning("The AI was unable to find any relevant tables")
            return str(e)
        if self.verbose:
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
