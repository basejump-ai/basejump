import copy
import re
import uuid

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
from basejump.core.database.crud import crud_connection, crud_table
from basejump.core.database.manager import TableManager
from basejump.core.database.vector_utils import get_vector_idx
from basejump.core.models import constants, enums, errors
from basejump.core.models import schemas as sch
from basejump.core.models.ai import formats as fmt
from basejump.core.models.ai import formatter
from basejump.core.models.ai.catalog import AICatalog
from basejump.core.models.prompts import DB_METADATA_PROMPT
from basejump.core.service.agents.tools import tool_utils
from basejump.core.service.agents.tools.base import BaseTool
from basejump.core.service.base import BaseChatAgent

logger = set_logging(handler_option="stream", name=__name__)

RELEVANCE_THRESHOLD = 0.1


class TableRetrieverTool(BaseTool):
    TABLES_TO_RETRIEVE: int = 12

    def __init__(
        self,
        db: AsyncSession,
        agent: BaseChatAgent,
        sql_tool_context: sch.SQLToolContext,
    ):
        self.db = db
        self.agent = agent
        self.service_context = sql_tool_context.service_context
        self.client_conn_params = sql_tool_context.client_conn_params
        self.conn_id = sql_tool_context.conn_id
        self.vector_id = sql_tool_context.vector_id
        self.prompt_metadata = sql_tool_context.prompt_metadata
        self.db_uuid = sql_tool_context.db_uuid
        self.schemas = sql_tool_context.client_conn_params.schemas or []
        self.verbose = sql_tool_context.verbose
        self.is_demo = False
        self.retrieved_sql_tables = False

    # TODO: This would change to 'get sql' once we have a SQL specific model and
    # would take no input args
    async def get_tools(self) -> list[FunctionTool]:
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
        return [sql_tool]

    async def setup_sql_table_vector_index(self, vector_id: int, client_id: int) -> VectorStoreIndex:
        """Load the vector index"""
        # Get the vector DB
        vector_db = await crud_connection.get_vector_connection_from_id(db=self.db, vector_id=vector_id)
        # Initialize the environment
        vector_schema = sch.VectorDBSchema.model_validate(vector_db)
        ai_catalog = AICatalog()
        settings = ai_catalog.get_settings(
            llm=self.agent.agent_llm, embedding_model_info=self.service_context.embedding_model_info
        )
        table_index = get_vector_idx(
            client_id=client_id,
            vector_schema=vector_schema,
            settings=settings,
            redis_client_async=self.service_context.redis_client_async,
        )

        return table_index

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
            running_db_index_binary = await self.service_context.redis_client_async.hget(  # type: ignore
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
        agent_llm = ai_catalog.get_llm(model_info=self.service_context.large_model_info)
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
            small_model_info=self.service_context.small_model_info,
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
            small_model_info=self.service_context.small_model_info,
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
        except Exception as e:
            logger.warning(f"Failed to use sub questions: {str(e)}")
        try:
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
            index_update_error = await self.service_context.redis_client_async.hget(  # type: ignore
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
