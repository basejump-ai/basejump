"""Utilities that support the AI functionality or other core business logic within the application"""

import asyncio
import copy
import json
import uuid
from asyncio import Task
from datetime import datetime
from typing import Callable, Optional

import boto3
from basejump.core.common.config.logconfig import set_logging
from basejump.core.database import query, upload
from basejump.core.database.crud import crud_chat, crud_connection, crud_result
from basejump.core.database.db_connect import ConnectDB
from basejump.core.database.db_utils import extract_visual_info
from basejump.core.database.format_response import get_title_description
from basejump.core.database.index import DBTableIndexer
from basejump.core.database.upload import S3_PREFIX
from basejump.core.database.vector_utils import get_index_name
from basejump.core.models import enums, models
from basejump.core.models import schemas as sch
from basejump.core.models.prompts import get_sql_result_prompt
from basejump.core.service.base import (
    AgentSetup,
    BaseAgent,
    ChatMessageHandler,
    SimpleAgent,
)
from basejump.core.service.tools.visualize import VisTool
from redis.asyncio import Redis as RedisAsync
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.orm.exc import NoResultFound

logger = set_logging(handler_option="stream", name=__name__)


async def setup_connection(
    db: AsyncSession,
    client_id: int,
    conn_params: sch.SQLDBSchema,
    db_id: int,
    login_params: sch.CreateDBConn,
    sql_engine: AsyncEngine,
) -> sch.GetSQLConn:
    # Verify the connection
    conn_db = ConnectDB(conn_params=conn_params)
    await asyncio.to_thread(conn_db.verify_client_connection)
    # Create the connection
    db_login = await crud_connection.create_db_conn(
        db=db,
        db_id=db_id,
        login_params=login_params,
        client_id=client_id,
        data_source_desc=conn_params.data_source_desc,
    )
    # Add to connection association table in the background
    background_tasks = set()
    task: Task = asyncio.create_task(
        crud_connection.setup_connection_assoc_table(
            client_id=client_id,
            conn_id=copy.copy(db_login.conn_id),
            conn_params=conn_params,
            sql_engine=sql_engine,
            db_id=db_id,
        )
    )
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    db_params = await crud_connection.get_database_params_from_id(db=db, db_id=db_login.db_id)
    assert db_params
    return sch.GetSQLConn(
        conn_uuid=db_login.conn_uuid,
        db_uuid=copy.copy(db_params.db_uuid),
    )


async def setup_vector(db: AsyncSession, client_id: int, index_db_tables: DBTableIndexer) -> int:
    vectordb_schema = sch.VectorDBSchema(
        vector_database_vendor=index_db_tables.vector_database_vendor.value,
        vector_datasource_type=index_db_tables.vector_datasource_type.value,
    )
    vectordb_schema.index_name = get_index_name(client_id=client_id)
    vector_db = await crud_connection.save_vector_store_info(
        db=db,
        client_id=client_id,
        vector_uuid=index_db_tables.vector_uuid,
        vectordb_schema=vectordb_schema,
    )
    return vector_db.vector_id


async def create_alias_name(db: AsyncSession, conn_params: sch.SQLDBSchema):
    if not conn_params.database_name_alias:
        conn_params.database_name_alias = conn_params.database_name
    alias_list = await crud_connection.get_db_aliases(db=db)
    for alias in alias_list:
        if conn_params.database_name_alias == alias.alias_name:
            alias_num_list = [
                alias.alias_number for alias in alias_list if conn_params.database_name_alias in alias.alias_name
            ]
            conn_params.database_name_alias_number = max(alias_num_list) + 1
            conn_params.database_name_alias = f"{alias.alias_name} ({conn_params.database_name_alias_number})"
            break


async def setup_db(
    db: AsyncSession,
    client_user: sch.ClientUserInfo,
    conn_params: sch.SQLDBSchema,
    redis_client_async: RedisAsync,  # TODO: Looks like this can be removed
    embedding_model_info: sch.AzureModelInfo,
) -> tuple[sch.SQLConn, DBTableIndexer]:
    # Verify the connection
    conn_db = ConnectDB(conn_params=conn_params)
    await asyncio.to_thread(conn_db.verify_client_connection)
    if conn_params.schemas:
        conn_params.schemas = await conn_db.validate_schemas()
    # Create the alias name if it doesn't exist
    await create_alias_name(db=db, conn_params=conn_params)
    assert conn_params.database_name_alias
    # Save the vector db connection
    db_uuid = uuid.uuid4()
    # TODO: See if index_db_tables schema is necessary, seems like it isn't needed
    index_db_tables = DBTableIndexer(
        client_id=client_user.client_id,
        client_uuid=client_user.client_uuid,
        db_uuid=db_uuid,
        embedding_model_info=embedding_model_info,
    )
    vector_id = await setup_vector(db=db, client_id=client_user.client_id, index_db_tables=index_db_tables)
    # Save the db connection
    db_creds = await crud_connection.save_db_connection(
        db=db,
        client_id=client_user.client_id,
        db_uuid=db_uuid,
        conn_params=conn_params,
        vector_id=vector_id,
    )
    conn_uuid = str(copy.copy(db_creds.db_login.conn_uuid))
    db_id = copy.copy(db_creds.db_params.db_id)
    conn_id = copy.copy(db_creds.db_login.conn_id)
    sql_conn = sch.SQLConn(
        conn_uuid=conn_uuid,
        db_uuid=copy.copy(db_creds.db_params.db_uuid),
        database_name_alias=conn_params.database_name_alias,
        db_id=db_id,
        conn_id=conn_id,
    )
    return sql_conn, index_db_tables


async def run_ai_sql_query(
    db: AsyncSession,
    sql_query: str,
    conn_id: int,
    db_conn_params: sch.SQLDBSchema,
    client_conn_params: sch.SQLDBSchema,
    prompt_metadata: sch.PromptMetadata,
    chat_metadata: sch.ChatMetadata,
    agent: BaseAgent,
    client_id: int,
    small_model_info: sch.ModelInfo,
    redis_client_async: RedisAsync,
) -> str:
    handler = ChatMessageHandler(
        prompt_metadata=prompt_metadata, chat_metadata=chat_metadata, redis_client_async=redis_client_async
    )
    # TODO: Find a way to start running the query right away, but then still send the running sql query
    # in the correct order
    await asyncio.sleep(1.5)  # Adding so thoughts have time to come in from response hook
    await handler.create_message(
        db=db,
        role=sch.MessageRole.ASSISTANT,
        content="Running SQL Query...",
        msg_type=enums.MessageType.THOUGHT,
    )
    await handler.send_api_message()
    if chat_metadata.return_sql_in_thoughts:
        await handler.create_message(
            db=db,
            role=sch.MessageRole.ASSISTANT,
            content=f"```sql\n{sql_query}\n```",
            msg_type=enums.MessageType.THOUGHT,
        )
        await handler.send_api_message()
    mng_query = query.ClientQueryManager(
        db_conn_params=db_conn_params, client_conn_params=client_conn_params, sql_query=sql_query
    )
    query_result = await mng_query.run_client_query_and_upload(
        initial_prompt=prompt_metadata.initial_prompt, client_id=client_id, small_model_info=small_model_info
    )
    await handler.create_message(
        db=db,
        role=sch.MessageRole.ASSISTANT,
        content="The SQL query executed successfully",
        msg_type=enums.MessageType.THOUGHT,
    )
    await handler.send_api_message()
    logger.debug("Completed running the SQL query")
    # TODO: Consider creating a class with these result handling functions
    assert isinstance(query_result, sch.QueryResult)
    query_result_str = get_sql_result_prompt(
        conn_id=conn_id,
        query_result=query_result,
    )
    # If no result, then don't save a report
    if not query_result:
        agent.query_result = sch.MessageQueryResult(sql_query=sql_query)
    else:
        await save_query_results(
            db=db,
            agent=agent,
            query_result=query_result,
            sql_query=sql_query,
            prompt_metadata=prompt_metadata,
            chat_metadata=chat_metadata,
            query_result_str=query_result_str,
            conn_id=conn_id,
            small_model_info=small_model_info,
        )
    return query_result_str


async def save_query_results(
    db: AsyncSession,
    agent: BaseAgent,
    query_result: sch.QueryResult,
    sql_query: str,
    prompt_metadata: sch.PromptMetadata,
    chat_metadata: sch.ChatMetadata,
    query_result_str: str,
    conn_id: int,
    small_model_info: sch.ModelInfo,
) -> None:
    # Get the title
    extract = await get_title_description(
        db=db,
        prompt_metadata=prompt_metadata,
        sql_query=sql_query,
        query_result=query_result_str,
        small_model_info=small_model_info,
    )
    # Save to the DB
    result_history = await crud_result.save_result_history(
        db=db,
        chat_id=chat_metadata.chat_id,
        query_result=query_result,
        title=extract.title,
        subtitle=extract.subtitle,
        description=extract.description,
        conn_id=conn_id,
        prompt_metadata=prompt_metadata,
        chat_metadata=chat_metadata,
    )
    agent.query_result = sch.MessageQueryResult.from_orm(result_history)
    await db.commit()  # NOTE: Calling commit again to avoid idle in transaction


async def update_agent_tokens(agent: BaseAgent, max_tokens: int = 500):
    """Used to change the max tokens for the agent"""
    # Simple agent doesn't use prompt_agent, which is where the agent is set
    # TODO: Update agent to be optional
    if not isinstance(agent, SimpleAgent):
        agent.agent.memory.token_limit = agent.memory.get_llm_token_limit(llm=agent.agent_llm)  # type: ignore
        agent.agent.agent_worker._llm.max_tokens = max_tokens  # type: ignore
        logger.debug("Updated the agent to max_tokens = %s", max_tokens)


async def create_prompt_base(
    db: AsyncSession,
    client_user: sch.ClientUserInfo,
    prompt: str,
    return_visual_json: bool = True,
) -> sch.PromptMetadataBase:
    """Create prompt metadata before starting to interact with the Agent"""
    prompt_id, prompt_uuid = await crud_chat.create_prompt_history(
        db=db, client_id=client_user.client_id, llm_type=enums.LLMType.DATA_AGENT
    )
    prompt_metadata_base = sch.PromptMetadataBase(
        initial_prompt=prompt,
        user_uuid=client_user.user_uuid,
        user_id=client_user.user_id,
        client_uuid=client_user.client_uuid,
        client_id=client_user.client_id,
        prompt_uuid=prompt_uuid,
        prompt_id=prompt_id,
        llm_type=enums.LLMType.DATA_AGENT,
        prompt_time=datetime.now(),
        return_visual_json=return_visual_json,
        user_role=client_user.user_role,
    )
    return prompt_metadata_base


async def refresh_visual_result(
    db: AsyncSession,
    sql_engine: AsyncEngine,
    small_model_info: sch.ModelInfo,
    large_model_info: sch.ModelInfo,
    embedding_model_info: sch.AzureModelInfo,
    redis_client_async: RedisAsync,
    visual_result: models.VisualResultHistory,
    client_user: sch.ClientUserInfo,
) -> models.VisualResultHistory:
    """Refresh the visualization result"""
    # Create the prompt that includes the axis from the prior chart
    visual_info = extract_visual_info(visual_json=json.loads(visual_result.visual_json))  # type: ignore
    # TODO: This was part of the logic for inferring the chart type to provide to the LLM to improve charting
    # Need to revisit this
    # match = re.search(r"type\s*=\s*(\w+)", visual_info)
    # if match:
    #     chart_type_base = match.group(1)
    #     try:
    #         chart_type_obj = sch.ChartType(chart_type=chart_type_base)
    #         chart_type = chart_type_obj.chart_type
    #     except Exception as e:
    #         logger.error(f"{chart_type_base} is not a valid chart type. Here is the error: {str(e)}")
    #     logger.info(f"Chart type: {chart_type}")
    # else:
    #     msg = "No chart type found"
    #     logger.error(msg)
    #     raise Exception(msg)
    prompt = f"""You are refreshing a plot you previously created. You need to use the same axis titles as \
well as the same/similar axis ranges and/or format. Here is the visual information from the previous plot:
{visual_info}
"""
    logger.debug("Refresh visual result prompt: %s", visual_info)
    # Query the VisTool
    result_uuid = visual_result.result_uuid
    prompt_metadata_base = await create_prompt_base(
        db=db, client_user=client_user, prompt=prompt, return_visual_json=True
    )
    agent_setup = AgentSetup.load_from_prompt_metadata(prompt_metadata_base=prompt_metadata_base)
    base_agent = SimpleAgent(
        prompt_metadata=agent_setup.prompt_metadata,
        sql_engine=sql_engine,
        large_model_info=large_model_info,
        redis_client_async=redis_client_async,
    )
    vis_tool = VisTool(
        db=db, agent=base_agent, small_model_info=small_model_info, embedding_model_info=embedding_model_info
    )
    await vis_tool.get_plot(result_uuid=result_uuid, prompt=prompt)
    # Return the new visual result
    assert base_agent.query_result, "There should be a query result - check your code"
    assert base_agent.query_result.visual_result_uuid
    return await crud_result.get_visual_result(db=db, visual_result_uuid=base_agent.query_result.visual_result_uuid)


async def refresh_result(
    db: AsyncSession,
    result: models.ResultHistory,
    client_id: int,
    small_model_info: sch.ModelInfo,
    db_conn_params: sch.SQLDBSchema,
    commit: bool = True,
) -> Optional[models.ResultHistory]:
    db_conn = await crud_connection.get_db_conn_from_id(db=db, conn_id=result.result_conn_id)
    if not db_conn:
        logger.warning("Missing db conn")
        return None
    db_params = await db_conn.awaitable_attrs.database_params
    conn_db = await ConnectDB.get_db_conn(db_conn=db_conn, db_params=db_params)
    # Get the initial prompt
    initial_prompt = await crud_chat.get_initial_prompt_for_result(db=db, result_uuid=result.result_uuid)
    assert initial_prompt, "Missing chat history"
    mng_query = query.ClientQueryManager(
        db_conn_params=db_conn_params,
        client_conn_params=conn_db.conn_params,
        sql_query=result.sql_query,
        result_uuid=result.result_uuid,
    )
    query_result = await mng_query.run_client_query_and_upload(
        initial_prompt=initial_prompt, client_id=client_id, small_model_info=small_model_info
    )
    # Update record
    # TODO: Update this to use schemas instead
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
        await db.commit()
        await db.refresh(result)
        return result
    return None


def streamfile(file_path):
    try:
        with open(file_path, "rb") as file:
            yield from file
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        # You can decide what to do in this case. Here we just return to stop the generator.
        return
    except Exception as e:
        logger.error(f"Error opening file {file_path}: {e}")
        # You might want to handle other types of exceptions as well.
        return


def stream_s3_file(file_path):
    """Generator to stream a file from S3."""
    chunk_size = 1024 * 1024
    s3_key, bucket = upload.get_s3_info_from_filepath(file_path)
    try:
        # Fetch the file from S3
        s3_client = boto3.client("s3")
        response = s3_client.get_object(Bucket=bucket, Key=s3_key)
        file_stream = response["Body"]

        # Read and yield chunks of the file
        while True:
            chunk = file_stream.read(chunk_size)
            if not chunk:
                break
            yield chunk

    except Exception as e:
        logger.error(f"Error fetching file {s3_key} from S3 bucket {bucket}: {e}")
        # Handle error: You could raise an HTTPException or handle differently
        return


def get_file_generator_func(file_path) -> Callable:
    if S3_PREFIX in file_path:
        stream_func = stream_s3_file
    else:
        stream_func = streamfile
    return stream_func


async def calc_trust_score(db: AsyncSession, number_of_days: int = 7) -> sch.TrustScore:
    try:
        row = await crud_chat.get_thumb_reaction_counts(db=db, number_of_days=number_of_days)
    except NoResultFound:
        total_messages = 0
    total_messages = row.total_messages if row.total_messages else 0
    thumbs_down_count = row.thumbs_down_count if row.thumbs_down_count else 0

    trust_score = 1.00
    if total_messages > 0:
        trust_score = 1 - (thumbs_down_count / total_messages)

    trust_score_obj = sch.TrustScore(
        total_messages=total_messages, thumbs_down_count=thumbs_down_count, trust_score=round(trust_score, 2)
    )
    return trust_score_obj


async def create_database_from_existing_connection(
    db: AsyncSession,
    client_id: int,
    db_id: int,
    login_params: sch.CreateDBConn,
    sql_engine: AsyncEngine,
) -> sch.GetSQLConn:
    """Use existing database credentials to create a new connection"""

    # Get the database parameters
    database = await crud_connection.get_database_params_from_id(db=db, db_id=db_id)
    db_params = sch.DBParamsBytes.from_orm(database)
    decrypted_db_params = ConnectDB.decrypt_db(db_params.dict())

    # Get the connection parameters
    db_conn = sch.DBConnSchema.parse_obj(login_params)
    db_conn_dict = db_conn.dict()
    del db_conn_dict["schemas"]  # HACK: Improve schema management between database and connections
    conn_params = sch.SQLDBSchema(**decrypted_db_params, **db_conn_dict)

    # Set up a new connection
    return await setup_connection(
        db=db,
        client_id=client_id,
        conn_params=conn_params,
        login_params=login_params,
        db_id=db_id,
        sql_engine=sql_engine,
    )
