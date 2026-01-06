import asyncio

from redis.asyncio import Redis as RedisAsync
from sqlalchemy.ext.asyncio import AsyncSession

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.client import query
from basejump.core.database.crud import crud_result
from basejump.core.database.result import store
from basejump.core.models import enums
from basejump.core.models import schemas as sch
from basejump.core.models.ai.formatter import get_title_description
from basejump.core.models.prompts import get_sql_result_prompt
from basejump.core.service.base import BaseAgent, ChatMessageHandler, SimpleAgent

logger = set_logging(handler_option="stream", name=__name__)


async def update_agent_tokens(agent: BaseAgent, max_tokens: int = 500):
    """Used to change the max tokens for the agent"""
    # Simple agent doesn't use prompt_agent, which is where the agent is set
    # TODO: Update agent to be optional
    if not isinstance(agent, SimpleAgent):
        agent.agent.memory.token_limit = agent.memory.get_llm_token_limit(llm=agent.agent_llm)  # type: ignore
        agent.agent.agent_worker._llm.max_tokens = max_tokens  # type: ignore
        logger.debug("Updated the agent to max_tokens = %s", max_tokens)


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
    result_store: store.ResultStore,
    verbose: bool = False,
) -> str:
    handler = ChatMessageHandler(
        prompt_metadata=prompt_metadata,
        chat_metadata=chat_metadata,
        redis_client_async=redis_client_async,
        verbose=verbose,
    )
    # TODO: Find a way to start running the query right away, but then still send the running sql query
    # in the correct order
    await asyncio.sleep(1.5)  # Adding so thoughts have time to come in from response hook
    running_query_msg = "Running SQL Query..."
    await handler.create_message(
        db=db,
        role=sch.MessageRole.ASSISTANT,
        content=running_query_msg,
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
    async with query.ClientQueryRecorder(
        client_conn_params=client_conn_params,
        sql_query=sql_query,
        initial_prompt=prompt_metadata.initial_prompt,
        client_id=client_id,
        small_model_info=small_model_info,
        result_store=result_store,
    ) as query_recorder:
        logger.info(running_query_msg)
        query_result = await query_recorder.astore_query_result()
    await handler.create_message(
        db=db,
        role=sch.MessageRole.ASSISTANT,
        content="The SQL query executed successfully",
        msg_type=enums.MessageType.THOUGHT,
    )
    await handler.send_api_message()
    logger.info("Completed running the SQL query")
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
