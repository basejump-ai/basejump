import json
from datetime import datetime
from typing import Optional

from llama_index.core.llms.function_calling import FunctionCallingLLM
from sqlalchemy.ext.asyncio import AsyncSession

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.client import query
from basejump.core.database.connector import Connector
from basejump.core.database.crud import crud_chat, crud_connection, crud_result
from basejump.core.database.db_utils import extract_visual_info
from basejump.core.database.result import store
from basejump.core.models import constants, models
from basejump.core.models import schemas as sch
from basejump.core.service.agents.tools.visualize import VisTool

logger = set_logging(handler_option="stream", name=__name__)


async def refresh_result(
    db: AsyncSession,
    service_context: sch.ServiceContext,
    client_user: sch.ClientUserInfo,
    result_store: store.ResultStore,
    result: models.ResultHistory,
    # HACK: Fix this
    commit: bool = True,
) -> Optional[models.ResultHistory]:
    db_conn = await crud_connection.get_db_conn_from_id(db=db, conn_id=result.result_conn_id)
    if not db_conn:
        logger.warning("Missing db conn")
        return None
    db_params = await db_conn.awaitable_attrs.database_params
    conn_db = await Connector.get_db_conn(db_conn=db_conn, db_params=db_params)
    # Get the initial prompt
    initial_prompt = await crud_chat.get_initial_prompt_for_result(db=db, result_uuid=result.result_uuid)
    assert initial_prompt, "Missing chat history"
    result_store.result_uuid = result.result_uuid
    async with query.ClientQueryRecorder(
        client_conn_params=conn_db.conn_params,
        sql_query=result.sql_query,
        initial_prompt=initial_prompt,
        client_id=client_user.client_id,
        small_model_info=service_context.small_model_info,
        result_store=result_store,
    ) as query_recorder:
        query_result = await query_recorder.astore_query_result()
    # Update record
    # TODO: Update this to use schemas instead
    # HACK: Fix this
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


async def refresh_visual_result(
    db: AsyncSession,
    llm: FunctionCallingLLM,
    service_context: sch.ServiceContext,
    visual_result: models.VisualResultHistory,
    client_user: sch.ClientUserInfo,
    result_store: store.ResultStore,
    query_result: Optional[sch.MessageQueryResult] = None,
) -> models.VisualResultHistory:
    """Refresh the visualization result"""

    if not query_result:
        query_result = sch.MessageQueryResult()
    # Create the prompt that includes the axis from the prior chart
    visual_info = extract_visual_info(visual_json=json.loads(visual_result.visual_json))  # type: ignore
    prompt = f"""You are refreshing a plot you previously created. You need to use the same axis titles as \
well as the same/similar axis ranges and/or format. Here is the visual information from the previous plot:
{visual_info}
"""
    logger.debug("Refresh visual result prompt: %s", visual_info)
    # Query the VisTool
    result_uuid = visual_result.result_uuid
    vis_tool = VisTool(
        llm=llm,
        db=db,
        query_result=query_result,
        service_context=service_context,
        user_uuid=visual_result.author_user_uuid,
        parent_msg_uuid=visual_result.parent_msg_uuid,
        result_store=result_store,
    )

    await vis_tool.get_plot(result_uuid=result_uuid, prompt=prompt)
    # Return the new visual result
    assert query_result, "There should be a query result - check your code"
    assert query_result.visual_result_uuid
    return await crud_result.get_visual_result(db=db, visual_result_uuid=query_result.visual_result_uuid)


async def refresh_results(
    db: AsyncSession,
    result: models.ResultHistory,
    result_store: store.ResultStore,
    service_context: sch.ServiceContext,
    client_user: sch.ClientUserInfo,
    visual_result: Optional[models.VisualResultHistory] = None,
) -> sch.QueryResult:
    """Refreshes results for both visual and SQL sources"""
    await refresh_result(
        db=db,
        result_store=result_store,
        result=result,
        service_context=service_context,
        client_user=client_user,
        commit=False,
    )
    result_manager = result_store.get_result_manager(result.result_file_path)
    file_gen_func = result_manager.get_stream_result_generator()
    stream_gen = file_gen_func()
    rows_base = next(stream_gen)
    rows = [tuple(row.split(",")) for row in rows_base.decode("utf-8").splitlines()]
    message_query_result = sch.MessageQueryResult.from_orm(result)
    query_res = sch.QueryResult(
        query_result=rows[: constants.AI_RESULT_PREVIEW_CT],
        preview_row_ct=constants.AI_RESULT_PREVIEW_CT,
        num_rows=result.row_num_total,
        num_cols=1,  # just a placeholder since it isn't used in the prompt
        result_type=result.result_type,
        sql_query=result.sql_query,
        result_uuid=str(result.result_uuid),
        # TODO: Clean up schema objs with preview row ct (they are redundant)
        ai_preview_row_ct=constants.AI_RESULT_PREVIEW_CT,
        result_file_path=result.result_file_path,
        preview_file_path=result.preview_file_path,
        message_query_result=message_query_result,
    )

    if visual_result:
        visual_result = await refresh_visual_result(
            db=db,
            service_context=service_context,
            visual_result=visual_result,
            client_user=client_user,
            result_store=result_store,
        )
        message_query_result.visual_result_uuid = visual_result.visual_result_uuid
        message_query_result.visual_json = visual_result.visual_json
        message_query_result.visual_explanation = visual_result.visual_explanation
    return query_res
