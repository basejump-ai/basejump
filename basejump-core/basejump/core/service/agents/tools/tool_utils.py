from typing import Optional

from basejump.core.common.config.logconfig import set_logging
from basejump.core.models import constants, models
from basejump.core.models import schemas as sch
from basejump.core.service.agents.base import BaseAgent
from basejump.core.service.agents.tools.sql.base import SQLTool
from basejump.core.service.agents.tools.visualize import VisTool

logger = set_logging(handler_option="stream", name=__name__)


async def update_agent_tokens(agent: BaseAgent, max_tokens: int = 500):
    """Used to change the max tokens for the agent"""
    # Simple agent doesn't use prompt_agent, which is where the agent is set
    # TODO: Update agent to be optional
    from basejump.core.service.agents.simple import SimpleAgent

    if not isinstance(agent, SimpleAgent):
        agent.agent.memory.token_limit = agent.memory.get_llm_token_limit(llm=agent.agent_llm)  # type: ignore
        agent.agent.agent_worker._llm.max_tokens = max_tokens  # type: ignore
        logger.debug("Updated the agent to max_tokens = %s", max_tokens)


async def refresh_results(
    sql_tool: SQLTool,
    vis_tool: VisTool,
    result: models.ResultHistory,
    visual_result: Optional[models.VisualResultHistory] = None,
) -> sch.QueryResult:
    """Refreshes results from both the SQLTool and VisualTool"""
    await sql_tool.refresh(
        result=result,
        commit=False,
    )
    result_manager = sql_tool.result_store.get_result_manager(result.result_file_path)
    file_gen_func = result_manager.get_stream_result_generator()
    stream_gen = file_gen_func()
    rows_base = next(stream_gen)
    rows = [tuple(row.split(",")) for row in rows_base.decode("utf-8").splitlines()]
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
    )
    sql_tool.agent.query_result = sch.MessageQueryResult.from_orm(result)
    if visual_result:
        visual_result = await vis_tool.refresh(result=visual_result)
        sql_tool.agent.query_result.visual_result_uuid = visual_result.visual_result_uuid
        sql_tool.agent.query_result.visual_json = visual_result.visual_json
        sql_tool.agent.query_result.visual_explanation = visual_result.visual_explanation
    return query_res
