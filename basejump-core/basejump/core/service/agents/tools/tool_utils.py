from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.connector import Connector
from basejump.core.models import models
from basejump.core.models import schemas as sch
from basejump.core.service.agents.base import BaseAgent

logger = set_logging(handler_option="stream", name=__name__)


async def update_agent_tokens(agent: BaseAgent, max_tokens: int = 500):
    """Used to change the max tokens for the agent"""
    # Simple agent doesn't use prompt_agent, which is where the agent is set
    # TODO: Update agent to be optional
    from basejump.core.service.agents.simple import SimpleAgent

    if not isinstance(agent, SimpleAgent):
        agent.agent.memory.token_limit = agent.agent.memory.get_llm_token_limit(llm=agent.agent_llm)  # type: ignore
        agent.agent.agent_worker._llm.max_tokens = max_tokens  # type: ignore
        logger.debug("Updated the agent to max_tokens = %s", max_tokens)


async def get_sql_tool_context(service_context: sch.ServiceContext, conn: models.DBConn) -> sch.SQLToolContext:
    conn_db = await Connector.get_db_conn(db_conn=conn, db_params=conn.database_params)
    return sch.SQLToolContext(
        client_conn_params=conn_db.conn_params,
        conn_id=conn.conn_id,
        conn_uuid=str(conn.conn_uuid),
        db_id=conn.db_id,
        db_uuid=str(conn.database_params.db_uuid),
        vector_id=conn.database_params.vector_id,
        service_context=service_context,
    )
