from basejump.core.common.config.logconfig import set_logging
from basejump.core.service.base import BaseAgent, SimpleAgent

logger = set_logging(handler_option="stream", name=__name__)


async def update_agent_tokens(agent: BaseAgent, max_tokens: int = 500):
    """Used to change the max tokens for the agent"""
    # Simple agent doesn't use prompt_agent, which is where the agent is set
    # TODO: Update agent to be optional
    if not isinstance(agent, SimpleAgent):
        agent.agent.memory.token_limit = agent.memory.get_llm_token_limit(llm=agent.agent_llm)  # type: ignore
        agent.agent.agent_worker._llm.max_tokens = max_tokens  # type: ignore
        logger.debug("Updated the agent to max_tokens = %s", max_tokens)
