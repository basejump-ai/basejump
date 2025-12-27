"""Code for the MermaidJS Agent"""

from typing import Optional

from basejump.core.common.config.logconfig import set_logging
from basejump.core.models import enums
from basejump.core.models import schemas as sch
from basejump.core.service.base import BaseAgent
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms import ChatMessage
from llama_index.core.tools.types import AsyncBaseTool
from redis.asyncio import Redis as RedisAsync
from sqlalchemy.ext.asyncio import AsyncEngine

logger = set_logging(handler_option="stream", name=__name__)


class MermaidAgent(BaseAgent):
    """An AI Agent for generated MermaidJS ERD diagrams"""

    def __init__(
        self,
        prompt_metadata: sch.PromptMetadata,
        sql_engine: AsyncEngine,
        redis_client_async: RedisAsync,
        large_model_info: sch.ModelInfo,
        chat_history: Optional[list[ChatMessage]] = None,
        agent_llm: Optional[FunctionCallingLLM] = None,
        max_iterations: int = 10,
    ):
        super().__init__(
            prompt_metadata=prompt_metadata,
            chat_history=chat_history,
            max_iterations=max_iterations,
            agent_llm=agent_llm,
            sql_engine=sql_engine,
            redis_client_async=redis_client_async,
            large_model_info=large_model_info,
        )

    @staticmethod
    def get_llm_type() -> enums.LLMType:
        return enums.LLMType.MERMAID_AGENT

    async def setup_tools(self) -> list[AsyncBaseTool]:
        # NOTE: This can be overwritten with your own tools, such as validation using the minlag/mermaid-cli \
        # docker image
        return []

    async def _chat(self, prompt: str) -> sch.Message:
        logger.debug("Here is the prompt sent to the mermaid agent: %s", prompt)
        return await self._chat_base(prompt=prompt)

    async def retrieve_mermaidjs_diagram(self) -> str:
        """Use the AI to create a mermaidJS ERD diagram"""
        agent_output = await self.prompt_agent()
        # Extract the correct format
        return agent_output.content
