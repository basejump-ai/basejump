"""Code for the MermaidJS Agent"""

from typing import Optional

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools.types import AsyncBaseTool

from basejump.core.common.config.logconfig import set_logging
from basejump.core.models import enums
from basejump.core.models import schemas as sch
from basejump.core.service.agents.base import BaseAgent
from basejump.core.service.agents.memory.agent import SimpleAgentMemory

logger = set_logging(handler_option="stream", name=__name__)


class MermaidAgent(BaseAgent):
    """An AI Agent for generated MermaidJS ERD diagrams"""

    def __init__(
        self,
        prompt_metadata: sch.PromptMetadata,
        service_context: sch.ServiceContext,
        memory: Optional[SimpleAgentMemory] = None,
        chat_history: Optional[list[ChatMessage]] = None,
        llm: Optional[FunctionCallingLLM] = None,
        max_iterations: int = 10,
    ):
        if memory and chat_history:
            logger.warning(
                """Both 'memory' and 'chat_history' were provided. The 'chat_history' parameter \
will be ignored; using memory's chat history instead."""
            )
        if not memory:
            memory = SimpleAgentMemory(
                service_context=service_context,
                prompt_metadata=prompt_metadata,
                chat_history=chat_history,
            )
        super().__init__(
            prompt_metadata=prompt_metadata,
            max_iterations=max_iterations,
            llm=llm,
            service_context=service_context,
            memory=memory,
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
