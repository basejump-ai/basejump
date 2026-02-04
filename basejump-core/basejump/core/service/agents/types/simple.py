from typing import Optional, Sequence

from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools.types import AsyncBaseTool

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.result import store
from basejump.core.models import enums
from basejump.core.models import schemas as sch
from basejump.core.service.agents.memory.agent import SimpleAgentMemory
from basejump.core.service.agents.types.base import BaseAgent

logger = set_logging(handler_option="stream", name=__name__)


class SimpleAgent(BaseAgent):
    """An AI Agent with the bare minimum"""

    def __init__(
        self,
        prompt_metadata: sch.PromptMetadata,
        service_context: sch.ServiceContext,
        result_store: Optional[store.ResultStore] = None,
        memory: Optional[SimpleAgentMemory] = None,
        llm: Optional[FunctionCallingLLM] = None,
        max_iterations: int = 10,
        verbose: bool = False,
    ):
        if not memory:
            memory = SimpleAgentMemory(
                service_context=service_context,
                prompt_metadata=prompt_metadata,
            )
        super().__init__(
            prompt_metadata=prompt_metadata,
            memory=memory,
            max_iterations=max_iterations,
            llm=llm,
            service_context=service_context,
            verbose=verbose,
        )
        self.result_store = result_store or store.LocalResultStore(client_id=self.prompt_metadata.client_id)

    @staticmethod
    def get_llm_type() -> enums.LLMType:
        return enums.LLMType.SIMPLE_AGENT

    async def setup_tools(self) -> Sequence[AsyncBaseTool]:
        return []

    async def _chat(self, prompt: str) -> sch.Message:
        logger.debug("Here is the prompt sent to the simple agent: %s", prompt)
        return await self._chat_base(prompt=prompt)
