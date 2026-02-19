from abc import ABC
from typing import Optional

from llama_index.core.llms import ChatMessage

from basejump.core.database.result import store
from basejump.core.models import schemas as sch


class BaseAgentMemory(ABC):
    def __init__(
        self,
        service_context: sch.ServiceContext,
        prompt_metadata: sch.PromptMetadata,
        chat_history: Optional[list[ChatMessage]] = None,
        query_result: Optional[sch.MessageQueryResult] = None,
        result_store: Optional[store.ResultStore] = None,
    ):
        self.service_context = service_context
        self.prompt_metadata = prompt_metadata
        self.chat_history = chat_history or []
        self.query_result = query_result or sch.MessageQueryResult()
        self.result_store = result_store or store.LocalResultStore(client_id=self.prompt_metadata.client_id)
