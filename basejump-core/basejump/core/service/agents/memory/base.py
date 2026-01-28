from abc import ABC, abstractmethod
from typing import Optional

from llama_index.core.base.llms.types import ChatMessage
from sqlalchemy.ext.asyncio import AsyncSession

from basejump.core.database.result import store
from basejump.core.models import models
from basejump.core.models import schemas as sch


class BaseAgentMemory(ABC):
    def __init__(
        self,
        db: AsyncSession,
        service_context: sch.ServiceContext,
        prompt_metadata: sch.PromptMetadata,
        chat_metadata: sch.ChatMetadata,
        conn_params: sch.SQLDBSchema,
        chat_history: Optional[list[ChatMessage]] = None,
        query_result: Optional[sch.MessageQueryResult] = None,
        result_store: Optional[store.ResultStore] = None,
    ):
        self.db = db
        self.service_context = service_context
        self.prompt_metadata = prompt_metadata
        self.chat_metadata = chat_metadata
        self.conn_params = conn_params
        self.chat_history = chat_history
        self.query_result = query_result
        self.result_store = result_store

    @abstractmethod
    def get_chat_history(self, chat: models.Chat, team_info: sch.TeamFields, system_prompt: Optional[str] = None):
        pass


class SimpleAgentMemory(BaseAgentMemory):
    def __init__(
        self,
        db: AsyncSession,
        service_context: sch.ServiceContext,
        prompt_metadata: sch.PromptMetadata,
        chat_metadata: sch.ChatMetadata,
        conn_params: sch.SQLDBSchema,
        chat_history: Optional[list[ChatMessage]] = None,
        query_result: Optional[sch.MessageQueryResult] = None,
        result_store: Optional[store.ResultStore] = None,
    ):
        self.db = db
        self.service_context = service_context
        self.prompt_metadata = prompt_metadata
        self.chat_metadata = chat_metadata
        self.conn_params = conn_params
        self.chat_history = chat_history
        self.query_result = query_result
        self.result_store = result_store

    def get_chat_history(chat: models.Chat, team_info: sch.TeamFields, system_prompt: Optional[str] = None):
        raise NotImplementedError("Get chat history not implemented for SimpleAgentMemory.")
