from redis.asyncio import Redis as RedisAsync
from sqlalchemy.ext.asyncio import AsyncSession

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.crud import crud_chat, crud_connection
from basejump.core.database.crud.crud_utils import create_callback_mgrs
from basejump.core.models import errors, models
from basejump.core.models import schemas as sch

logger = set_logging(handler_option="stream", name=__name__)


class AgentSetup:
    def __init__(self, prompt_metadata: sch.PromptMetadata):
        """
        Setup methods for agents
        """
        self.prompt_metadata = prompt_metadata

    @staticmethod
    def _load_from_prompt_metadata(prompt_metadata_base: sch.PromptMetadataBase):
        callback_managers = create_callback_mgrs(prompt_metadata_base.model_name)

        # NOTE: Re-instantiating prompt metadata here since this is background submitted
        prompt_metadata = sch.PromptMetadata(
            **prompt_metadata_base.dict(),
            token_counter=callback_managers.token_counter,
            llama_debug=callback_managers.llama_debug,
            callback_manager=callback_managers.callback_manager,
        )
        return prompt_metadata

    @classmethod
    def load_from_prompt_metadata(cls, prompt_metadata_base: sch.PromptMetadataBase):
        prompt_metadata = cls._load_from_prompt_metadata(prompt_metadata_base=prompt_metadata_base)
        return cls(prompt_metadata=prompt_metadata)


class ChatAgentSetup(AgentSetup):
    def __init__(
        self,
        db: AsyncSession,
        embedding_model_info: sch.AzureModelInfo,
        prompt_metadata: sch.PromptMetadata,
        chat_metadata: sch.ChatMetadata,
        redis_client_async: RedisAsync,
    ):
        """
        Setup methods for chat agents
        """
        super().__init__(prompt_metadata=prompt_metadata)
        self.db = db
        self.chat_metadata = chat_metadata
        self.redis_client_async = redis_client_async
        self.embedding_model_info = embedding_model_info

    # TODO: Review if this should be an instance method instead
    @classmethod
    async def load_from_metadata(
        cls,
        db: AsyncSession,
        prompt_metadata_base: sch.PromptMetadataBase,
        chat_metadata: sch.ChatMetadata,
        redis_client_async: RedisAsync,
        embedding_model_info: sch.AzureModelInfo,
    ):
        prompt_metadata = cls._load_from_prompt_metadata(prompt_metadata_base=prompt_metadata_base)
        cls.chat_metadata = chat_metadata
        return cls(
            prompt_metadata=prompt_metadata,
            chat_metadata=chat_metadata,
            db=db,
            redis_client_async=redis_client_async,
            embedding_model_info=embedding_model_info,
        )

    @staticmethod
    async def get_connections(db: AsyncSession, team_id: int, user_id: int) -> list[models.Connection]:
        # Get the connections available for the AI
        try:
            connections = await crud_connection.get_team_connections(db=db, team_id=team_id, user_id=user_id)
        except Exception as e:
            logger.error(e)
            raise errors.GetTeamConnError
        return connections

    async def get_chat(self) -> models.Chat:
        chat = await crud_chat.get_chat(
            db=self.db, chat_uuid=self.chat_metadata.chat_uuid, user_id=self.prompt_metadata.user_id
        )
        try:
            assert chat
        except AssertionError:
            raise errors.ChatUUIDNotFound
        if not chat.chat_name:
            chat.chat_name = self.prompt_metadata.initial_prompt
        return chat
