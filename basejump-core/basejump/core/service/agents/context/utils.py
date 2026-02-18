import uuid

from redis.asyncio import Redis as RedisAsync

from basejump.core.common.config.logconfig import set_logging
from basejump.core.models import schemas as sch
from basejump.core.models.ai import formatter
from basejump.core.service.agents.memory.semantic import SemanticMemory

logger = set_logging(handler_option="stream", name=__name__)


async def upload_sql_query_example(
    sql_query: str,
    client_user: sch.ClientUserInfo,
    prompt: str,
    db_uuid: uuid.UUID,
    redis_client_async: RedisAsync,
    small_model_info: sch.ModelInfo,
    recent_interactions: list[sch.MessagePair] = [],
):
    if recent_interactions:
        prompt = await formatter.contextualize_prompt(
            recent_interactions=recent_interactions, small_model_info=small_model_info
        )
    semantic_memory = SemanticMemory(client_id=client_user.client_id, redis_client_async=redis_client_async)

    metadata = sch.SemCacheMetadata(
        result_uuid="",
        prompt_uuid="",
        verified_user_role=client_user.user_role,
        verified_user_uuid=str(client_user.user_uuid),
        sql_query=sql_query,
        timestamp="",
        conn_uuid="",
    )
    await semantic_memory.store(prompt=prompt, response="", metadata=metadata, db_uuid=db_uuid)
    logger.debug("Uploaded sql query to semantic memory: %s", sql_query)
