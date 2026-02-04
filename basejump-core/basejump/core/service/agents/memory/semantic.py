import asyncio
import uuid
from typing import Optional

from redis.asyncio import Redis as RedisAsync
from redisvl.query.filter import Tag
from sqlalchemy.ext.asyncio import AsyncSession

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.vector_utils import (
    AsyncSemanticCache,
    init_semcache,
    update_verified_result_vectors,
)
from basejump.core.models import models
from basejump.core.models import schemas as sch

logger = set_logging(handler_option="stream", name=__name__)


class SemanticMemory:
    def __init__(
        self,
        redis_client_async: RedisAsync,
    ):
        self.redis_client_async = redis_client_async
        self.cache: Optional[AsyncSemanticCache] = None

    async def get_cached_prompts(
        self,
        prompt: str,
        client_id: int,
        db_uuid: str,
        num_results=1,
        distance_threshold: Optional[float] = None,
    ) -> list[sch.SemCacheResponse]:
        if not self.cache:
            try:
                # TODO: Determine why the semantic cache has issues initializing sometimes
                semcache_init_timeout = 60
                async with asyncio.timeout(semcache_init_timeout):
                    self.cache = await init_semcache(
                        client_id=client_id,
                        redis_client_async=self.redis_client_async,
                    )
            except TimeoutError:
                logger.warning(f"Connection to the semcache timed out after {semcache_init_timeout} seconds")
                return []
        client_id_filter = Tag("client_id") == str(client_id)
        db_uuid_filter = Tag("db_uuid") == db_uuid
        complex_filter = db_uuid_filter & client_id_filter
        semcache_responses = await self.cache.acheck(
            prompt=prompt,
            filter_expression=complex_filter,
            distance_threshold=distance_threshold,
            num_results=num_results,
        )
        if not semcache_responses:
            return []
        return_semcache_responses = []
        for semcache_response in semcache_responses:
            metadata = semcache_response["metadata"]
            semcache_obj = sch.SemCacheResponse(
                response=semcache_response["response"],
                prompt=semcache_response["prompt"],
                vector_dist=semcache_response["vector_distance"],
                verified=True,
                **metadata,
            )
            return_semcache_responses.append(semcache_obj)
        return return_semcache_responses

    async def store(
        self,
        db: AsyncSession,
        prompt_uuid: uuid.UUID,
        result: models.ResultHistory,
        response: str,
        client_user: sch.ClientUserInfo,
        conn_uuid: uuid.UUID,
        db_uuid: uuid.UUID,
    ) -> None:
        await update_verified_result_vectors(
            db=db,
            result=result,
            prompt_uuid=prompt_uuid,
            content=response,
            verified=True,
            client_user=client_user,
            conn_uuid=conn_uuid,
            db_uuid=db_uuid,
            redis_client_async=self.redis_client_async,
        )
