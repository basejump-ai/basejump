import asyncio
import uuid
from typing import Optional

from llama_index.vector_stores.redis import TokenEscaper
from redis.asyncio import Redis as RedisAsync
from redis.commands.search.query import Query
from redisvl.query.filter import Tag

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.vector.utils import AsyncSemanticCache, get_index_name
from basejump.core.models import schemas as sch

logger = set_logging(handler_option="stream", name=__name__)
REDIS_SEMCACHE_PREFIX = "semcache_"


class SemanticMemory:
    def __init__(
        self,
        client_id: int,
        redis_client_async: RedisAsync,
    ):
        self.redis_client_async = redis_client_async
        self.client_id = client_id
        self.cache: Optional[AsyncSemanticCache] = None

    def get_index_name(self) -> str:
        idx_nm = get_index_name(client_id=self.client_id)
        return REDIS_SEMCACHE_PREFIX + idx_nm

    @property
    def index_name(self) -> str:
        return self.get_index_name()

    async def setup(self) -> AsyncSemanticCache:
        llmcache = await AsyncSemanticCache.setup(
            name=self.index_name,
            redis_client=self.redis_client_async,
            filterable_fields=[
                {"name": "client_id", "type": "tag"},
                {"name": "result_uuid", "type": "tag"},
                {"name": "db_uuid", "type": "tag"},
            ],
        )
        return llmcache

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
                    self.cache = await self.setup()
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
        prompt: str,
        response: str,
        metadata: sch.SemCacheMetadata,
        db_uuid: uuid.UUID,
    ) -> None:
        llmcache = await self.setup()
        await llmcache.astore(
            prompt=prompt,
            response=response,
            metadata=metadata.model_dump(),
            filters={
                "client_id": str(self.client_id),
                "result_uuid": str(metadata.result_uuid),
                "db_uuid": str(db_uuid),
            },
        )
        logger.info("Stored prompt in semantic cache: %s", prompt)

    async def delete(self, result_uuid: uuid.UUID):
        token_escaper = TokenEscaper()
        result_uuid_esc = token_escaper.escape(str(result_uuid))
        search_str = f"@result_uuid:{{{result_uuid_esc}}}"
        try:
            idx_result = await self.redis_client_async.ft(self.index_name).search(Query(search_str))
            doc_id = idx_result.docs[0].id
            await self.redis_client_async.delete(doc_id)
            logger.info("Deleted sem cache for result: %s", str(result_uuid))
        except Exception:
            logger.debug(f"No sem cache result found for {str(result_uuid)}, skipping")
