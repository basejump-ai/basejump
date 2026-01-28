import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Optional

from redis.asyncio import Redis as RedisAsync
from redisvl.query.filter import Tag
from sqlalchemy.ext.asyncio import AsyncSession

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.crud import crud_result
from basejump.core.database.result.store import ResultStore
from basejump.core.database.vector_utils import (
    AsyncSemanticCache,
    init_semcache,
    update_verified_result_vectors,
)
from basejump.core.models import models, prompts
from basejump.core.models import schemas as sch
from basejump.core.service.agents.memory.utils import refresh_results

logger = set_logging(handler_option="stream", name=__name__)


class SemanticMemory:
    def __init__(
        self,
        redis_client_async: RedisAsync,
    ):
        self.redis_client_async = redis_client_async
        self.cache: Optional[AsyncSemanticCache] = None

    async def get_cached_prompt(
        self, prompt: str, client_id: int, distance_threshold: float, db_uuids: set[str], num_results=1
    ) -> Optional[sch.SemCacheResponse]:
        if not self.cache:
            try:
                # TODO: Determine why the semantic cache has issues initializing sometimes
                semcache_init_timeout = 10
                async with asyncio.timeout(semcache_init_timeout):
                    self.cache = await init_semcache(
                        client_id=client_id,
                        redis_client_async=self.redis_client_async,
                    )
            except TimeoutError:
                logger.warning(f"Connection to the semcache timed out after {semcache_init_timeout} seconds")
                return None
        client_id_filter = Tag("client_id") == str(client_id)
        db_uuid_filter = Tag("db_uuid") == db_uuids
        complex_filter = db_uuid_filter & client_id_filter
        semcache_response_raw = await self.cache.acheck(
            prompt=prompt,
            filter_expression=complex_filter,
            distance_threshold=distance_threshold,
            num_results=num_results,
        )
        metadata = semcache_response_raw[0]["metadata"]
        semcache_response = sch.SemCacheResponse(
            response=semcache_response_raw[0]["response"],
            prompt=semcache_response_raw[0]["prompt"],
            vector_dist=semcache_response_raw[0]["vector_distance"],
            verified=True,
            **metadata,
        )
        return semcache_response

    async def get_cached_response(
        self,
        prompt: str,
        db: AsyncSession,
        service_context: sch.ServiceContext,
        result_store: ResultStore,
        client_id: int,
        semcache_response: sch.SemCacheResponse,
    ) -> sch.MessagePair:
        # Get query result
        result = await crud_result.get_result(db=db, result_uuid=uuid.UUID(semcache_response.result_uuid))
        visual_result = await crud_result.get_visual_result_from_result(db=db, result_id=result.result_id)
        query_result = sch.MessageQueryResult.from_orm(result)
        if visual_result:
            query_result.visual_result_uuid = visual_result.visual_result_uuid
            query_result.visual_json = visual_result.visual_json
            query_result.visual_explanation = visual_result.visual_explanation

        # Return recent response
        cached_response_timestamp = datetime.strptime(semcache_response.timestamp, "%Y-%m-%d %H:%M:%S.%f%z")
        refresh_not_needed = datetime.now(cached_response_timestamp.tzinfo) - cached_response_timestamp < timedelta(
            days=1
        )
        if refresh_not_needed:
            # Create a message to return
            logger.info("Cached message found - returning cached message.")
            return sch.MessagePair(
                prompt=sch.ChatPrompt(prompt=semcache_response.prompt),
                response=sch.ChatResponse(
                    response=semcache_response.response,
                    query_result=query_result,
                    metadata=sch.ResponseMetadata.parse_obj(semcache_response),
                ),
            )
        else:
            # Refresh the results
            query_result = await refresh_results(
                db=db,
                service_context=service_context,
                client_id=client_id,
                result=result,
                result_store=result_store,
                visual_result=visual_result,
                query_result=query_result,
            )
            # Get the response using SQL query results
            new_prompt_base = prompts.sql_result_prompt_basic(query_result=query_result)
            prompt = (
                f"""The user asked this question: {prompt}. \
    This SQL query has been ran for you: {query_result.sql_query}. """
                + new_prompt_base
            )
            return sch.MessagePair(prompt=sch.ChatPrompt(prompt=prompt))

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
