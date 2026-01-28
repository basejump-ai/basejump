import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

from redisvl.query.filter import Tag
from sqlalchemy.ext.asyncio import AsyncSession

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database import auth
from basejump.core.database.crud import crud_result
from basejump.core.database.result import store
from basejump.core.database.vector_utils import init_semcache
from basejump.core.models import constants, enums, prompts
from basejump.core.models import schemas as sch
from basejump.core.service.agents.results.refresh import ResultRefresher

logger = set_logging(handler_option="stream", name=__name__)


class SemanticMemory:
    def __init__(
        self,
        db: AsyncSession,
        service_context: sch.ServiceContext,
        prompt_metadata: sch.PromptMetadata,
        chat_metadata: sch.ChatMetadata,
        conn_params: sch.SQLDBSchema,
        query_result: Optional[sch.MessageQueryResult] = None,
        result_store: Optional[store.ResultStore] = None,
    ):
        self.db = db
        self.service_context = service_context
        self.prompt_metadata = prompt_metadata
        self.chat_metadata = chat_metadata
        self.conn_params = conn_params
        self.result_store = result_store or store.LocalResultStore(client_id=self.prompt_metadata.client_id)
        self.query_result = query_result or sch.MessageQueryResult()

    async def get_cached_prompt(
        self, prompt: str, distance_threshold: float, db_uuids: set[str], num_results=1
    ) -> list[dict[str, Any]]:
        try:
            # TODO: Determine why the semantic cache has issues initializing sometimes
            semcache_init_timeout = 10
            async with asyncio.timeout(semcache_init_timeout):
                llmcache = await init_semcache(
                    client_id=self.prompt_metadata.client_id,
                    redis_client_async=self.service_context.redis_client_async,
                )
        except TimeoutError:
            logger.warning(f"Connection to the semcache timed out after {semcache_init_timeout} seconds")
            return []
        client_id_filter = Tag("client_id") == str(self.prompt_metadata.client_id)
        db_uuid_filter = Tag("db_uuid") == db_uuids
        complex_filter = db_uuid_filter & client_id_filter
        return await llmcache.acheck(
            prompt=prompt,
            filter_expression=complex_filter,
            distance_threshold=distance_threshold,
            num_results=num_results,
        )

    async def get_cached_response(self, semcache_response: sch.SemCacheResponse) -> sch.MessagePair:
        # Get query result
        result = await crud_result.get_result(db=self.db, result_uuid=uuid.UUID(semcache_response.result_uuid))
        visual_result = await crud_result.get_visual_result_from_result(db=self.db, result_id=result.result_id)
        self.query_result = sch.MessageQueryResult.from_orm(result)
        if visual_result:
            self.query_result.visual_result_uuid = visual_result.visual_result_uuid
            self.query_result.visual_json = visual_result.visual_json
            self.query_result.visual_explanation = visual_result.visual_explanation

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
                    query_result=self.query_result,
                    metadata=sch.ResponseMetadata.parse_obj(semcache_response),
                ),
            )
        else:
            # Refresh the results
            refresher = ResultRefresher(
                db=self.db,
                prompt_metadata=self.prompt_metadata,
                service_context=self.service_context,
                conn_params=self.conn_params,
                result_store=self.result_store,
                query_result=self.query_result,
            )
            query_result = await refresher.refresh_results(result=result, visual_result=visual_result)
            # Get the response using SQL query results
            new_prompt_base = prompts.sql_result_prompt_basic(query_result=query_result)
            prompt = (
                f"""The user asked this question: {self.prompt_metadata.initial_prompt}. \
    This SQL query has been ran for you: {self.query_result.sql_query}. """
                + new_prompt_base
            )
            return sch.MessagePair(prompt=sch.ChatPrompt(prompt=prompt))

    async def check_cache(self, prompt, conn_uuids: set[str], db_uuids: set[str]) -> Optional[sch.MessagePair]:
        # See if a similar prompt has been cached
        semcache_response_raw = await self.get_cached_prompt(
            prompt=prompt, distance_threshold=constants.REDIS_SEMCACHE_EXACT_DISTANCE, db_uuids=db_uuids
        )
        if not semcache_response_raw:
            return None

        # Load cached response
        logger.info("Semantic similarity distance: %s", semcache_response_raw[0]["vector_distance"])
        metadata = semcache_response_raw[0]["metadata"]
        can_verify = auth.check_can_verify(
            required_role=enums.UserRoles(metadata["verified_user_role"]),
            user_role=enums.UserRoles(self.prompt_metadata.user_role),
        )
        semcache_response = sch.SemCacheResponse(
            response=semcache_response_raw[0]["response"],
            prompt=semcache_response_raw[0]["prompt"],
            vector_dist=semcache_response_raw[0]["vector_distance"],
            can_verify=can_verify,
            verified=True,
            **metadata,
        )
        self.chat_metadata.semcache_response = semcache_response  # save for later use in SQL query tool

        # Confirm permissions
        if metadata["conn_uuid"] not in conn_uuids:
            return None

        # Get the response
        return await self.get_cached_response(semcache_response=semcache_response)
