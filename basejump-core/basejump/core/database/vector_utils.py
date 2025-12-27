import json
import uuid
from typing import Any, Dict, List, Optional

from basejump.core.common.config.logconfig import set_logging
from basejump.core.models import constants, enums, models
from basejump.core.models import schemas as sch
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.indices.base import BaseIndex
from llama_index.vector_stores.redis import RedisVectorStore, TokenEscaper
from redis.asyncio import Redis as RedisAsync
from redis.commands.search.query import Query
from redisvl.extensions.constants import (
    CACHE_VECTOR_FIELD_NAME,
    ENTRY_ID_FIELD_NAME,
    INSERTED_AT_FIELD_NAME,
    METADATA_FIELD_NAME,
    PROMPT_FIELD_NAME,
    RESPONSE_FIELD_NAME,
    UPDATED_AT_FIELD_NAME,
)
from redisvl.extensions.llmcache.base import BaseLLMCache
from redisvl.extensions.llmcache.schema import SemanticCacheIndexSchema
from redisvl.extensions.llmcache.semantic import SemanticCache
from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.schema import IndexSchema
from redisvl.utils.utils import validate_vector_dims
from redisvl.utils.vectorize import BaseVectorizer, HFTextVectorizer
from sqlalchemy.ext.asyncio import AsyncSession

logger = set_logging(handler_option="stream", name=__name__)

REDIS_PARTITION_PREFIX = "basejump_pclient"
REDIS_SEMCACHE_PREFIX = "semcache_"
# WARNING: Changing this will change the index location.
# To change this number, add this value to the connect.vector_db table.
# Then the modulo can be calculated based off of that instead of this constant
REDIS_INDEX_CT = 100


def get_index_schema(index_name: str) -> IndexSchema:
    """Create a centralized index schema definition"""
    schema = IndexSchema.from_dict(
        {
            "index": {"name": index_name, "prefix": index_name + "/vector"},
            "fields": [
                # Required fields
                {"name": "id", "type": "tag"},
                {"name": "doc_id", "type": "tag"},
                {"name": "text", "type": "text"},
                {"name": "vector", "type": "vector", "attrs": {"dims": 1536, "algorithm": "flat"}},
                *constants.VECTOR_FILTERS,
            ],
        }
    )
    return schema


def get_index_name(client_id: int) -> str:
    """Gets the index name

    Warnings
    -------
    Keep this as a suffix unless changing ManageVectorIndexes.get_redis_indexes since that function
    depends on this one appending the type to the end.
    """
    # TODO: Use this function in ManageVectorIndexes.get_redis_indexes so there is a more clear dependency
    # index_name = (str(vector_uuid) + vector_datasource_type.value).lower()
    modulo_result = int(client_id) % int(REDIS_INDEX_CT)
    return REDIS_PARTITION_PREFIX + str(modulo_result)


def get_semcache_index_name(client_id: int) -> str:
    idx_nm = get_index_name(client_id=client_id)
    return REDIS_SEMCACHE_PREFIX + idx_nm


async def get_table_info_from_vector_db(
    index_name: str, tbl_uuids: list[uuid.UUID], start: int, offset: int, redis_client_async: RedisAsync
) -> str:
    # TODO: Query only the relevant tables
    tbl_uuids_str = "|".join([str(uuid) for uuid in tbl_uuids])
    token_escaper = TokenEscaper()
    tbl_uuids_esc = token_escaper.escape(tbl_uuids_str)
    search_str = f"@id:{{{tbl_uuids_esc}}}"
    logger.debug(f"Redis search using index name: {index_name} \n Redis search using search str: {search_str}")
    index = await redis_client_async.ft(index_name).search(
        Query(search_str).return_field("_node_content").paging(start, offset)
    )
    db_table_info = ""
    for idx, doc in enumerate(index.docs):
        node_content = json.loads(doc._node_content)
        if node_content["metadata"].get("table_info"):
            if idx < index.total:
                db_table_info += node_content["metadata"]["table_info"] + "\n"
            else:
                db_table_info += node_content["metadata"]["table_info"]

    return db_table_info


async def delete_nodes(client_id: int, node_uuids: list[uuid.UUID], redis_client_async: RedisAsync):
    index_name = get_index_name(client_id=client_id)
    schema = IndexSchema.from_dict(
        {
            "index": {"name": index_name, "prefix": index_name + "/vector"},
            "fields": [
                # Required fields
                {"name": "id", "type": "tag"},
                {"name": "doc_id", "type": "tag"},
                {"name": "text", "type": "text"},
                {"name": "vector", "type": "vector", "attrs": {"dims": 1536, "algorithm": "flat"}},
                *constants.VECTOR_FILTERS,
            ],
        }
    )
    vector_store = RedisVectorStore(redis_client_async=redis_client_async, schema=schema, legacy_filters=True)
    try:
        await vector_store.adelete_nodes(node_ids=[str(node_uuid) for node_uuid in node_uuids])
        logger.info("Deleting excess vector docs")
    except Exception as e:
        logger.warning("Error deleting vector docs. Here is the error: %s", str(e))


# Modified from the redisvl package in the semantic.py module
class AsyncSemanticCache(SemanticCache):
    """Async Semantic Cache for Large Language Models."""

    _index: SearchIndex
    _aindex: Optional[AsyncSearchIndex] = None

    def __init__(self, ttl, vectorizer, distance_threshold):
        """Semantic Cache for Large Language Models.

        Args:
            name (str, optional): The name of the semantic cache search index.
                Defaults to "llmcache".
            distance_threshold (float, optional): Semantic threshold for the
                cache. Defaults to 0.1.
            ttl (Optional[int], optional): The time-to-live for records cached
                in Redis. Defaults to None.
            vectorizer (Optional[BaseVectorizer], optional): The vectorizer for the cache.
                Defaults to HFTextVectorizer.
            filterable_fields (Optional[List[Dict[str, Any]]]): An optional list of RedisVL fields
                that can be used to customize cache retrieval with filters.
            redis_client(Optional[Redis], optional): A redis client connection instance.
                Defaults to None.
            redis_url (str, optional): The redis url. Defaults to redis://localhost:6379.
            connection_kwargs (Dict[str, Any]): The connection arguments
                for the redis client. Defaults to empty {}.
            overwrite (bool): Whether or not to force overwrite the schema for
                the semantic cache index. Defaults to false.

        Raises:
            TypeError: If an invalid vectorizer is provided.
            TypeError: If the TTL value is not an int.
            ValueError: If the threshold is not between 0 and 1.
            ValueError: If existing schema does not match new schema and overwrite is False.
        """
        BaseLLMCache.__init__(self, ttl)
        self._vectorizer = vectorizer
        self._dtype = self.aindex.schema.fields[CACHE_VECTOR_FIELD_NAME].attrs.datatype
        self.set_threshold(distance_threshold)

    @classmethod
    async def setup(
        cls,
        name: str = "llmcache",
        distance_threshold: float = 0.1,
        ttl: Optional[int] = None,
        vectorizer: Optional[BaseVectorizer] = None,
        filterable_fields: Optional[List[Dict[str, Any]]] = None,
        redis_client: Optional[RedisAsync] = None,
        redis_url: str = "redis://localhost:6379",
        connection_kwargs: Dict[str, Any] = {},
        overwrite: bool = False,
        **kwargs,
    ):
        cls.redis_kwargs = {
            "redis_client": redis_client,
            "redis_url": redis_url,
            "connection_kwargs": connection_kwargs,
        }

        # Use the index name as the key prefix by default
        if "prefix" in kwargs:
            prefix = kwargs["prefix"]
        else:
            prefix = name

        # Set vectorizer default
        if vectorizer is None:
            vectorizer = HFTextVectorizer(model="sentence-transformers/all-mpnet-base-v2")

        # Process fields and other settings
        cls.return_fields = [
            ENTRY_ID_FIELD_NAME,
            PROMPT_FIELD_NAME,
            RESPONSE_FIELD_NAME,
            INSERTED_AT_FIELD_NAME,
            UPDATED_AT_FIELD_NAME,
            METADATA_FIELD_NAME,
        ]

        # Create semantic cache schema and index
        dtype = kwargs.get("dtype", "float32")
        schema = SemanticCacheIndexSchema.from_params(name, prefix, vectorizer.dims, dtype)
        schema = cls._modify_schema(cls, schema, filterable_fields)
        cls._aindex = AsyncSearchIndex(schema=schema)
        cls._index = cls._aindex

        # Handle redis connection
        if redis_client:
            await cls._aindex.set_client(redis_client)
        elif redis_url:
            await cls._aindex.connect(redis_url=redis_url, **connection_kwargs)

        # Check for existing cache index
        if not overwrite and await cls._aindex.exists():
            existing_index = await AsyncSearchIndex.from_existing(name, redis_client=cls._aindex.client)
            # HACK The only diff was the weight data types, so forcing it to float for both
            if cls._aindex.schema.fields.get("prompt") and existing_index.schema.fields.get("prompt"):
                cls._aindex.schema.fields["prompt"].attrs.weight = float(
                    cls._aindex.schema.fields["prompt"].attrs.weight
                )
                existing_index.schema.fields["prompt"].attrs.weight = float(
                    existing_index.schema.fields["prompt"].attrs.weight
                )
            if cls._aindex.schema.fields.get("response") and existing_index.schema.fields.get("response"):
                cls._aindex.schema.fields["response"].attrs.weight = float(
                    cls._aindex.schema.fields["response"].attrs.weight
                )
                existing_index.schema.fields["response"].attrs.weight = float(
                    existing_index.schema.fields["response"].attrs.weight
                )
            # HACK: Comparing the schemas directly didn't work, so casting to str
            if str(existing_index.schema) != str(cls._aindex.schema):
                raise ValueError(
                    f"Existing index {name} schema does not match the user provided schema for the semantic cache. "
                    "If you wish to overwrite the index schema, set overwrite=True during initialization."
                )

        # Create the search index
        await cls._aindex.create(overwrite=overwrite, drop=False)

        # Initialize and validate vectorizer
        if not isinstance(vectorizer, BaseVectorizer):
            raise TypeError("Must provide a valid redisvl.vectorizer class.")

        validate_vector_dims(
            vectorizer.dims,
            cls._aindex.schema.fields[CACHE_VECTOR_FIELD_NAME].attrs.dims,
        )
        return cls(ttl=ttl, vectorizer=vectorizer, distance_threshold=distance_threshold)


async def delete_semcache_result(
    result_uuid: uuid.UUID,
    semcache_idx_nm: str,
    redis_client_async: RedisAsync,
):
    token_escaper = TokenEscaper()
    result_uuid_esc = token_escaper.escape(str(result_uuid))
    search_str = f"@result_uuid:{{{result_uuid_esc}}}"
    try:
        idx_result = await redis_client_async.ft(semcache_idx_nm).search(Query(search_str))
        doc_id = idx_result.docs[0].id
        await redis_client_async.delete(doc_id)
        logger.info("Deleted sem cache for result: %s", str(result_uuid))
    except Exception:
        logger.debug(f"No sem cache result found for {str(result_uuid)}, skipping")


async def init_semcache(
    client_id: int, redis_client_async: RedisAsync, idx_name: Optional[str] = None
) -> AsyncSemanticCache:
    if not idx_name:
        idx_name = get_semcache_index_name(client_id=client_id)
    llmcache = await AsyncSemanticCache.setup(
        name=idx_name,
        redis_client=redis_client_async,
        distance_threshold=constants.REDIS_SEMCACHE_SIMILAR_DISTANCE,
        filterable_fields=[
            {"name": "client_id", "type": "tag"},
            {"name": "result_uuid", "type": "tag"},
            {"name": "db_uuid", "type": "tag"},
        ],
    )
    return llmcache


async def update_verified_result_vectors(
    db: AsyncSession,
    result: models.ResultHistory,
    prompt_uuid: uuid.UUID,
    content: str,
    verified: bool,
    client_user: sch.ClientUserInfo,
    conn_uuid: uuid.UUID,
    db_uuid: uuid.UUID,
    redis_client_async: RedisAsync,
) -> None:
    """Update a result and indicate if it is verified or not. Verified means that the result
    has been checked by a human and verified that it is correct."""
    semcache_idx_nm = get_semcache_index_name(client_id=client_user.client_id)
    if not verified:
        await delete_semcache_result(
            result_uuid=result.result_uuid, semcache_idx_nm=semcache_idx_nm, redis_client_async=redis_client_async
        )
    elif verified:
        # Save to the Redis semantic cache
        llmcache = await init_semcache(
            client_id=client_user.client_id, idx_name=semcache_idx_nm, redis_client_async=redis_client_async
        )
        # Get the response from the result
        # Store the response in the cache
        sem_cache_metadata = sch.SemCacheMetadata(
            result_uuid=str(result.result_uuid),
            prompt_uuid=str(prompt_uuid),
            verified_user_role=client_user.user_role,
            verified_user_uuid=str(client_user.user_uuid),
            sql_query=result.sql_query,
            timestamp=str(result.timestamp),
            conn_uuid=str(conn_uuid),
        )
        await llmcache.astore(
            prompt=result.initial_prompt,
            response=content,
            metadata=sem_cache_metadata.model_dump(),
            filters={
                "client_id": str(client_user.client_id),
                "result_uuid": str(result.result_uuid),
                "db_uuid": str(db_uuid),
            },
        )


def get_redis_index(index_name: str, settings: Settings, redis_client_async: RedisAsync) -> BaseIndex:  # type:ignore
    schema = IndexSchema.from_dict(
        {
            "index": {"name": index_name, "prefix": index_name + "/vector"},
            "fields": [
                # Required fields
                {"name": "id", "type": "tag"},
                {"name": "doc_id", "type": "tag"},
                {"name": "text", "type": "text"},
                {"name": "vector", "type": "vector", "attrs": {"dims": 1536, "algorithm": "flat"}},
                *constants.VECTOR_FILTERS,
            ],
        }
    )
    vector_store = RedisVectorStore(redis_client_async=redis_client_async, schema=schema, legacy_filters=True)
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=settings.embed_model  # type: ignore
    )
    return vector_index


def get_vector_idx(
    client_id: int, vector_schema: sch.VectorDBSchema, settings: Settings, redis_client_async: RedisAsync  # type: ignore # noqa
) -> VectorStoreIndex:
    """Method for retrieving the vector store index"""
    if not vector_schema.index_name:
        index_name = get_index_name(client_id=client_id)
    else:
        index_name = vector_schema.index_name

    if vector_schema.vector_database_vendor == enums.VectorVendorType.REDIS:
        base_index = get_redis_index(index_name=index_name, settings=settings, redis_client_async=redis_client_async)
    else:
        raise NotImplementedError
    # TODO: Need else here so there isn't error for no base_index
    logger.debug("Using index name: %s", index_name)
    vector_index = VectorStoreIndex(
        index_struct=base_index.index_struct,
        embed_model=settings.embed_model,  # type: ignore
        storage_context=base_index.storage_context,
    )
    return vector_index
