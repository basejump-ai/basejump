from typing import Any, Dict, List, Optional

from redis.asyncio import Redis as RedisAsync
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
from redisvl.utils.utils import validate_vector_dims
from redisvl.utils.vectorize import BaseVectorizer, HFTextVectorizer


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
