import json
import uuid
from typing import Optional

import redis
from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.indices.base import BaseIndex
from llama_index.vector_stores.redis import RedisVectorStore, TokenEscaper
from redis.asyncio import Redis as RedisAsync
from redis.commands.search.query import Query
from redisvl.schema import IndexSchema

from basejump.core.common.config.logconfig import set_logging
from basejump.core.models import constants, enums
from basejump.core.models import schemas as sch

logger = set_logging(handler_option="stream", name=__name__)

REDIS_PARTITION_PREFIX = "basejump_pclient"
DOCS_INDEX_PREFIX = "docs_"
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
        logger.debug("Deleting excess vector docs")
    except Exception as e:
        logger.warning("Error deleting vector docs. Here is the error: %s", str(e))


def get_redis_vector_store(
    index_name: str,
    redis_client_async: RedisAsync,
    # HACK: Including redis_client since when using the from_documents method of VectorStoreIndex,
    # it uses _redis_client instead of the async redis client
    redis_client: Optional[redis.Redis] = None,
    overwrite: bool = False,
) -> RedisVectorStore:
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
    return RedisVectorStore(
        redis_client=redis_client,
        redis_client_async=redis_client_async,
        schema=schema,
        legacy_filters=True,
        overwrite=overwrite,
    )


def get_redis_index(
    index_name: str,
    embed_model: BaseEmbedding,
    redis_client_async: RedisAsync,
    redis_client: Optional[redis.Redis] = None,
    overwrite: bool = False,
) -> BaseIndex:
    vector_store = get_redis_vector_store(
        index_name=index_name, redis_client_async=redis_client_async, redis_client=redis_client, overwrite=overwrite
    )
    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    return vector_index


def get_vector_idx(
    client_id: int, vector_schema: sch.VectorDBSchema, embed_model: BaseEmbedding, redis_client_async: RedisAsync
) -> VectorStoreIndex:
    """Method for retrieving the vector store index"""
    if not vector_schema.index_name:
        index_name = get_index_name(client_id=client_id)
    else:
        index_name = vector_schema.index_name

    if vector_schema.vector_database_vendor == enums.VectorVendorType.REDIS:
        base_index = get_redis_index(
            index_name=index_name, embed_model=embed_model, redis_client_async=redis_client_async
        )
    else:
        raise NotImplementedError
    # TODO: Need else here so there isn't error for no base_index
    logger.debug("Using index name: %s", index_name)
    vector_index = VectorStoreIndex(
        index_struct=base_index.index_struct,
        embed_model=embed_model,
        storage_context=base_index.storage_context,
    )
    return vector_index


def get_docs_index_name(client_id: int, db_uuid: uuid.UUID) -> str:
    idx_nm = get_index_name(client_id=client_id)
    return DOCS_INDEX_PREFIX + str(db_uuid).replace("-", "_") + idx_nm
