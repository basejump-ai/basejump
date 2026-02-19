import uuid

import redis
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from redis.asyncio import Redis as RedisAsync

from basejump.core.common.common_utils import find_markdown_files
from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.utils import get_docs_index_name, get_redis_vector_store
from basejump.core.models import schemas as sch
from basejump.core.models.ai.catalog import AICatalog
from basejump.core.service.agents.memory.semantic import SemanticMemory
from basejump.core.service.agents.utils import contextualize_prompt

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
        prompt = await contextualize_prompt(recent_interactions=recent_interactions, small_model_info=small_model_info)
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


def index_database_docs(
    redis_client_async: RedisAsync,
    redis_client: redis.Redis,
    embedding_model_info: sch.AzureModelInfo,
    file_path: str,
    client_id: int,
    db_uuid: uuid.UUID,
):
    """Indexes all markdown files for a given file path into a vector store for a specific database"""
    index_name = get_docs_index_name(client_id=client_id, db_uuid=db_uuid)
    index_docs(
        redis_client_async=redis_client_async,
        redis_client=redis_client,
        embedding_model_info=embedding_model_info,
        file_path=file_path,
        index_name=index_name,
    )


def index_docs(
    redis_client_async: RedisAsync,
    redis_client: redis.Redis,
    embedding_model_info: sch.AzureModelInfo,
    file_path: str,
    index_name: str,
):
    """Indexes all markdown files for a given file path into a vector store"""
    # Get the documents
    md_files = find_markdown_files(file_path)
    documents = SimpleDirectoryReader(input_files=md_files).load_data()

    # Set up the vector store
    vector_store = get_redis_vector_store(
        index_name=index_name, redis_client_async=redis_client_async, redis_client=redis_client
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    ai_catalog = AICatalog()
    embed_model = ai_catalog.get_embedding_model(model_info=embedding_model_info)

    # Index the documents
    logger.info("Indexing database documents using the following index name: %s", index_name)
    VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    logger.info("Database document index completed for: %s", index_name)
