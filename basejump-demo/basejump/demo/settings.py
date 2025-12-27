import os

import redis.asyncio as redis_async
from redis.asyncio import Redis as RedisAsync

from basejump.core.models import schemas as sch
from basejump.core.models import enums
from basejump.core.database.db_connect import ConnectDB
from llama_index.llms.azure_openai import AzureOpenAI

# Setup database
description = "Useful for finding information about clients, teams, and users."
conn_params = sch.SQLDBSchema(
    database_type=enums.DatabaseType.POSTGRES,
    drivername=enums.DBAsyncDriverName.POSTGRES,
    username=os.environ["LOCAL_DB_USER"],
    password=os.environ["LOCAL_DB_PASSWORD"],
    host=os.environ["LOCAL_DB_HOST"],
    port=int(os.environ["LOCAL_DB_PORT"]),
    database_name=os.environ["LOCAL_DB_NAME"],
    query={},
    schemas=[sch.DBSchema(schema_nm="account")],
    database_desc=description,
    data_source_desc=description,
    include_default_schema=False,
    ssl=False,  # Turning off SSL for toy demo example, should always be True in production
)
conn_db = ConnectDB(conn_params=conn_params)
sql_engine = conn_db.connect_async_db()


def get_redis_client_async_instance() -> RedisAsync:
    return redis_async.Redis(
        host=os.getenv("LOCAL_REDIS_HOST"),
        port=os.getenv("LOCAL_REDIS_PORT"),
        decode_responses=False,
        ssl=False,
    )


# Setup embedding model
embedding_endpoint_info = sch.AzureEndpointInfo(
    endpoint=os.environ["AZURE_EMBEDDING_MODEL_ENDPOINT"],
    api_key=os.environ["AZURE_EMBEDDING_MODEL_KEY"],
    deployment_name=os.environ["AZURE_EMBEDDING_MODEL_DEPLOY_NAME"],
)
embedding_model_info = sch.AzureModelInfo(
    model_name=enums.AIModelSchema.ADA3_SMALL,
    endpoint_info=embedding_endpoint_info,
    api_version="2024-06-01",
)

# Setup small model
small_model_endpoint_info = sch.AzureEndpointInfo(
    endpoint=os.environ["AZURE_SMALL_MODEL_ENDPOINT"],
    api_key=os.environ["AZURE_SMALL_MODEL_KEY"],
    deployment_name=os.environ["AZURE_SMALL_MODEL_DEPLOY_NAME"],
)
small_model_info = sch.AzureModelInfo(
    model_name=enums.AIModelSchema.GPT4oMINI,
    endpoint_info=small_model_endpoint_info,
    api_version="2024-06-01",
)

# Setup large model
large_model_endpoint_info = sch.AzureEndpointInfo(
    endpoint=os.environ["AZURE_LARGE_MODEL_ENDPOINT"],
    api_key=os.environ["AZURE_LARGE_MODEL_KEY"],
    deployment_name=os.environ["AZURE_LARGE_MODEL_DEPLOY_NAME"],
)
large_model_info = sch.AzureModelInfo(
    model_name=enums.AIModelSchema.GPT4o,
    endpoint_info=large_model_endpoint_info,
    api_version="2024-06-01",
)

# Setup LLM
LLM = AzureOpenAI(
    model=large_model_info.model_name.value,
    temperature=0,
    max_tokens=large_model_info.max_tokens,
    deployment_name=large_model_info.endpoint_info.deployment_name,
    api_key=large_model_info.endpoint_info.api_key,
    azure_endpoint=large_model_info.endpoint_info.endpoint,
    api_version=large_model_info.api_version,
)
