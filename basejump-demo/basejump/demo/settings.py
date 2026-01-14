import os

import redis.asyncio as redis_async
from redis.asyncio import Redis as RedisAsync

from basejump.core.common.config.settings import settings
from basejump.core.database.connect import PostgresDB
from basejump.core.models import enums
from basejump.core.models import schemas as sch

# Setup database
description = "Useful for finding information about clients, teams, and users."
conn_params = sch.SQLDBSchema(
    database_type=enums.DatabaseType.POSTGRES,
    drivername=enums.DBAsyncDriverName.POSTGRES,
    # NOTE: These settings should be defined in an .env file with the BASEJUMP_ prefix
    username=settings.db_user,
    password=settings.db_password.get_secret_value(),
    host=settings.db_host,
    port=settings.db_port,
    database_name=settings.db_name,
    query={},
    schemas=[sch.DBSchema(schema_nm="account")],
    database_desc=description,
    data_source_desc=description,
    include_default_schema=False,
    ssl=False,  # Turning off SSL for toy demo example, should always be True in production
)
conn_db = PostgresDB(conn_params=conn_params)
sql_engine = conn_db.connect_async_db()

client_conn_params = sch.SQLDBSchema(**conn_params.dict())
client_conn_params.drivername = enums.DBDriverName.POSTGRES


def get_redis_client_async_instance() -> RedisAsync:
    return redis_async.Redis(
        host=settings.redis_host,  # type: ignore
        port=settings.redis_port,  # type: ignore
        decode_responses=False,
        ssl=False,
    )


# Set up embedding model
embedding_endpoint_info = sch.AzureEndpointInfo(
    endpoint=os.environ["BASEJUMP_AZURE_EMBEDDING_MODEL_ENDPOINT"],
    api_key=os.environ["BASEJUMP_AZURE_EMBEDDING_MODEL_KEY"],
    deployment_name=os.environ["BASEJUMP_AZURE_EMBEDDING_MODEL_DEPLOY_NAME"],
)
embedding_model_info = sch.AzureModelInfo(
    model_name=enums.AIModelSchema.ADA3_SMALL,
    endpoint_info=embedding_endpoint_info,
    api_version="2024-06-01",
)

# Set up small model
small_model_endpoint_info = sch.AzureEndpointInfo(
    endpoint=os.environ["BASEJUMP_AZURE_SMALL_MODEL_ENDPOINT"],
    api_key=os.environ["BASEJUMP_AZURE_SMALL_MODEL_KEY"],
    deployment_name=os.environ["BASEJUMP_AZURE_SMALL_MODEL_DEPLOY_NAME"],
)
small_model_info = sch.AzureModelInfo(
    model_name=enums.AIModelSchema.GPT4oMINI,
    endpoint_info=small_model_endpoint_info,
    api_version="2024-06-01",
)

# Set up large model
large_model_endpoint_info = sch.AzureEndpointInfo(
    endpoint=os.environ["BASEJUMP_AZURE_LARGE_MODEL_ENDPOINT"],
    api_key=os.environ["BASEJUMP_AZURE_LARGE_MODEL_KEY"],
    deployment_name=os.environ["BASEJUMP_AZURE_LARGE_MODEL_DEPLOY_NAME"],
)
large_model_info = sch.AzureModelInfo(
    model_name=enums.AIModelSchema.GPT4o,
    endpoint_info=large_model_endpoint_info,
    api_version="2024-12-01-preview",
)

# NOTE: Also supports Claude served via AWS. Uncomment this section and remove the
# large model info for Azure to test.
# large_model_endpoint_info = sch.AWSEndpointInfo(
#     endpoint=os.environ["BASEJUMP_AWS_LARGE_MODEL_ENDPOINT"],
#     access_key=os.environ["BASEJUMP_AWS_USER_ACCESS_KEY_ID"],
#     secret_access_key=os.environ["BASEJUMP_AWS_USER_SECRET_ACCESS_KEY"],
#     deployment_region=os.environ["BASEJUMP_AWS_LARGE_MODEL_DEPLOYMENT_REGION"],
# )
# large_model_info = sch.AWSModelInfo(model_name=enums.AIModelSchema.SONNET37, endpoint_info=large_model_endpoint_info)
