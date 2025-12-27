import os
import secrets
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from basejump.core.models import schemas as sch
from basejump.core.database.aicatalog import AICatalog
from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.db_connect import LocalSession
from basejump.core.database import upload
from basejump.core.models import enums, models
from basejump.core.common.common_utils import hash_value
from basejump.core.database.crud import crud_main
from basejump.core.database.index import index_db
from basejump.core.service import service_utils
from basejump.core.service.agents.mermaid import MermaidAgent
from basejump.core.models import prompts
from basejump.demo import crud, schemas
from redis.asyncio import Redis as RedisAsync
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from basejump.core.database.vector_utils import get_index_name, get_index_schema
from basejump.core.database.crud import crud_utils
from llama_index.vector_stores.redis import RedisVectorStore
from basejump.core.service.base import AgentSetup, ChatAgentSetup
from basejump.core.service.agents.data_chat import DataChatAgent
from basejump.demo import settings
from contextlib import asynccontextmanager
from llama_index.core.llms import ChatMessage

logger = set_logging(handler_option="stream", name=__name__)


@asynccontextmanager
async def run_session(client_id: int):
    session = LocalSession(client_id=client_id, engine=settings.sql_engine)
    db = await session.open()
    try:
        yield db
    except Exception as e:
        logger.error(e)
        await db.rollback()
        raise e
    await settings.redis_client_async.aclose()


async def create_client(
    sql_engine: AsyncEngine,
    client_name: str,
    client_type: enums.ClientType,
    description: str,
) -> schemas.GetClient:
    """Create a client"""
    session = LocalSession(client_id=0, engine=sql_engine)
    await session.create_schemas()

    # Create any tables if they don't exist
    async with sql_engine.begin() as conn:
        await conn.run_sync(models.Base.metadata.create_all)
    try:
        db = await session.open()
        client_secret = secrets.token_hex(32)
        hashed_client_secret = hash_value(client_secret)
        client = sch.CreateClient(
            client_name=client_name,
            hashed_client_secret=hashed_client_secret,
            client_type=client_type,
        )
        new_client = await crud_main.create_client(
            db=db, client=client, sql_engine=sql_engine, description=description
        )
        client_secret_uuid = new_client.client_secret_uuid
        default_storage_conn = models.ClientStorageConnection(
            client_id=new_client.client_id,
            alias="basejump_default",
            storage_provider="AWS_S3",
            region="us-east-2",
            bucket_name=os.environ["AWS_STORAGE_BUCKET_NAME"],
            access_key=os.environ["AWS_USER_ACCESS_KEY_ID"],
            secret_access_key=os.environ["AWS_USER_SECRET_ACCESS_KEY"],
            active=True,
            prefix=upload.get_default_prefix(client_uuid=new_client.client_uuid),
            internal=True,
        )
        db.add(default_storage_conn)
        await db.commit()
    except Exception as e:
        logger.error(e)
        raise e
    finally:
        await session.close()
    return schemas.GetClient(
        client_name=client_name,
        client_id=new_client.client_id,
        client_uuid=new_client.client_uuid,
        client_secret=client_secret,
        client_secret_uuid=client_secret_uuid,
        role=enums.APIUserRoles.INTERNAL.value,
        description=new_client.description,
        hashed_client_secret=hashed_client_secret,
        client_type=enums.ClientType.DEMO,
    )


async def create_team(
    db: AsyncSession, team_name: str, client_id: int, team_desc: str
) -> schemas.GetTeam:
    """Get a team"""
    team = sch.BaseTeam(team_name=team_name, client_id=client_id, team_desc=team_desc)
    team_result = await crud_main.create_team(db=db, team=team)
    return schemas.GetTeam.from_orm(team_result)


async def create_user(
    db: AsyncSession,
    client_id: int,
    username: str,
    email_address: str,
    role: enums.UserRoles = enums.UserRoles.MEMBER,
) -> schemas.GetUser:
    """Create a user"""
    base_user = sch.BaseUser(
        client_id=client_id,
        username=username,
        role=role,
        email_address=email_address,
    )
    user = await crud_main.create_user(db=db, user=base_user)
    return schemas.GetUser.from_orm(user)


async def add_user_to_team(
    db: AsyncSession,
    username: str,
    team_name: str,
    user_id: int,
    team_id: int,
) -> schemas.GetUserTeam:
    """Add a user to a team"""
    await crud.add_user_to_team(
        db=db, username=username, team_name=team_name, user_id=user_id, team_id=team_id
    )
    return schemas.GetUserTeam(user_id=user_id, team_id=team_id)


async def add_client_database(
    db: AsyncSession,
    client_id: int,
    conn_params: sch.SQLDBSchema,
    redis_client_async: RedisAsync,
    embedding_model_info: sch.AzureModelInfo,
    client_user: sch.ClientUserInfo,
    small_model_info: sch.AzureModelInfo,
    sql_engine: AsyncEngine,
) -> schemas.GetSQLConn:
    """Create a database connection and save it in the database"""
    # Set up the database
    sql_conn, index_db_tables = await service_utils.setup_db(
        db=db,
        client_user=client_user,
        redis_client_async=redis_client_async,
        conn_params=conn_params,
        embedding_model_info=embedding_model_info,
    )
    get_sql_conn = schemas.GetSQLConn(
        conn_id=sql_conn.conn_id,
        conn_uuid=sql_conn.conn_uuid,
        db_uuid=sql_conn.db_uuid,
        db_id=sql_conn.db_id,
    )

    # Index the database
    await index_db(
        index_db_tables=index_db_tables,
        conn_params=conn_params,
        client_user=client_user,
        db_id=sql_conn.db_id,
        db_uuid=sql_conn.db_uuid,
        conn_id=sql_conn.conn_id,
        small_model_info=small_model_info,
        redis_client_async=redis_client_async,
        sql_engine=sql_engine,
    )
    return get_sql_conn


async def add_connection_to_team(
    db: AsyncSession, client_id: int, team_id: int, conn_id: int
) -> None:
    await crud.add_connection_to_team(
        db=db, client_id=client_id, team_id=team_id, conn_id=conn_id
    )
    logger.info(f"Added connection {conn_id} to team {team_id}")


async def create_chat(
    db: AsyncSession, client_id: int, user_id: int, team_id: int
) -> schemas.GetChat:
    """Create a chat instance"""
    index_name = get_index_name(client_id=client_id)
    vector_id = await crud_utils.get_next_val(
        db=db, full_table_nm=str(models.DBVector.__table__), column_nm="vector_id"
    )
    db_vector = models.DBVector(
        client_id=client_id,
        vector_id=vector_id,
        vector_uuid=uuid.uuid4(),
        vector_database_vendor=enums.VectorVendorType.REDIS.value,
        vector_datasource_type=enums.VectorSourceType.CHAT.value,
        index_name=index_name,
    )
    db.add(db_vector)
    await db.commit()
    chat = models.Chat(
        user_id=user_id,
        team_id=team_id,
        client_id=client_id,
        chat_name="A test chat",
        chat_description="A test chat",
        vector_id=vector_id,
    )
    db.add(chat)
    await db.commit()
    await db.refresh(chat)
    return schemas.GetChat(
        chat_uuid=chat.chat_uuid, chat_id=chat.chat_id, vector_id=vector_id
    )


async def chat(
    db: AsyncSession,
    index_name: str,
    prompt: str,
    chat_id: int,
    team_id: int,
    team_info: sch.TeamFields,
    client_user: sch.ClientUserInfo,
    embedding_model_info: sch.AzureModelInfo,
    sql_engine: AsyncEngine,
    redis_client_async: RedisAsync,
    conn_params: sch.SQLDBSchema,
    vector_id: int,
    chat_uuid: uuid.UUID,
    team_uuid: uuid.UUID,
    large_model_info: sch.AzureModelInfo,
    small_model_info: sch.AzureModelInfo,
    client_llm: enums.AIModelSchema = enums.AIModelSchema.GPT4o,
    return_visual_json: bool = True,
) -> sch.Message:
    # Setup the chat
    prompt_metadata_base = await service_utils.create_prompt_base(
        db=db,
        client_user=client_user,
        prompt=prompt,
        return_visual_json=return_visual_json,
    )
    schema = get_index_schema(index_name=index_name)
    vector_store = RedisVectorStore(
        redis_client_async=redis_client_async, schema=schema, legacy_filters=True
    )
    agent_setup = AgentSetup.load_from_prompt_metadata(
        prompt_metadata_base=prompt_metadata_base
    )
    chat_metadata = sch.ChatMetadata(
        chat_id=chat_id,
        chat_uuid=chat_uuid,
        vector_id=vector_id,
        index_name=index_name,
        team_uuid=team_uuid,
        team_id=team_id,
        parent_msg_uuid=uuid.uuid4(),
        curr_chat_history=[],
        vector_store=vector_store,
        embedding_model_info=embedding_model_info,
    )

    # Setup the agent
    chat_setup = ChatAgentSetup(
        db=db,
        prompt_metadata=agent_setup.prompt_metadata,
        chat_metadata=chat_metadata,
        redis_client_async=redis_client_async,
        embedding_model_info=embedding_model_info,
        team_info=team_info,
    )
    chat = await chat_setup.get_chat()
    chat_history = await chat_setup.get_chat_history(chat=chat)

    # Initialize the agent
    agent = DataChatAgent(
        db_conn_params=conn_params,
        chat_history=chat_history,
        prompt_metadata=chat_setup.prompt_metadata,
        chat_metadata=chat_metadata,
        agent_llm=client_llm,
        redis_client_async=redis_client_async,
        large_model_info=large_model_info,
        small_model_info=small_model_info,
        embedding_model_info=embedding_model_info,
        sql_engine=sql_engine,
    )
    message = await agent.prompt_agent()
    return message


async def setup_mermaid_agent(
    client_user: sch.ClientUserInfo,
    prompt_id: int,
    prompt_uuid: uuid.UUID,
    large_model_info: sch.AzureModelInfo,
    sql_engine: AsyncEngine,
    redis_client_async: RedisAsync,
) -> MermaidAgent:

    # Setup the agent prompts
    prompt_metadata_base = sch.PromptMetadataBase(
        initial_prompt="",
        user_id=client_user.user_id,
        user_uuid=client_user.user_uuid,
        client_uuid=client_user.client_uuid,
        client_id=client_user.client_id,
        user_role=client_user.user_role,
        prompt_uuid=prompt_uuid,
        prompt_id=prompt_id,
        llm_type=enums.LLMType.MERMAID_AGENT,
        prompt_time=datetime.now(),
    )
    agent_setup = AgentSetup.load_from_prompt_metadata(
        prompt_metadata_base=prompt_metadata_base
    )

    # Set up the agent
    large_model_info.max_tokens = 4096
    ai_catalog = AICatalog()
    agent_llm = ai_catalog.get_llm(model_info=large_model_info)

    # Set up the mermaid agent
    mermaid_agent = MermaidAgent(
        prompt_metadata=agent_setup.prompt_metadata,
        chat_history=[
            ChatMessage(
                role=sch.MessageRole.SYSTEM,
                content=prompts.MERMAIDJS_SYSTEM_PROMPT,
                timestamp=datetime.now(ZoneInfo("UTC")),
            )
        ],
        max_iterations=8,
        agent_llm=agent_llm,
        sql_engine=sql_engine,
        large_model_info=large_model_info,
        redis_client_async=redis_client_async,
    )
    return mermaid_agent
