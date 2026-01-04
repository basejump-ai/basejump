import os
import secrets
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, Optional
from zoneinfo import ZoneInfo

from llama_index.core.llms import ChatMessage
from llama_index.vector_stores.redis import RedisVectorStore
from redis.asyncio import Redis as RedisAsync
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from basejump.core.common.common_utils import hash_value
from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.crud import crud_main, crud_utils
from basejump.core.database.db_connect import LocalSession
from basejump.core.database.index import index_db
from basejump.core.database.result import store
from basejump.core.database.vector_utils import get_index_name, get_index_schema
from basejump.core.models import enums, errors, models, prompts
from basejump.core.models import schemas as sch
from basejump.core.models.ai.catalog import AICatalog
from basejump.core.service import service_utils
from basejump.core.service.agents.data_chat import DataChatAgent
from basejump.core.service.agents.mermaid import MermaidAgent
from basejump.core.service.base import AgentSetup, ChatAgentSetup
from basejump.demo import crud, schemas, settings

logger = set_logging(handler_option="stream", name=__name__)


@asynccontextmanager
async def run_session(client_id: Optional[int] = None) -> AsyncGenerator:
    session = LocalSession(client_id=client_id or 0, engine=settings.sql_engine)
    db = await session.open()
    redis_client_async = settings.get_redis_client_async_instance()
    core_session = sch.CoreSession(redis_client_async=redis_client_async, sql_engine=settings.sql_engine)
    try:
        yield core_session, db
    except Exception as e:
        logger.error(e)
        await db.rollback()
        raise e
    finally:
        await session.close()
        await settings.sql_engine.dispose()
        await redis_client_async.aclose()


async def create_client(
    sql_engine: AsyncEngine,
    client_name: str,
    client_type: enums.ClientType,
    description: str,
    client_id: Optional[int] = None,
    external_storage: bool = False,
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
            db=db,
            client=client,
            sql_engine=sql_engine,
            description=description,
            client_id=client_id,
        )
        client_secret_uuid = new_client.client_secret_uuid
        if external_storage:
            # NOTE: Only support AWS object storage currently
            # TODO: Add other object storage
            default_storage_conn = models.ClientStorageConnection(
                client_id=new_client.client_id,
                alias="basejump_default",
                storage_provider="AWS_S3",
                region=os.environ["AWS_REGION"],
                bucket_name=os.environ["AWS_STORAGE_BUCKET_NAME"],
                access_key=os.environ["AWS_USER_ACCESS_KEY_ID"],
                secret_access_key=os.environ["AWS_USER_SECRET_ACCESS_KEY"],
                active=True,
                prefix=store.S3ResultStore.get_default_prefix(client_uuid=new_client.client_uuid),
                internal=True,
            )
            db.add(default_storage_conn)
            await db.commit()
    except errors.AlreadyExists as e:
        raise e
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


async def create_internal_client(db: AsyncSession, sql_engine: AsyncEngine) -> schemas.GetClientBase:
    try:
        client_id = 0
        client_result = await create_client(
            sql_engine=sql_engine,
            client_id=client_id,
            client_name="Default client",
            client_type=enums.ClientType.INTERNAL,
            description="A client for internal/dev use only.",
        )
        logger.info(client_result)
        return client_result
    except errors.AlreadyExists:
        pass
    client = await crud_main.get_client_from_id(db=db, client_id=client_id)
    assert client, "No client found with that ID"
    return schemas.GetClientBase.from_orm(client)


async def create_team(
    db: AsyncSession,
    team_name: str,
    client_id: int,
    team_desc: str,
    team_id: Optional[int] = None,
) -> schemas.GetTeam:
    """Get a team"""
    team = sch.BaseTeam(team_name=team_name, client_id=client_id, team_desc=team_desc)
    team_result = await crud_main.create_team(db=db, team=team, team_id=team_id)
    return schemas.GetTeam.from_orm(team_result)


async def create_internal_user(db: AsyncSession, client_id: int):
    # Create a user
    try:
        user_id = 0
        user_result = await create_user(
            db=db,
            user_id=user_id,
            client_id=client_id,
            username="Default user",
        )
        logger.info(user_result)
    except errors.AlreadyExists:
        pass
    user = await crud_main.get_user_from_id(db=db, user_id=user_id)
    assert user, "No user found with that ID"
    return schemas.GetUser.from_orm(user)


async def create_internal_user_info(
    db: AsyncSession,
    service_context: sch.ServiceContext,
) -> sch.UserInfo:
    """Create a client, team, and user for an internal user"""
    client = await create_internal_client(db=db, sql_engine=service_context.sql_engine)
    team = await create_internal_team(db=db, client_id=client.client_id)
    user = await create_internal_user(db=db, client_id=client.client_id)
    return sch.UserInfo(
        client_id=client.client_id,
        client_uuid=client.client_uuid,
        user_id=user.user_id,
        user_uuid=user.user_uuid,
        user_role=user.role,
        team_id=team.team_id,
        team_uuid=team.team_uuid,
        team_name=team.team_name,
        team_desc=team.team_desc,
    )


async def create_internal_team(db: AsyncSession, client_id: int):
    try:
        team_id = 0
        team_result = await create_team(
            db=db,
            team_id=team_id,
            team_name="Default team",
            client_id=client_id,
            team_desc="A team in charge of managing ABC",
        )
        logger.info(team_result)
    except errors.AlreadyExists:
        pass
    team = await crud_main.get_team_from_id(db=db, team_id=team_id)
    assert team, "No team found with that ID"
    return schemas.GetTeam.from_orm(team)


async def create_user(
    db: AsyncSession,
    client_id: int,
    username: str,
    role: enums.UserRoles = enums.UserRoles.MEMBER,
    user_id: Optional[int] = None,
    email_address: Optional[str] = None,
) -> schemas.GetUser:
    """Create a user"""
    base_user = sch.BaseUser(
        client_id=client_id,
        username=username,
        role=role,
        email_address=email_address,
    )
    user = await crud_main.create_user(db=db, user=base_user, user_id=user_id)
    return schemas.GetUser.from_orm(user)


async def add_user_to_team(
    db: AsyncSession,
    username: str,
    team_name: str,
    user_id: int,
    team_id: int,
) -> schemas.GetUserTeam:
    """Add a user to a team"""
    await crud.add_user_to_team(db=db, username=username, team_name=team_name, user_id=user_id, team_id=team_id)
    return schemas.GetUserTeam(user_id=user_id, team_id=team_id)


async def setup_database(
    db: AsyncSession,
    service_context: sch.ServiceContext,
    user_info: sch.UserInfo,
    conn_params: sch.SQLDBSchema,
    verbose: bool = False,
) -> schemas.GetSQLConn:
    """Create a database connection and save it in the database"""
    # Set up the database
    client_user = sch.ClientUserInfo.model_validate(user_info)
    sql_conn, index_db_tables = await service_utils.setup_db(
        db=db,
        client_user=client_user,
        redis_client_async=service_context.redis_client_async,
        conn_params=conn_params,
        embedding_model_info=service_context.embedding_model_info,
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
        small_model_info=service_context.small_model_info,
        redis_client_async=service_context.redis_client_async,
        sql_engine=service_context.sql_engine,
        verbose=verbose,
    )
    return get_sql_conn


async def add_connection_to_team(db: AsyncSession, client_id: int, team_id: int, conn_id: int) -> None:
    await crud.add_connection_to_team(db=db, client_id=client_id, team_id=team_id, conn_id=conn_id)
    logger.info(f"Added connection {conn_id} to team {team_id}")


async def create_chat(db: AsyncSession, client_id: int, user_id: int, team_id: int) -> schemas.GetChat:
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
    return schemas.GetChat(chat_uuid=chat.chat_uuid, chat_id=chat.chat_id, vector_id=vector_id)


async def setup_mermaid_agent(
    client_user: sch.ClientUserInfo,
    prompt_id: int,
    prompt_uuid: uuid.UUID,
    large_model_info: sch.ModelInfo,
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
        model_name=large_model_info.model_name,
        llm_type=enums.LLMType.MERMAID_AGENT,
        prompt_time=datetime.now(),
    )
    agent_setup = AgentSetup.load_from_prompt_metadata(prompt_metadata_base=prompt_metadata_base)

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


async def chat(
    db: AsyncSession,
    prompt: str,
    service_context: sch.ServiceContext,
    user_info: sch.UserInfo,
    connection: Optional[schemas.GetSQLConn] = None,
    chat: Optional[schemas.GetChat] = None,
    get_chat_history: bool = False,
) -> sch.Message:
    # Create a chat
    if not chat:
        chat = await create_chat(
            db=db,
            client_id=user_info.client_id,
            team_id=user_info.team_id,
            user_id=user_info.user_id,
        )
    # Set up the prompt
    client_user = sch.ClientUserInfo.model_validate(user_info)
    prompt_metadata_base = await service_utils.create_prompt_base(
        db=db,
        client_user=client_user,  # TODO: Replace with UserInfo instead
        prompt=prompt,
        model_name=service_context.large_model_info.model_name,
    )

    # Set up the vector store
    index_name = get_index_name(client_id=user_info.client_id)
    schema = get_index_schema(index_name=index_name)
    vector_store = RedisVectorStore(
        redis_client_async=service_context.redis_client_async,
        schema=schema,
        legacy_filters=True,
    )

    # Set up the agent
    agent_setup = AgentSetup.load_from_prompt_metadata(prompt_metadata_base=prompt_metadata_base)
    chat_metadata = sch.ChatMetadata(
        chat_id=chat.chat_id,
        chat_uuid=chat.chat_uuid,
        vector_id=chat.vector_id,
        index_name=index_name,
        team_uuid=user_info.team_uuid,
        team_id=user_info.team_id,
        parent_msg_uuid=uuid.uuid4(),
        curr_chat_history=[],
        vector_store=vector_store,
        embedding_model_info=service_context.embedding_model_info,
    )

    # Get chat history
    chat_history = None
    if get_chat_history:
        chat_setup = ChatAgentSetup(
            db=db,
            prompt_metadata=agent_setup.prompt_metadata,
            chat_metadata=chat_metadata,
            redis_client_async=service_context.redis_client_async,
            embedding_model_info=service_context.embedding_model_info,
            team_info=sch.TeamFields.model_validate(user_info),
        )
        retrieved_chat = await chat_setup.get_chat()
        chat_history = await chat_setup.get_chat_history(chat=retrieved_chat)

    # Prompt the agent
    ai_catalog = AICatalog()
    agent_llm = ai_catalog.get_llm(model_info=service_context.large_model_info)
    agent = DataChatAgent(
        db_conn_params=settings.conn_params,
        prompt_metadata=agent_setup.prompt_metadata,
        chat_metadata=chat_metadata,
        chat_history=chat_history,
        agent_llm=agent_llm,
        service_context=service_context,
        conn_id=connection.conn_id if connection else None,
    )
    message = await agent.prompt_agent()
    return message


def create_service_context(core_session: sch.CoreSession) -> sch.ServiceContext:
    return sch.ServiceContext(
        sql_engine=core_session.sql_engine,
        redis_client_async=core_session.redis_client_async,
        large_model_info=settings.large_model_info,
        small_model_info=settings.small_model_info,
        embedding_model_info=settings.embedding_model_info,
    )
