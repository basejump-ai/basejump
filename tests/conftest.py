"""
This file is automatically imported by pytest
and contains shared fixtures and hooks for the test suite.
"""

import pytest
from contextlib import asynccontextmanager

from basejump.demo import (
    settings,
    service,
    schemas,
)  # inits env vars so it needs to be first # isort: skip
from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.session import LocalSession
from basejump.core.models import enums
from basejump.core.models import schemas as sch

logger = set_logging(handler_option="stream", name=__name__)

# TODO: Make more DRY with the basejump.demo package


# TODO: Will get an HTTPX event loop closed error when using pytest due to LLamaIndex managing the httpx client
# and the interaction with pytest. Long-term fix is to pass in an async_http_client to AzureOpenAIEmbedding and
# AzureOpenAI objects.
@asynccontextmanager
async def get_session(test_env: schemas.PyTestEnv) -> schemas.PyTestEnv:
    """Manages objects that cannot be shared across tests due to pytest
    using a new event loop for each test as opposed to one event loop for all tests"""
    sql_engine = settings.conn_db.connect_async_db()
    session = LocalSession(client_id=test_env.client_id, engine=sql_engine)
    db = await session.open()
    redis_client_async = settings.get_redis_client_async_instance()
    test_env.db = db
    test_env.redis_client_async = redis_client_async
    test_env.sql_engine = sql_engine
    core_session = sch.CoreSession(redis_client_async=redis_client_async, sql_engine=sql_engine)
    test_env.service_context = service.create_service_context(core_session=core_session)
    yield test_env
    await session.close()
    await sql_engine.dispose()
    await redis_client_async.aclose()


@pytest.fixture(scope="session")
async def client_init():
    """Set up a client"""
    # Create a client
    sql_engine = settings.conn_db.connect_async_db()
    client_result = await service.create_client(
        sql_engine=sql_engine,
        client_name="ABC Company",
        client_type=enums.ClientType.DEMO,
        description="A company that provides ABC as a service.",
    )
    logger.info(client_result)

    # Create a session
    session = LocalSession(client_id=1, engine=sql_engine)
    db = await session.open()

    try:
        # Create a team
        team_result = await service.create_team(
            db=db,
            team_name="AI power users",
            client_id=client_result.client_id,
            team_desc="A team in charge of managing ABC",
        )
        logger.info(team_result)

        # Create a user
        user_result = await service.create_user(
            db=db,
            client_id=client_result.client_id,
            username="John Doe",
            email_address="john@gmail.com",
        )
        logger.info(user_result)

        # Create a client user object
        client_user = sch.ClientUserInfo(
            client_id=client_result.client_id,
            client_uuid=client_result.client_uuid,
            user_id=user_result.user_id,
            user_uuid=user_result.user_uuid,
            user_role="MEMBER",
        )

        user_info = sch.UserInfo(
            client_id=client_user.client_id,
            client_uuid=client_user.client_uuid,
            user_id=client_user.user_id,
            user_uuid=client_user.user_uuid,
            user_role=client_user.user_role,
            team_id=team_result.team_id,
            team_uuid=team_result.team_uuid,
        )

        # Create a connection params object
        client_conn_params = sch.SQLDBSchema(**settings.conn_params.dict())
        client_conn_params.drivername = enums.DBDriverName.POSTGRES

        # Create an object for passing variables
        env_vars = schemas.PyTestEnv(
            client_id=client_result.client_id,
            client_uuid=client_result.client_uuid,
            team_id=team_result.team_id,
            team_uuid=team_result.team_uuid,
            user_id=user_result.user_id,
            user_uuid=user_result.user_uuid,
            client_secret=client_result.client_secret,
            username=user_result.username,
            team_name=team_result.team_name,
            client_user=client_user,
            team_info=sch.TeamFields.model_validate(team_result),
            client_conn_params=client_conn_params,
            user_info=user_info,
        )
    except Exception as e:
        await db.rollback()
        raise e
    finally:
        await session.close()
        await sql_engine.dispose()
    yield env_vars

    # TODO: Drop AWS S3 Files / create local file saving alternative


@pytest.fixture(scope="function")
async def client_session(client_init):
    """Get a session after the client has been set up"""
    async with get_session(client_init) as updated_env:
        yield updated_env


@pytest.fixture(scope="session")
async def db_init(client_init):
    """Setup the database using client_init as a dependency"""
    # Update the client database to a synchronous connection since not all DBs support asynch connections
    # Get new connections
    sql_engine = settings.conn_db.connect_async_db()
    session = LocalSession(client_id=client_init.client_id, engine=sql_engine)
    db = await session.open()
    redis_client_async = settings.get_redis_client_async_instance()
    redis_client = settings.get_redis_client_instance()

    # Add database
    core_session = sch.CoreSession(
        redis_client=redis_client, redis_client_async=redis_client_async, sql_engine=sql_engine
    )
    service_context = service.create_service_context(core_session=core_session)
    db_result = await service.setup_database(
        db=db,
        service_context=service_context,
        user_info=client_init.user_info,
        conn_params=client_init.client_conn_params,
        index_docs=True,
    )

    # Update test env vars
    client_init.db_id = db_result.db_id
    client_init.db_uuid = db_result.db_uuid
    client_init.conn_id = db_result.conn_id
    client_init.conn_uuid = db_result.conn_uuid

    # Add a connection to a team
    await service.add_connection_to_team(
        db=db,
        client_id=client_init.client_id,
        team_id=client_init.team_id,
        conn_id=db_result.conn_id,
    )

    # Associate a user with a team
    user_team_result = await service.add_user_to_team(
        db=db,
        username=client_init.username,
        team_name=client_init.team_name,
        user_id=client_init.user_id,
        team_id=client_init.team_id,
    )
    logger.info(user_team_result)
    await session.close()
    await sql_engine.dispose()
    await redis_client_async.aclose()
    return client_init


@pytest.fixture(scope="function")
async def db_session(db_init):
    """Get a session after the database and client have been set up"""
    async with get_session(db_init) as updated_env:
        yield updated_env


@pytest.fixture(scope="session")
async def chat_init(db_init):
    """Setup the chat using db_init as a dependency"""
    sql_engine = settings.conn_db.connect_async_db()
    session = LocalSession(client_id=db_init.client_id, engine=sql_engine)
    db = await session.open()

    # Create a chat
    create_chat_result = await service.create_chat(
        db=db,
        client_id=db_init.client_id,
        team_id=db_init.team_id,
        user_id=db_init.user_id,
    )

    # Ask the AI a question
    redis_client_async = settings.get_redis_client_async_instance()
    service_context = sch.ServiceContext(
        sql_engine=sql_engine,
        redis_client_async=redis_client_async,
        large_model_info=settings.large_model_info,
        small_model_info=settings.small_model_info,
        embedding_model_info=settings.embedding_model_info,
    )
    chat_result = await service.chat(
        db=db,
        prompt="Give me a report of all clients.",
        service_context=service_context,
        user_info=db_init.user_info,
        chat=create_chat_result,
        use_docs=True,
    )

    db_init.chat_id = create_chat_result.chat_id
    db_init.chat_uuid = create_chat_result.chat_uuid
    db_init.vector_id = create_chat_result.vector_id
    # Here is the LLM response
    logger.info("LLM response: %s", chat_result.content)
    # Here is the SQL query that was ran
    logger.debug("SQL query: %s", chat_result.query_result.sql_query)
    # Use this to get the result in AWS S3
    db_init.result_uuid = chat_result.query_result.result_uuid
    logger.debug("Result UUID: %s", chat_result.query_result.result_uuid)
    await session.close()
    await sql_engine.dispose()
    await redis_client_async.aclose()
    return db_init


@pytest.fixture(scope="function")
async def chat_session(chat_init):
    """Get a session after the client, database, and chat have been set up"""
    async with get_session(chat_init) as updated_env:
        yield updated_env
