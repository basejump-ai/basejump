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
from basejump.core.database.db_connect import LocalSession
from basejump.core.models import enums
from basejump.core.models import schemas as sch
from basejump.core.database.vector_utils import get_index_name

logger = set_logging(handler_option="stream", name=__name__)

# TODO: Make more DRY with the basejump.demo package


@asynccontextmanager
async def get_session(test_env: schemas.PyTestEnv) -> schemas.PyTestEnv:
    """Manages objects that cannot be shared across tests due to pytest
    using a new event loop for each test as opposed to one event loop for all tests"""
    sql_engine = settings.conn_db.connect_async_db()
    session = LocalSession(client_id=test_env.client_id, engine=sql_engine)
    db = await session.open()
    redis_client_async = settings.get_redis_client_async_instance()
    updated_env = schemas.PyTestEnv(
        **{
            k: v
            for k, v in test_env.dict().items()
            if k not in ["db", "redis_client_async", "sql_engine"]
        },
        db=db,
        redis_client_async=redis_client_async,
        sql_engine=sql_engine
    )
    yield updated_env

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

    # Add database
    db_result = await service.add_client_database(
        db=db,
        client_id=client_init.client_id,
        # Using the same database here for simplicity, but feel free to update
        conn_params=client_init.client_conn_params,
        redis_client_async=redis_client_async,
        client_user=client_init.client_user,
        embedding_model_info=settings.embedding_model_info,
        small_model_info=settings.small_model_info,
        sql_engine=sql_engine,
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
    chat_result = await service.chat(
        db=db,
        index_name=get_index_name(client_id=db_init.client_id),
        prompt="Give me a report of all clients.",
        chat_id=create_chat_result.chat_id,
        team_id=db_init.team_id,
        team_info=db_init.team_info,
        client_user=db_init.client_user,
        embedding_model_info=settings.embedding_model_info,
        sql_engine=sql_engine,
        redis_client_async=redis_client_async,
        conn_params=db_init.client_conn_params,
        vector_id=create_chat_result.vector_id,
        chat_uuid=create_chat_result.chat_uuid,
        team_uuid=db_init.team_uuid,
        large_model_info=settings.large_model_info,
        small_model_info=settings.small_model_info,
        client_llm=settings.LLM,
    )

    db_init.chat_id = create_chat_result.chat_id
    db_init.chat_uuid = create_chat_result.chat_uuid
    db_init.vector_id = create_chat_result.vector_id
    # Here is the LLM response
    logger.info(chat_result.content)
    # Here is the SQL query that was ran
    logger.info(chat_result.query_result.sql_query)
    # Use this to get the result in AWS S3
    db_init.result_uuid = chat_result.query_result.result_uuid
    logger.info(chat_result.query_result.result_uuid)
    await session.close()
    await sql_engine.dispose()
    await redis_client_async.aclose()
    return db_init


@pytest.fixture(scope="function")
async def chat_session(chat_init):
    """Get a session after the client, database, and chat have been set up"""
    async with get_session(chat_init) as updated_env:
        yield updated_env
