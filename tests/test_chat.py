import pytest

from basejump.core.database.crud import crud_chat
from basejump.demo import settings, service, schemas
from basejump.core.service import service_utils
from basejump.core.service.agents.context.utils import upload_sql_query_example


@pytest.mark.chat
async def test_getchat(chat_session):
    """Test getting a chat"""
    # Test getting a chat
    initial_chat = await crud_chat.get_chat(
        db=chat_session.db,
        chat_uuid=chat_session.chat_uuid,
        user_id=chat_session.user_id,
    )
    assert initial_chat, "Missing chat"

    # testing getting an empty chat
    chat = await crud_chat.get_chat(
        db=chat_session.db,
        chat_uuid=chat_session.chat_uuid,
        user_id=chat_session.user_id,
        empty_chats_only=True,
    )
    assert not chat  # no empty chats

    # test getting all chats for a user
    chats = await crud_chat.get_chats(
        db=chat_session.db,
        user_id=chat_session.user_id,
    )
    assert chats

    # test getting all empty chats for a user
    chats = await crud_chat.get_chats(db=chat_session.db, user_id=chat_session.user_id, empty_chats_only=True)
    assert not chats  # Should be empty

    # test chat messages
    redis_client_async = settings.get_redis_client_async_instance()
    await crud_chat.delete_chat_msgs_from_vector(
        db=chat_session.db,
        client_id=chat_session.client_id,
        msg_uuids=[msg.msg_uuid for msg in await initial_chat.awaitable_attrs.msgs],
        redis_client_async=redis_client_async,
    )
    await redis_client_async.aclose()


@pytest.mark.chat
async def test_getviz(chat_session):
    """Test getting a visual result"""

    # Set chat variables
    get_chat = schemas.GetChat(
        chat_uuid=chat_session.chat_uuid,
        chat_id=chat_session.chat_id,
        vector_id=chat_session.vector_id,
    )

    # Get a visual result
    chat_response = await service.chat(
        db=chat_session.db,
        prompt="Give me a bar chart of count of clients by type",
        service_context=chat_session.service_context,
        user_info=chat_session.user_info,
        chat=get_chat,
    )
    assert chat_response.query_result, "Missing chat query result"
    assert chat_response.query_result.visual_json

    # Get a chat
    chat = await crud_chat.get_chat(
        db=chat_session.db,
        chat_uuid=chat_session.chat_uuid,
        user_id=chat_session.user_id,
        include_all_client_info=True,
    )
    assert chat, "Missing chat"

    # Get a message
    messages = await chat.awaitable_attrs.msgs
    msg_uuid = messages[0].msg_uuid
    message = await crud_chat.get_message(db=chat_session.db, msg_uuid=msg_uuid)
    assert message


@pytest.mark.chat
async def test_get_trust_score(chat_session):
    """Test getting a trust score"""
    result = await service_utils.calc_trust_score(db=chat_session.db)
    assert result


@pytest.mark.chat
async def test_save_sql_query(chat_session):
    """Test saving a SQL query"""
    await upload_sql_query_example(
        sql_query="select * from account.client",
        client_user=chat_session.client_user,
        prompt="Get me a list of all clients",
        db_uuid=chat_session.client_user,
        redis_client_async=chat_session.redis_client_async,
        small_model_info=chat_session.service_context.small_model_info,
    )
    breakpoint()
    get_chat = schemas.GetChat(
        chat_uuid=chat_session.chat_uuid,
        chat_id=chat_session.chat_id,
        vector_id=chat_session.vector_id,
    )

    await service.chat(
        db=chat_session.db,
        prompt="Give me a list of all clients",
        service_context=chat_session.service_context,
        user_info=chat_session.user_info,
        chat=get_chat,
    )
