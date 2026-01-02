import pytest

from basejump.core.database.crud import crud_chat
from basejump.demo import settings, service, schemas
from basejump.core.service import service_utils
from basejump.core.models import schemas as sch


@pytest.mark.chat
async def test_getchat(chat_session):
    """Test getting a chat"""
    # Test getting a chat
    initial_chat = await crud_chat.get_chat(
        db=chat_session.db,
        chat_uuid=chat_session.chat_uuid,
        user_id=chat_session.user_id,
    )

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

    # Get a visual result
    user_info = schemas.UserInfo(
        client_id=chat_session.client_id,
        client_uuid=chat_session.client_id,
        user_id=chat_session.user_id,
        user_uuid=chat_session.user_uuid,
        user_uuid=chat_session.user_role,
        team_id=chat_session.team_id,
        team_uuid=chat_session.team_uuid,
    )
    service_context = sch.ServiceContext(
        sql_engine=chat_session.core_session.sql_engine,
        redis_client_async=chat_session.core_session.redis_client_async,
        db=chat_session.core_session.db,
        large_model_info=settings.large_model_info,
        small_model_info=settings.small_model_info,
        embedding_model_info=settings.embedding_model_info,
    )
    chat_response = await service.chat(
        prompt="Give me a bar chart of count of clients by type",
        service_context=service_context,
        user_info=user_info,
        allow_unrestricted_db_chat=False,
    )
    assert chat_response.query_result.visual_json

    # Get a chat
    chat_response = await crud_chat.get_chat(
        db=chat_session.db,
        chat_uuid=chat_session.chat_uuid,
        user_id=chat_session.user_id,
        include_all_client_info=True,
    )
    assert chat_response

    # Get a message
    messages = await chat_response.awaitable_attrs.msgs
    msg_uuid = messages[0].msg_uuid
    message = await crud_chat.get_message(db=chat_session.db, msg_uuid=msg_uuid)
    assert message


@pytest.mark.chat
async def test_get_trust_score(chat_session):
    """Test getting a trust score"""
    result = await service_utils.calc_trust_score(db=chat_session.db)
    assert result
