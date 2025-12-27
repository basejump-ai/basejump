import pytest

from basejump.core.database.crud import crud_result


@pytest.mark.result
async def test_get_results(chat_session):
    """Test getting results"""
    # Test getting a filtered result
    await crud_result.get_result_filtered(
        db=chat_session.db,
        user_uuid=chat_session.user_uuid,
        result_uuid=chat_session.result_uuid,
    )

    # Test getting a result
    await crud_result.get_result(
        db=chat_session.db,
        result_uuid=chat_session.result_uuid,
    )
