import pytest

from basejump.core.database.crud import crud_main


@pytest.mark.main
async def test_get_team(client_session):
    await crud_main.get_team(db=client_session.db, team_uuid=client_session.team_uuid)


@pytest.mark.main
async def test_get_user(client_session):
    await crud_main.get_user(db=client_session.db, user_uuid=client_session.user_uuid)
