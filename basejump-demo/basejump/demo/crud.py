from basejump.core.models import models
from sqlalchemy.ext.asyncio import AsyncSession
from basejump.core.common.config.logconfig import set_logging

logger = set_logging(handler_option="stream", name=__name__)


async def add_user_to_team(db: AsyncSession, username: str, team_name: str, user_id: int, team_id: int) -> None:
    db_user_team = models.UserTeamAssociation(user_id=user_id, team_id=team_id)
    db.add(db_user_team)
    await db.commit()
    await db.refresh(db_user_team)
    logger.info(f"Added user {username} to team {team_name}")


async def add_connection_to_team(db: AsyncSession, client_id: int, team_id: int, conn_id: int):
    assoc = models.ConnTeamAssociation(client_id=client_id, team_id=team_id, conn_id=conn_id)
    db.add(assoc)
    await db.commit()
