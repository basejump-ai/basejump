"""
Functions to interact with the database for tables related to the main.py endpoint module
and account level tables
"""

import uuid
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.db_connect import LocalSession
from basejump.core.models import enums, errors, models
from basejump.core.models import schemas as sch

logger = set_logging(handler_option="stream", name=__name__)


async def get_user(db: AsyncSession, user_uuid: uuid.UUID):
    """Get a user object"""
    # Use scalar here since a user can be associated with multiple teams
    user = await db.execute(select(models.User).filter_by(user_uuid=user_uuid))
    return user.scalar_one_or_none()


async def get_user_from_id(db: AsyncSession, user_id: int):
    """Get a user object"""
    # Use scalar here since a user can be associated with multiple teams
    user = await db.execute(select(models.User).filter_by(user_id=user_id))
    return user.scalar_one_or_none()


async def create_user(db: AsyncSession, user: sch.BaseUser, user_id: Optional[int] = None):
    # Create dictionary for user arguments
    user_dict = user.model_dump()
    if user_id is not None:
        user_dict["user_id"] = user_id
        # Confirm user is already not in the database
        user_exists = await get_user_from_id(db=db, user_id=user_id)
        if user_exists:
            raise errors.AlreadyExists("The user already exists")

    # Add to database
    db_user = models.User(**user_dict)
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)

    logger.info(f"Created new user with username: {db_user.username} and user ID: {db_user.user_id}")  # noqa
    return db_user


async def get_team(db: AsyncSession, team_uuid: uuid.UUID):
    """Get a team"""
    team = await db.execute(select(models.Team).filter_by(team_uuid=team_uuid))
    return team.scalar_one_or_none()


async def get_team_from_id(db: AsyncSession, team_id: int):
    """Get a team"""
    team = await db.execute(select(models.Team).filter_by(team_id=team_id))
    return team.scalar_one_or_none()


async def create_team(db: AsyncSession, team: sch.BaseTeam, team_id: Optional[int] = None) -> models.Team:
    # Create dictionary for team arguments
    team_dict = team.model_dump()
    if team_id is not None:
        team_dict["team_id"] = team_id
        # Confirm team is already not in the database
        team_exists = await get_team_from_id(db=db, team_id=team_id)
        if team_exists:
            raise errors.AlreadyExists("The team already exists")

    # Add to database
    db_team = models.Team(**team_dict)
    db.add(db_team)
    await db.commit()
    await db.refresh(db_team)
    logger.info(f"Created new team with name: {db_team.team_name} and team ID: {db_team.team_id}")  # noqa
    return db_team


async def get_client(db: AsyncSession, client_uuid: uuid.UUID) -> Optional[models.Client]:
    client = await db.execute(select(models.Client).filter_by(client_uuid=client_uuid))
    return client.scalar_one_or_none()


async def get_client_from_id(db: AsyncSession, client_id: int) -> Optional[models.Client]:
    client = await db.execute(select(models.Client).filter_by(client_id=client_id))
    return client.scalar_one_or_none()


async def create_client(
    db: AsyncSession,
    client: sch.CreateClient,
    sql_engine: AsyncEngine,
    description: Optional[str] = None,
    client_id: Optional[int] = None,
) -> sch.NewClient:
    # Create dictionary for client arguments
    client_dict = client.model_dump(exclude={"description", "hashed_client_secret"})
    if client_id is not None:
        client_dict["client_id"] = client_id
        # Confirm client is already not in the database
        client_exists = await get_client_from_id(db=db, client_id=client_id)
        if client_exists:
            raise errors.AlreadyExists("The client already exists")

    # Add client to the database
    db_client = models.Client(**client_dict)
    db.add(db_client)
    await db.commit()
    await db.refresh(db_client)
    client_assoc = models.ClientSecretAssociation(
        client_id=db_client.client_id,
        hashed_client_secret=client.hashed_client_secret,
        role=enums.APIUserRoles.INTERNAL.value,
        description=description,
    )
    db.add(client_assoc)

    session = LocalSession(client_id=db_client.client_id, engine=sql_engine)
    await session.create_schemas()  # Create client-specific schemas
    await session.create_views()  # Create client-specific views

    # Logging
    logger.info(f"Created new client with name: {db_client.client_name} and client ID: {db_client.client_id}")  # noqa
    new_client = sch.NewClientBase(
        **client.dict(), client_id=db_client.client_id, client_uuid=str(db_client.client_uuid)
    )
    await db.commit()
    await db.refresh(client_assoc)
    return sch.NewClient(**new_client.dict(), client_secret_uuid=str(client_assoc.client_secret_uuid))
