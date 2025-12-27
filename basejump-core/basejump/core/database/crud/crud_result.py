"""Functions to interact with the database for tables related to the result.py endpoint module"""

import uuid
from abc import ABC, abstractmethod
from typing import Optional

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.crud import crud_connection, crud_main
from basejump.core.models import models
from basejump.core.models import schemas as sch
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

logger = set_logging(handler_option="stream", name=__name__)

# CRUD helper functions


class ResultPermissionMgmtBase(ABC):
    """A class to ensure results are getting filtered appropriately"""

    def __init__(self, conn_ids: Optional[list] = None):
        self.conn_ids = conn_ids

    @abstractmethod
    async def results_filter(self, stmt, team_uuids: list, user_uuid: uuid.UUID, include_join: bool = True):
        pass

    def get_permission_filter(self, user_uuid: uuid.UUID, team_uuids: list) -> list:
        and_condition = and_(
            models.SavedResultHistory.author_team_uuid.in_(team_uuids),
            models.SavedResultHistory.share_w_team.is_(True),
        )
        permission_filter = [
            and_condition,
            models.SavedResultHistory.author_user_uuid == user_uuid,
        ]
        if self.conn_ids:
            permission_filter += [
                and_(models.ResultHistory.result_conn_id.in_(self.conn_ids), models.ResultHistory.verified.is_(True))
            ]

        return permission_filter

    async def result_access_check(self, stmt, include_all_client_info: bool, db: AsyncSession, user_uuid: uuid.UUID):
        """Check access for a single result"""
        # Check if it is the authors
        if not include_all_client_info:
            user = await crud_main.get_user(db=db, user_uuid=user_uuid)
            teams = await user.awaitable_attrs.teams
            # Get the shared team results on that team
            team_uuids = [str(team.team_uuid) for team in teams]
            # Get the authors teams and see if it is part of the team
            stmt = self.results_filter(stmt=stmt, team_uuids=team_uuids, user_uuid=user_uuid)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()


class SavedResultPermissionMgmt(ResultPermissionMgmtBase):
    def results_filter(self, stmt, team_uuids: list, user_uuid: uuid.UUID, include_join: bool = True):
        filt = self.get_permission_filter(team_uuids=team_uuids, user_uuid=user_uuid)
        # NOTE: Join is necessary to avoid implicit cartesian joins from SQLAlchemy
        if include_join:
            stmt = stmt.join(
                models.ResultHistory, models.SavedResultHistory.result_id == models.ResultHistory.result_id
            )
        stmt = stmt.filter(or_(*filt))
        return stmt


class ResultPermissionMgmt(ResultPermissionMgmtBase):
    def __init__(self, conn_ids: Optional[list] = None):
        self.conn_ids = conn_ids

    def results_filter(self, stmt, team_uuids: list, user_uuid: uuid.UUID, include_join: bool = True):
        filt = self.get_permission_filter(team_uuids=team_uuids, user_uuid=user_uuid)
        # NOTE: Join is necessary to avoid implicit cartesian joins from SQLAlchemy
        if include_join:
            stmt = stmt.join(
                models.SavedResultHistory,
                models.ResultHistory.result_id == models.SavedResultHistory.result_id,
                isouter=True,
            )
        stmt = stmt.filter(or_(models.ResultHistory.result_author_user_uuid == user_uuid, *filt))
        return stmt


# CRUD functions


async def get_result_filtered(
    db: AsyncSession,
    user_uuid: uuid.UUID,
    include_all_client_info: bool = False,
    result_uuid: Optional[uuid.UUID] = None,
    result_id: Optional[int] = None,
) -> Optional[models.ResultHistory]:
    """This function applies a check to see if the user should have access to the data for a single result"""
    assert result_uuid or result_id, "Need either a result id or a result uuid"
    stmt = select(models.ResultHistory).distinct()
    if result_uuid:
        stmt = stmt.filter_by(result_uuid=result_uuid)
    if result_id:
        stmt = stmt.filter_by(result_id=result_id)
    conn_ids = await crud_connection.get_user_connections(db=db, user_uuid=user_uuid)
    result_mgmt = ResultPermissionMgmt(conn_ids=conn_ids)
    return await result_mgmt.result_access_check(
        db=db, stmt=stmt, include_all_client_info=include_all_client_info, user_uuid=user_uuid
    )


async def save_result_history(
    db: AsyncSession,
    chat_id: int,
    query_result: sch.QueryResult,
    title: str,
    subtitle: str,
    description: str,
    conn_id: int,
    prompt_metadata: sch.PromptMetadata,
    chat_metadata: sch.ChatMetadata,
):
    # Save the report history
    result_hist = models.ResultHistory(
        result_uuid=query_result.result_uuid,
        chat_id=chat_id,
        client_id=prompt_metadata.client_id,
        result_exp_time=query_result.result_exp_time,
        result_conn_id=conn_id,
        result_file_path=query_result.result_file_path,
        row_num_total=query_result.num_rows,
        preview_file_path=query_result.preview_file_path,
        row_num_preview=query_result.preview_row_ct,
        result_title=title,
        result_subtitle=subtitle,
        result_description=description,
        metric_value=query_result.metric_value,
        metric_value_formatted=query_result.metric_value_formatted,
        result_type=query_result.result_type.value,
        result_author_user_uuid=prompt_metadata.user_uuid,
        result_author_team_uuid=chat_metadata.team_uuid,
        sql_query=query_result.sql_query,
        aborted_upload=query_result.aborted_upload,
        parent_msg_uuid=chat_metadata.parent_msg_uuid,
        initial_prompt=prompt_metadata.initial_prompt,
        verified=chat_metadata.semcache_response.verified if chat_metadata.semcache_response else False,
        verified_user_role=(
            chat_metadata.semcache_response.verified_user_role if chat_metadata.semcache_response else None
        ),
        verified_user_uuid=(
            chat_metadata.semcache_response.verified_user_uuid if chat_metadata.semcache_response else None
        ),
        parent_verified_result_uuid=(
            chat_metadata.semcache_response.result_uuid if chat_metadata.semcache_response else None
        ),
    )
    db.add(result_hist)
    await db.commit()  # Needs to be committed so a report is available right away
    await db.refresh(result_hist)
    return result_hist


async def get_visual_result(db: AsyncSession, visual_result_uuid: uuid.UUID) -> models.VisualResultHistory:
    stmt = select(models.VisualResultHistory).filter_by(visual_result_uuid=visual_result_uuid)
    result = await db.execute(stmt)
    return result.scalar_one()


async def get_visual_result_from_result(db: AsyncSession, result_id: int) -> Optional[models.VisualResultHistory]:
    stmt = select(models.VisualResultHistory).filter_by(result_id=result_id)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_result(db: AsyncSession, result_uuid: uuid.UUID) -> models.ResultHistory:
    stmt = select(models.ResultHistory).filter_by(result_uuid=result_uuid)
    result = await db.execute(stmt)
    return result.scalar_one()


async def get_saved_results_using_results(
    db: AsyncSession,
    result_id: Optional[int] = None,
    result_uuid: Optional[uuid.UUID] = None,
    visual_result_id: Optional[int] = None,
    parent_msg_uuid: Optional[uuid.UUID] = None,
) -> Optional[models.SavedResultHistory]:
    stmt = select(models.SavedResultHistory)
    if not visual_result_id and not (result_id or result_uuid):
        return None
    if result_id:
        stmt = stmt.filter(models.SavedResultHistory.result_id == result_id)
    elif result_uuid:
        stmt = stmt.filter(models.SavedResultHistory.result_uuid == result_uuid)
    else:
        raise ValueError("Either result_id or result_uuid need to be defined.")
    if visual_result_id:
        stmt = stmt.filter(models.SavedResultHistory.visual_result_id == visual_result_id)
    if parent_msg_uuid:
        stmt = stmt.filter(models.SavedResultHistory.parent_msg_uuid == parent_msg_uuid)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()
