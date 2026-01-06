"""Utilities that support the AI functionality or other core business logic within the application"""

import os

from cryptography.fernet import Fernet
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.exc import NoResultFound

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.crud import crud_chat, crud_connection
from basejump.core.models import errors
from basejump.core.models import schemas as sch

logger = set_logging(handler_option="stream", name=__name__)


async def calc_trust_score(db: AsyncSession, number_of_days: int = 7) -> sch.TrustScore:
    try:
        row = await crud_chat.get_thumb_reaction_counts(db=db, number_of_days=number_of_days)
    except NoResultFound:
        total_messages = 0
    total_messages = row.total_messages if row.total_messages else 0
    thumbs_down_count = row.thumbs_down_count if row.thumbs_down_count else 0

    trust_score = 1.00
    if total_messages > 0:
        trust_score = 1 - (thumbs_down_count / total_messages)

    trust_score_obj = sch.TrustScore(
        total_messages=total_messages, thumbs_down_count=thumbs_down_count, trust_score=round(trust_score, 2)
    )
    return trust_score_obj


async def get_client_active_storage_conn(db: AsyncSession, client_id: int) -> sch.ClientStorageConn:
    """Get the active client storage connection + decrypt sensitive fields"""
    storage_conn = await crud_connection.get_client_active_storage_conn(db=db, client_id=client_id)
    if not storage_conn:
        raise errors.NotFoundError("No active client storage connection found")
    storage_conn_schema = sch.ClientStorageConnEncrypted.model_validate(storage_conn)
    storage_conn_dict = storage_conn_schema.model_dump(exclude={"access_key", "secret_access_key"})
    try:
        f = Fernet(os.environ["ENCRYPTION_KEY"])
    except KeyError:
        raise errors.MissingEnvironmentVariable("Missing the ENCRYPTION_KEY environment variable.")
    storage_conn_dict["access_key"] = f.decrypt(storage_conn.access_key).decode("utf-8")
    storage_conn_dict["secret_access_key"] = f.decrypt(storage_conn.secret_access_key).decode("utf-8")
    return sch.ClientStorageConn.model_validate(storage_conn_dict)
