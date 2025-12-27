from typing import Optional
import uuid


from basejump.core.models import schemas as sch
from basejump.core.models import enums
from pydantic import ConfigDict, BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine
from redis.asyncio import Redis as RedisAsync


class GetClient(sch.CreateClient):
    client_id: int
    client_uuid: uuid.UUID
    client_secret: str
    role: enums.AllUserRoles


class GetTeam(sch.TeamFields):
    team_id: int
    team_uuid: uuid.UUID
    model_config = ConfigDict(from_attributes=True)


class GetUser(BaseModel):
    client_id: int
    user_id: int
    user_uuid: uuid.UUID
    username: str
    role: enums.UserRoles
    email_address: str
    model_config = ConfigDict(from_attributes=True)


class GetUserTeam(BaseModel):
    user_id: int
    team_id: int


class SQLDBSchemaBase(sch.DBParamsSchemaBase, sch.DBConnSchema):
    pass


class GetSQLConn(sch.BaseModel):
    conn_id: int
    conn_uuid: uuid.UUID
    db_id: int
    db_uuid: uuid.UUID


class GetChat(sch.BaseModel):
    chat_uuid: uuid.UUID
    chat_id: int
    vector_id: int


class PyTestEnv(sch.BaseModel):
    client_id: int
    client_uuid: uuid.UUID
    team_id: int
    team_uuid: uuid.UUID
    user_id: int
    user_uuid: uuid.UUID
    client_secret: str
    username: str
    team_name: str
    client_user: sch.ClientUserInfo
    team_info: sch.TeamFields
    client_conn_params: Optional[sch.SQLDBSchema] = None
    chat_id: Optional[int] = None
    chat_uuid: Optional[uuid.UUID] = None
    result_uuid: Optional[uuid.UUID] = None
    vector_id: Optional[int] = None
    db_id: Optional[int] = None
    db_uuid: Optional[uuid.UUID] = None
    conn_id: Optional[int] = None
    conn_uuid: Optional[uuid.UUID] = None
    db: Optional[AsyncSession] = None
    redis_client_async: Optional[RedisAsync] = None
    sql_engine: Optional[AsyncEngine] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
