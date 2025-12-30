"""Models defining the structure of the database"""

import os
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Optional

import sqlalchemy as sa
from basejump.core.common.config.logconfig import set_logging
from basejump.core.models import enums
from basejump.core.models import schemas as sch
from cryptography.fernet import Fernet
from pydantic import BaseModel, ConfigDict
from sqlalchemy import TIMESTAMP, UUID, ForeignKey, String, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.schema import ForeignKeyConstraint
from sqlalchemy.sql import func
from sqlalchemy.types import BIGINT, Integer
from typing_extensions import Annotated

logger = set_logging(handler_option="stream", name=__name__)


# =======================================================
# Encryption for any models requiring an encrypted column
# =======================================================
class Encrypted(sa.TypeDecorator):
    impl = sa.LargeBinary
    cache_ok = True

    def __init__(self, encryption_key: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encryption_key = encryption_key
        self.fernet = Fernet(encryption_key)

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = self.fernet.encrypt(value.encode("utf-8"))
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = self.fernet.decrypt(value).decode("utf-8")
        return value


# TODO: Update more tables to use StrEnums

big_int = Annotated[int, "bigint"]


class Base(AsyncAttrs, DeclarativeBase):
    type_annotation_map = {
        dict: postgresql.JSONB,
        datetime: TIMESTAMP(timezone=True),
        list: postgresql.ARRAY(Integer),
        big_int: BIGINT,
        list[dict]: postgresql.JSONB,
    }


# =====================
# Account Models
# =====================


class Client(Base):
    """
    A client in the Basejump product.
    This is the highest account grouping level."""

    __tablename__ = "client"
    __table_args__ = {"schema": "account"}

    client_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    client_uuid: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), server_default=text("gen_random_uuid()"), unique=True
    )
    client_name: Mapped[str]
    client_type: Mapped[enums.ClientType]
    verify_mode: Mapped[enums.VerifyMode] = mapped_column(server_default=enums.VerifyMode.EXPLORE.value)
    llm: Mapped[enums.AIModelSchemaClientOptions] = mapped_column(
        sa.Enum(
            enums.AIModelSchemaClientOptions,
            values_callable=lambda enum: [e.value for e in enum],
            name="aimodelschemas",
        ),
        server_default=text(f"'{enums.AIModelSchemaClientOptions.GPT41.value}'"),
    )
    timestamp: Mapped[datetime] = mapped_column(server_default=func.now())
    connections: Mapped[list["Connection"]] = relationship(
        back_populates="client",
        lazy="selectin",  # codespell:ignore selectin
        passive_deletes="all",
    )


class Team(Base):
    """A team in the Basejump product. A team controls access to data."""

    __tablename__ = "team"
    __table_args__ = {"schema": "account"}

    client_id: Mapped[int] = mapped_column(ForeignKey("account.client.client_id", ondelete="CASCADE"))
    team_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, unique=True)
    team_uuid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), server_default=text("gen_random_uuid()"))
    team_name: Mapped[str]
    team_desc: Mapped[str]  # A team description to provide to the AI
    timestamp: Mapped[datetime] = mapped_column(server_default=func.now())

    team_users: Mapped[list["UserTeamAssociation"]] = relationship(
        back_populates="team",
        lazy="selectin",  # codespell:ignore selectin
        passive_deletes="all",
    )
    team_connection_associations: Mapped[list["ConnTeamAssociation"]] = relationship(
        back_populates="team",
        lazy="selectin",  # codespell:ignore selectin
        passive_deletes="all",
    )


class User(Base):
    """A user of the Basejump product"""

    __tablename__ = "user"
    __table_args__ = (
        sa.PrimaryKeyConstraint("client_id", "user_id"),
        {"schema": "account"},
    )

    client_id = mapped_column(ForeignKey("account.client.client_id", ondelete="CASCADE"))
    user_id: Mapped[int] = mapped_column(autoincrement=True, unique=True)
    user_uuid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), server_default=text("gen_random_uuid()"))
    service_user_uuid: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True))
    username: Mapped[str]
    email_address: Mapped[Optional[str]]  # Optional since it was added later and not all users are mapped over
    hashed_refresh_token: Mapped[Optional[str]]
    role: Mapped[enums.UserRoles]
    timestamp: Mapped[datetime] = mapped_column(server_default=func.now())
    # Relationship to Team
    user_teams: Mapped[list["UserTeamAssociation"]] = relationship(
        back_populates="user",
        lazy="selectin",  # codespell:ignore selectin
        passive_deletes="all",
    )
    teams = association_proxy("user_teams", "team")


# =====================
# Connection Models
# =====================


class Connection(Base):
    __tablename__ = "connection"
    __table_args__ = (
        sa.UniqueConstraint("client_id", "conn_id"),
        {"schema": "connect"},
    )

    client_id: Mapped[int] = mapped_column(
        ForeignKey("account.client.client_id", ondelete="CASCADE"), primary_key=True
    )
    conn_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conn_uuid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), server_default=text("gen_random_uuid()"))
    conn_type: Mapped[str]
    data_source_desc: Mapped[str]

    client: Mapped["Client"] = relationship(back_populates="connections", lazy="joined", passive_deletes="all")

    __mapper_args__ = {
        "polymorphic_identity": "connection",
        "polymorphic_on": "conn_type",
    }


class DBParams(Base):
    """Holds the information to connect to a SQL database"""

    __tablename__ = "database"
    __table_args__ = {"schema": "connect"}

    client_id: Mapped[int] = mapped_column(ForeignKey("account.client.client_id", ondelete="CASCADE"))
    db_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    db_uuid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), server_default=text("gen_random_uuid()"))
    vector_id: Mapped[int] = mapped_column(ForeignKey("connect.vector_db.vector_id", ondelete="CASCADE"))
    database_type: Mapped[bytes]
    drivername: Mapped[bytes]
    host: Mapped[bytes]
    database_name: Mapped[bytes]
    database_name_alias: Mapped[bytes]
    database_name_alias_number: Mapped[int] = mapped_column(server_default="0")
    port: Mapped[Optional[bytes]]
    database_desc: Mapped[bytes]
    query: Mapped[Optional[bytes]]
    ssl_mode: Mapped[enums.SSLModes]
    ssl_root_cert: Mapped[Optional[str]]
    ssl: Mapped[Optional[bool]] = mapped_column(server_default=text("true"))
    schemas: Mapped[
        Optional[bytes]
    ]  # TODO: This needs to be a different object than conn schemas since it won't have jinja values
    table_filter_string: Mapped[Optional[str]]
    include_default_schema: Mapped[Optional[bool]] = mapped_column(server_default=text("true"))
    include_views: Mapped[Optional[bool]] = mapped_column(server_default=text("false"))
    include_materialized_views: Mapped[Optional[bool]] = mapped_column(server_default=text("false"))
    include_partitioned_tables: Mapped[Optional[bool]] = mapped_column(server_default=text("false"))
    timestamp: Mapped[datetime] = mapped_column(server_default=func.now())
    connections: Mapped[list["DBConn"]] = relationship(
        passive_deletes="all",
        back_populates="database_params",  # codespell:ignore selectin
    )
    tables: Mapped[list["DBTables"]] = relationship(passive_deletes="all")  # codespell:ignore selectin


class DBConn(Connection):
    """Holds the information to connect to a SQL database"""

    __tablename__ = "db_conn"
    __table_args__ = (  # type: ignore
        ForeignKeyConstraint(
            ["client_id", "conn_id"],
            ["connect.connection.client_id", "connect.connection.conn_id"],
            ondelete="CASCADE",
        ),
        {"schema": "connect"},
    )

    client_id: Mapped[int]
    conn_id: Mapped[int] = mapped_column(primary_key=True)
    db_id: Mapped[int] = mapped_column(ForeignKey("connect.database.db_id", ondelete="CASCADE"))
    schemas: Mapped[Optional[list[dict]]]
    username: Mapped[bytes]
    password: Mapped[bytes]
    timestamp: Mapped[datetime] = mapped_column(server_default=func.now())

    database_params: Mapped["DBParams"] = relationship(passive_deletes="all", back_populates="connections")
    tables_assoc: Mapped[list["ConnTableAssociation"]] = relationship(
        back_populates="connections", passive_deletes="all"  # codespell:ignore selectin
    )
    tables = association_proxy("tables_assoc", "tables")
    __mapper_args__ = {
        "polymorphic_identity": enums.ConnectionType.SQL.value,
    }


class DBVector(Base):
    """
    Holds the information of where the vector indexes are stored
    along with the information to connect to them
    """

    __tablename__ = "vector_db"
    __table_args__ = {"schema": "connect"}
    client_id: Mapped[int] = mapped_column(ForeignKey("account.client.client_id", ondelete="CASCADE"))
    vector_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    vector_uuid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), server_default=text("gen_random_uuid()"))
    vector_database_vendor: Mapped[str]
    vector_datasource_type: Mapped[str]
    index_name: Mapped[str]
    vector_metadata: Mapped[Optional[dict]]
    # TODO: Set all times to be stored in UTC as a standard
    # This can be done like this: server_default=text("timezone('utc', now())")
    timestamp: Mapped[datetime] = mapped_column(server_default=func.now())


class DBTables(Base):
    """Holds the SQL tables related to a SQL DB connection"""

    __tablename__ = "database_tables"
    __table_args__ = {"schema": "connect"}

    client_id: Mapped[int] = mapped_column(
        ForeignKey("account.client.client_id", ondelete="CASCADE"), primary_key=True
    )
    tbl_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tbl_uuid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), server_default=text("gen_random_uuid()"))
    db_id: Mapped[int] = mapped_column(ForeignKey("connect.database.db_id", ondelete="CASCADE"))
    # TODO: Break out the schema name
    table_name: Mapped[str]  # This includes the schema aka the full_table_name
    context: Mapped[Optional[str]]
    ignore: Mapped[Optional[bool]] = mapped_column(server_default=text("false"))
    primary_keys: Mapped[Optional[list[str]]] = mapped_column(postgresql.ARRAY(String))
    timestamp: Mapped[datetime] = mapped_column(server_default=func.now())
    columns: Mapped[list["DBTableColumns"]] = relationship(passive_deletes="all")  # codespell:ignore selectin
    connection: Mapped[list["ConnTableAssociation"]] = relationship(
        back_populates="tables",
        lazy="selectin",  # codespell:ignore selectin
        passive_deletes="all",
    )


class DBTableColumns(Base):
    """Holds the columns related to a SQL DB table"""

    __tablename__ = "table_columns"
    __table_args__ = (
        # NOTE: Needs to be table args for the relationships to recognize a foreign key is actually defined
        ForeignKeyConstraint(
            ["client_id", "tbl_id"],
            ["connect.database_tables.client_id", "connect.database_tables.tbl_id"],
            ondelete="CASCADE",
        ),
        {"schema": "connect"},
    )

    # NOTE: table_columns and a few other tables need client_id as part of the primary key for partition key purposes
    client_id: Mapped[int] = mapped_column(primary_key=True)
    col_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    col_uuid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), server_default=text("gen_random_uuid()"))
    tbl_id: Mapped[int]
    column_name: Mapped[str]
    column_type: Mapped[str]
    description: Mapped[Optional[str]]
    foreign_key_table_name: Mapped[Optional[str]]
    foreign_key_column_name: Mapped[Optional[str]]
    primary_key: Mapped[Optional[bool]]
    distinct_values: Mapped[Optional[list[str]]] = mapped_column(postgresql.ARRAY(String))
    ignore: Mapped[Optional[bool]] = mapped_column(server_default=text("false"))
    timestamp: Mapped[datetime] = mapped_column(server_default=func.now())
    quoted: Mapped[Optional[bool]] = mapped_column(server_default=text("false"))


# =====================
# AI Models
# =====================
class Chat(Base):
    """Contains chat ID information related to a given user"""

    __tablename__ = "user_chat_id"
    __table_args__ = {"schema": "ai_model"}

    client_id: Mapped[int] = mapped_column(
        ForeignKey("account.client.client_id", ondelete="CASCADE"), primary_key=True
    )
    chat_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    chat_uuid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), server_default=text("gen_random_uuid()"))
    user_id: Mapped[int] = mapped_column(ForeignKey("account.user.user_id", ondelete="CASCADE"))
    team_id: Mapped[int] = mapped_column(ForeignKey("account.team.team_id", ondelete="CASCADE"))
    chat_in_index: Mapped[bool] = mapped_column(server_default=text("false"))
    chat_name: Mapped[Optional[str]]
    chat_description: Mapped[Optional[str]]
    # Don't want to cascade delete since deleting the vector index doesn't necessarily mean we want to delete the chat
    vector_id: Mapped[Optional[int]] = mapped_column(ForeignKey("connect.vector_db.vector_id"))
    timestamp: Mapped[datetime] = mapped_column(server_default=func.current_timestamp())
    last_updt_date: Mapped[Optional[datetime]]

    msgs: Mapped[list["ChatHistory"]] = relationship(passive_deletes="all")


# TODO: Possibly refactor with ChatHistory to make it more normalized
# This is primarily here to ensure prompt_uuid integrity between all of the tables
# TODO: Move initial prompt out of chat history and result history into this model
class PromptHistory(Base):
    __tablename__ = "prompt_history"
    __table_args__ = (
        sa.UniqueConstraint("client_id", "prompt_uuid"),
        {"schema": "ai_model"},
    )

    client_id: Mapped[int] = mapped_column(
        ForeignKey("account.client.client_id", ondelete="CASCADE"), primary_key=True
    )
    prompt_id: Mapped[big_int] = mapped_column(sa.BigInteger, sa.Identity(), autoincrement=True, primary_key=True)
    prompt_uuid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), server_default=text("gen_random_uuid()"))
    llm_type: Mapped[enums.LLMType]


class ChatHistory(Base):
    """Contains chat history for a given chat ID"""

    __tablename__ = "chat_history"
    __table_args__ = (
        ForeignKeyConstraint(
            ["client_id", "chat_id"],
            ["ai_model.user_chat_id.client_id", "ai_model.user_chat_id.chat_id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["client_id", "prompt_uuid"],
            [
                "ai_model.prompt_history.client_id",
                "ai_model.prompt_history.prompt_uuid",
            ],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("client_id", "msg_uuid"),
        {"schema": "ai_model"},
    )

    client_id: Mapped[int] = mapped_column(primary_key=True)
    msg_id: Mapped[big_int] = mapped_column(sa.BigInteger, sa.Identity(), autoincrement=True, primary_key=True)
    msg_uuid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), server_default=text("gen_random_uuid()"))
    msg_in_index: Mapped[bool] = mapped_column(server_default=text("false"))
    parent_msg_uuid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    result_uuid: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True))
    visual_result_uuid: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True))
    chat_id: Mapped[int]
    # TODO: Switch this for prompt_id instead
    prompt_uuid: Mapped[uuid.UUID]
    initial_prompt: Mapped[str]
    prompt_time: Mapped[datetime] = mapped_column(server_default=func.current_timestamp())
    content: Mapped[str]
    internal_content: Mapped[Optional[str]]  # BC v0.26.1
    role: Mapped[Optional[str]]
    msg_type: Mapped[Optional[str]]  # TODO: Make this an enum
    sql_query: Mapped[Optional[str]]
    result_type: Mapped[Optional[str]]
    thumbs_up: Mapped[Optional[bool]]
    timestamp: Mapped[datetime] = mapped_column(server_default=func.current_timestamp())


# TODO: This could eventually just be a polymorphic relationship with SavedResultHistory
class ResultHistory(Base):
    """Contains report history for a given message ID"""

    __tablename__ = "result_history"
    __table_args__ = (
        ForeignKeyConstraint(
            ["client_id", "result_conn_id"],
            ["connect.connection.client_id", "connect.connection.conn_id"],
            ondelete="CASCADE",
        ),
        {"schema": "ai_model"},
    )
    client_id: Mapped[int] = mapped_column(
        ForeignKey("account.client.client_id", ondelete="CASCADE"), primary_key=True
    )
    result_id: Mapped[big_int] = mapped_column(sa.BigInteger, sa.Identity(), primary_key=True, autoincrement=True)
    result_uuid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), server_default=text("gen_random_uuid()"))
    chat_id: Mapped[
        Optional[int]
    ]  # BC v0.27.1 Did not start relating to chat until v0.27.1 so all results before that have null chat IDs
    parent_msg_uuid: Mapped[Optional[uuid.UUID]]
    result_conn_id: Mapped[int]
    result_exp_time: Mapped[datetime]
    result_file_path: Mapped[str]
    row_num_total: Mapped[int]
    preview_file_path: Mapped[str]
    row_num_preview: Mapped[int]
    result_type: Mapped[str]
    result_title: Mapped[str]
    result_subtitle: Mapped[str]
    result_description: Mapped[str]
    metric_value: Mapped[Optional[str]]
    metric_value_formatted: Mapped[Optional[str]]
    refresh_result: Mapped[Optional[bool]] = mapped_column(server_default=text("false"))
    sql_query: Mapped[str]
    aborted_upload: Mapped[bool] = mapped_column(server_default=text("false"))
    result_author_user_uuid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    result_author_team_uuid: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True))
    initial_prompt: Mapped[Optional[str]]
    verified: Mapped[bool] = mapped_column(server_default=text("false"))
    verified_user_role: Mapped[Optional[enums.UserRoles]]
    verified_user_uuid: Mapped[Optional[uuid.UUID]]
    parent_verified_result_uuid: Mapped[Optional[uuid.UUID]]
    timestamp: Mapped[datetime] = mapped_column(server_default=func.current_timestamp())


class VisualResultHistory(Base):
    """Contains the visual report for a given parent message ID"""

    __tablename__ = "visual_result_history"
    __table_args__ = (
        ForeignKeyConstraint(
            ["client_id", "result_id"],
            ["ai_model.result_history.client_id", "ai_model.result_history.result_id"],
            ondelete="CASCADE",
        ),
        {"schema": "ai_model"},
    )
    client_id: Mapped[int] = mapped_column(
        ForeignKey("account.client.client_id", ondelete="CASCADE"), primary_key=True
    )
    visual_result_id: Mapped[big_int] = mapped_column(
        sa.BigInteger, sa.Identity(), primary_key=True, autoincrement=True
    )
    visual_result_uuid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), server_default=text("gen_random_uuid()"))
    parent_msg_uuid: Mapped[Optional[uuid.UUID]]
    result_id: Mapped[int]
    result_uuid: Mapped[uuid.UUID]
    visual_json: Mapped[dict]
    visual_explanation: Mapped[str]
    timestamp: Mapped[datetime] = mapped_column(server_default=func.current_timestamp())

    saved_results: Mapped[list["SavedResultHistory"]] = relationship(back_populates="visual_result")


class SavedResultHistory(Base):
    """This table stores saved result information that links visualizations and datasets together"""

    __tablename__ = "saved_result_history"
    __table_args__ = (
        ForeignKeyConstraint(
            ["client_id", "result_id"],
            ["ai_model.result_history.client_id", "ai_model.result_history.result_id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["client_id", "visual_result_id"],
            ["ai_model.visual_result_history.client_id", "ai_model.visual_result_history.visual_result_id"],
            ondelete="CASCADE",
        ),
        {"schema": "ai_model"},
    )
    client_id: Mapped[int] = mapped_column(
        ForeignKey("account.client.client_id", ondelete="CASCADE"), primary_key=True
    )
    saved_result_id: Mapped[big_int] = mapped_column(
        sa.BigInteger, sa.Identity(), primary_key=True, autoincrement=True
    )
    saved_result_uuid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), server_default=text("gen_random_uuid()"))
    result_id: Mapped[int]
    result_uuid: Mapped[uuid.UUID]
    visual_result_id: Mapped[Optional[int]]
    visual_result_uuid: Mapped[Optional[uuid.UUID]]
    parent_msg_uuid: Mapped[
        Optional[uuid.UUID]
    ]  # B.C. v0.28.0 Really should be required, but have some nulls from before migration
    title: Mapped[str]
    subtitle: Mapped[str]
    description: Mapped[str]
    share_w_team: Mapped[bool] = mapped_column(server_default=text("false"))
    author_user_uuid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    author_team_uuid: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True))
    timestamp: Mapped[datetime] = mapped_column(server_default=func.current_timestamp())

    result: Mapped["ResultHistory"] = relationship(lazy="joined", passive_deletes="all", overlaps="saved_results")
    visual_result: Mapped["VisualResultHistory"] = relationship(
        lazy="joined", passive_deletes="all", overlaps="result"
    )


class TokenCount(Base):
    """Contains the token count for a given message UUID and user UUID"""

    __tablename__ = "token_count"
    __table_args__ = (
        ForeignKeyConstraint(
            ["client_id", "prompt_id"],
            ["ai_model.prompt_history.client_id", "ai_model.prompt_history.prompt_id"],
            ondelete="CASCADE",
        ),
        {"schema": "ai_model"},
    )
    client_id: Mapped[int] = mapped_column(primary_key=True)
    token_id: Mapped[big_int] = mapped_column(sa.BigInteger, sa.Identity(), primary_key=True, autoincrement=True)
    token_uuid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), server_default=text("gen_random_uuid()"))
    prompt_id: Mapped[int]
    prompt: Mapped[str]
    ai_model_provider: Mapped[str]
    ai_model_nm: Mapped[str]  # TODO: Need an enum for the AICatalog names
    cost_per_1k_tokens_input: Mapped[Decimal]
    cost_per_1k_tokens_output: Mapped[Decimal]
    total_embedding_token_count: Mapped[Decimal]
    prompt_llm_token_count: Mapped[Decimal]
    completion_llm_token_count: Mapped[Decimal]
    total_llm_token_count: Mapped[Decimal]
    timestamp: Mapped[datetime] = mapped_column(server_default=func.current_timestamp())


class ClientStorageConnection(Base):
    __tablename__ = "client_storage_conn"
    __table_args__ = (
        ForeignKeyConstraint(
            ["client_id"],
            ["account.client.client_id"],
            ondelete="CASCADE",
        ),
        {"schema": "connect"},
    )
    client_id: Mapped[int]
    storage_uuid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), server_default=text("gen_random_uuid()"))
    storage_id: Mapped[big_int] = mapped_column(sa.BigInteger, sa.Identity(), primary_key=True, autoincrement=True)
    alias: Mapped[str]
    storage_provider: Mapped[str]
    region: Mapped[str]
    bucket_name: Mapped[str]
    prefix: Mapped[str]
    access_key: Mapped[Encrypted] = sa.orm.mapped_column(Encrypted(os.environ["ENCRYPTION_KEY"]))
    secret_access_key: Mapped[Encrypted] = sa.orm.mapped_column(Encrypted(os.environ["ENCRYPTION_KEY"]))
    active: Mapped[bool] = mapped_column(server_default=text("false"))
    internal: Mapped[bool]


class DBCredentials(BaseModel):
    db_login: DBConn
    db_params: DBParams

    class Config:
        arbitrary_types_allowed = True


# =====================
# Association Tables
# =====================
class UserTeamAssociation(Base):
    """An association table to add and delete users from teams"""

    __tablename__ = "user_team"
    __table_args__ = {"schema": "account"}

    user_id: Mapped[int] = mapped_column(
        ForeignKey("account.user.user_id", ondelete="CASCADE"),
        primary_key=True,
    )
    team_id: Mapped[int] = mapped_column(
        ForeignKey("account.team.team_id", ondelete="CASCADE"),
        primary_key=True,
    )

    # Define relationships
    user: Mapped["User"] = relationship(back_populates="user_teams", lazy="joined", passive_deletes="all")
    team: Mapped["Team"] = relationship(back_populates="team_users", lazy="joined", passive_deletes="all")


class ConnTeamAssociation(Base):
    """An association table to add and delete connections from teams"""

    __tablename__ = "team_connection"
    __table_args__ = (
        ForeignKeyConstraint(
            ["client_id", "conn_id"],
            ["connect.connection.client_id", "connect.connection.conn_id"],
            ondelete="CASCADE",
        ),
        {"schema": "connect"},
    )

    client_id: Mapped[int] = mapped_column(primary_key=True)
    conn_id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(
        ForeignKey(
            "account.team.team_id",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )

    # Define relationships
    team: Mapped["Team"] = relationship(
        back_populates="team_connection_associations",
        lazy="joined",
        passive_deletes="all",
    )
    connection: Mapped["Connection"] = relationship(lazy="joined", passive_deletes="all")


class ConnTableAssociation(Base):
    """An association table to add and delete tables from connections"""

    __tablename__ = "table_connection"
    __table_args__ = (
        ForeignKeyConstraint(
            ["client_id", "tbl_id"],
            ["connect.database_tables.client_id", "connect.database_tables.tbl_id"],
            ondelete="CASCADE",
        ),
        {"schema": "connect"},
    )

    client_id: Mapped[int] = mapped_column(
        ForeignKey("account.client.client_id", ondelete="CASCADE"), primary_key=True
    )
    conn_id: Mapped[int] = mapped_column(
        ForeignKey(
            "connect.db_conn.conn_id",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    tbl_id: Mapped[int] = mapped_column(primary_key=True)

    # Define relationships
    connections: Mapped["DBConn"] = relationship(back_populates="tables_assoc", passive_deletes="all")
    tables: Mapped["DBTables"] = relationship(back_populates="connection", passive_deletes="all")


class TokenUserAssociation(Base):
    __tablename__ = "user_tokens"
    __table_args__ = (
        ForeignKeyConstraint(
            ["client_id", "token_id"],
            ["ai_model.token_count.client_id", "ai_model.token_count.token_id"],
            ondelete="CASCADE",
        ),
        {"schema": "ai_model"},
    )

    client_id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey(
            "account.user.user_id",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    token_id: Mapped[int] = mapped_column(primary_key=True)


class DemoVectorAssociation(Base):
    __tablename__ = "demo_vector"
    __table_args__ = {"schema": "connect"}
    client_id: Mapped[int] = mapped_column(
        ForeignKey("account.client.client_id", ondelete="CASCADE"), primary_key=True
    )
    vector_id: Mapped[int] = mapped_column(
        ForeignKey("connect.vector_db.vector_id", ondelete="CASCADE"), primary_key=True
    )
    demo_db_uuid: Mapped[uuid.UUID]
    demo_client_id: Mapped[int]
    demo_client_uuid: Mapped[uuid.UUID]


class ClientSecretAssociation(Base):
    __tablename__ = "client_secret_assoc"
    __table_args__ = {"schema": "account"}

    client_secret_uuid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), server_default=text("gen_random_uuid()"))
    client_id: Mapped[int] = mapped_column(ForeignKey("account.client.client_id", ondelete="CASCADE"))
    hashed_client_secret: Mapped[str] = mapped_column(primary_key=True)
    role: Mapped[enums.AllUserRoles]
    public_key: Mapped[Optional[str]]  # Optional since not all old accounts were migrated over
    description: Mapped[Optional[str]]  # Optional since not all old accounts were migrated over


class VisualResult(BaseModel):
    visual_result: VisualResultHistory
    result_uuid: uuid.UUID
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ChatHistoryCombined(BaseModel):
    chat_hist: ChatHistory
    visual_result_hist: Optional[VisualResultHistory] = None
    result_hist: Optional[ResultHistory] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class SQLTableFromDB(sch.SQLTable):
    db_table: DBTables
    model_config = ConfigDict(arbitrary_types_allowed=True)
