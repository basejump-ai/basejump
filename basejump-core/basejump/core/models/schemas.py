"""Models for endpoints"""

import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Callable, Dict, Literal, Optional, Union

import pandas as pd
import sqlalchemy as sa
from basejump.core.common.config.logconfig import set_logging
from basejump.core.models import constants, enums
from llama_index.core.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
    TokenCountingHandler,
)
from llama_index.core.llms import MessageRole
from llama_index.core.objects import SQLTableSchema
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.ext.asyncio import AsyncSession

logger = set_logging(handler_option="stream", name=__name__)


class VectorVendor(BaseModel):
    vector_database_vendor: enums.VectorVendorType
    vector_datasource_type: enums.VectorSourceType
    index_name: str
    api_key: Optional[str] = None
    environment: Optional[str] = None


class DBSchema(BaseModel):
    """Used to provide values for jinjafied schemas"""

    schema_nm: str = Field(
        description="The schema name. Can include jinja values. Example would be geo{{region}}{{country}}",
        examples=["geo{{region}}{{county}}"],
    )
    jinja_values: Optional[dict[str, str]] = Field(
        default=None,
        description="""\
If the schema has any jinja included to make the schema name dynamic, you would provide values
to resolve the jinja. If the schema is my_schema{{id}} then the jinja values would look like {'id': 1234} \
in order to resolve the id jinja value.""",
        examples=[{"region": "_nw", "country": "_us"}],
    )
    schema_nm_rendered: Optional[str] = Field(
        default=None,
        description="This field can be ignored when submitting data to the API since the API will render the schemas",
    )


class SchemaMap(BaseModel):
    old_schema: str
    new_schema: str


class DBParamsSchemaBase(BaseModel):
    """Parameters to connect to a database"""

    database_type: enums.DatabaseType = Field(examples=["postgres"], description="The name of the SQL DBMS.")
    host: str = Field(examples=["myhost.com"], description="The database host.")
    # TODO: Should this be optional?
    port: Optional[int] = Field(None, examples=[5432], description="The database port.")
    database_name: str = Field(examples=["Example DB"], description="A descriptive database name.")
    database_name_alias: Optional[str] = Field(
        default=None,
        description=constants.DB_ALIAS_NAME_DESC,
    )
    database_name_alias_number: Optional[int] = Field(
        default=0, description="Ignore this field. Used for numbering the alias name."
    )
    ssl_mode: enums.SSLModes = Field(default=enums.SSLModes.REQUIRE)
    ssl_root_cert: Optional[str] = Field(
        default=None,
        description="""The PEM formatted SSL/TSL certificate. If the SSL mode is greater than require, \
then an ssl root certificate is required.""",
    )
    ssl: bool = Field(default=True, description="Whether to connect using SSL")
    # BC v0.28.0
    query: Optional[dict] = Field(None, description="An optional field for passing SSL parameters.")
    # query: Optional[dict] = Field(example="{}", description="An optional field for passing SSL parameters.")
    database_desc: str = Field(
        examples=["My example database used for demos"],
        description="A description of the database. Useful in retrieval for the AI.",
    )
    include_default_schema: bool = Field(
        default=True,
        examples=[True],
        description="This flag is to indicate if you would like all tables "
        "without a schema included as available to the AI and end user",
    )
    schemas: list[DBSchema] = Field(
        default_factory=list,
        examples=[["public", "archive"]],
        description="A list of the schemas you want the AI to have access to.",
    )
    schema_maps: list[SchemaMap] = Field(
        default_factory=list,
        description="Update prior schemas to a new schema. This is primarily useful for jinja schemas to \
            preserve table definitions.",
    )
    table_filter_string: Optional[str] = Field(
        default=None,
        description="Tables containing this string will be excluded from the database",
    )
    include_views: Optional[bool] = Field(
        default=True,
        description="Whether to include views in the tables users are able to query",
    )  # BC v0.27.0: Switch include views to False as soon as I get to the next release
    include_materialized_views: Optional[bool] = Field(
        default=False,
        description="Whether to include materialized views in the tables users are able to query",
    )
    include_partitioned_tables: Optional[bool] = Field(
        default=False,
        description="Whether to include source tables that are partitioned in the tables users are able to query. \
This is not partitioned tables, but rather the parent table that is being partitioned.",
    )
    model_config = ConfigDict(from_attributes=True)


class DBParamsSchema(DBParamsSchemaBase):
    drivername: Union[enums.DBDriverName, enums.DBAsyncDriverName] = Field(
        examples=["postgresql"], description="The SQLAlchemy drivername."
    )


class DBParamsBytes(BaseModel):
    """Parameters to connect to a database as encrypted bytes"""

    database_type: bytes
    drivername: bytes
    host: bytes
    port: Optional[bytes] = None
    database_name: bytes
    database_name_alias: bytes
    query: bytes
    database_desc: bytes
    schemas: bytes
    include_default_schema: bool
    table_filter_string: Optional[str] = None
    include_views: bool
    include_materialized_views: bool
    include_partitioned_tables: bool
    ssl_mode: enums.SSLModes
    ssl_root_cert: Optional[str] = None
    ssl: bool = True
    model_config = ConfigDict(from_attributes=True)


class DBConnSchema(BaseModel):
    username: str = Field(description="The user name credential.")
    password: str = Field(description="The password credential.")
    data_source_desc: str = Field(description="A description of the connection.")


class SQLDBSchema(DBParamsSchema, DBConnSchema):
    pass


class SQLConnSchema(BaseModel):
    conn_params: SQLDBSchema
    conn_id: int
    conn_uuid: uuid.UUID
    db_id: int
    vector_id: int
    db_uuid: uuid.UUID


class SQLDBBytesSchema(DBParamsBytes):
    username: bytes
    password: bytes
    model_config = ConfigDict(from_attributes=True)


class GetSQLConn(BaseModel):
    conn_uuid: uuid.UUID = Field(examples=[str(uuid.uuid4())])
    db_uuid: uuid.UUID = Field(examples=[str(uuid.uuid4())])
    database_name_alias: Optional[str] = Field(
        default=None,
        description=constants.DB_ALIAS_NAME_DESC,
    )


class SQLConn(GetSQLConn):
    db_id: int
    conn_id: int


class DBColumn(BaseModel):
    column_name: str
    table_name: str
    schema_name: Optional[str] = None
    filters: list = Field(
        default_factory=list,
        description="A list of values used to filter the column if the column was used in a WHERE clause. \
It will only be more than one value if an IN was used.",
    )
    column_w_func: Optional[str] = Field(
        default=None, description="The raw column with surrounding functions as a string"
    )
    cast_type: Optional[str] = Field(
        default=None,
        description="If the column is being cast, this provides \
the value it is being cast to.",
    )
    column_type: Optional[str] = None
    quoted: bool = Field(default=False, description="Whether or not the column is quoted in the DB")


class SharedTableColumns(BaseModel):
    description: Optional[str] = Field(
        default=None, examples=["This is my column"], description="A description of the column."
    )
    foreign_key_table_name: Optional[str] = Field(
        default=None, examples=["other_table"], description="The foreign key table of the column (if any)."
    )
    foreign_key_column_name: Optional[str] = Field(
        default=None, examples=["other_column"], description="The foreign key column name of the column (if any)."
    )
    primary_key: Optional[bool] = Field(default=False, description="Whether or not this column is a primary key.")
    distinct_values: Optional[list] = Field(
        default=[],
        description="""These are distinct values that exist in this column. \
Only to be used for low cardinality fields""",
        examples=["[Male, Female]"],
    )
    ignore: Optional[bool] = Field(
        default=False,
        description="""Use this field to remove this column from consideration for the AI for use in SQL queries. \
This will also remove this table as a viewable table for users when exploring the database as well.""",
    )


class SQLTableColumn(BaseModel):
    column_name: str = Field(examples=["my_column"])
    column_type: str = Field(examples=["VARCHAR"])
    # HACK: Everything below is from SharedTableColumns, but inheritance is not used to preserve ordering
    # TODO: See if there is a way inheritance can be used and this order still preserved
    description: Optional[str] = Field(default=None, examples=["This is my column"])
    foreign_key_table_name: Optional[str] = Field(default=None, examples=["other_table"])
    foreign_key_column_name: Optional[str] = Field(default=None, examples=["other_column"])
    primary_key: Optional[bool] = False
    distinct_values: Optional[list] = Field(
        default=[],
        description="""These are distinct values that exist in this column. \
Only to be used for low cardinality fields""",
        examples=["[Male, Female]"],
    )
    ignore: Optional[bool] = Field(
        default=False,
        description="""Use this field to remove this column from consideration for the AI for use in SQL queries. \
This will also remove this table as a viewable table for users when exploring the database as well.""",
    )
    model_config = ConfigDict(from_attributes=True)
    quoted: bool = Field(default=False, description="Whether or not the column is quoted in the DB")
    new: bool = Field(default=False, description="Whether or not the column is new", exclude=True)


class UpdateTableColumns(SharedTableColumns):
    col_uuid: uuid.UUID = Field(description=constants.COL_UUID_DSC)


class GetSQLTableColumn(SQLTableColumn):
    col_uuid: uuid.UUID


class SQLTableInfo(BaseModel):
    table_name: str = Field(
        examples=["table_nm1"],
        description="The full name of the table (including the schema if needed)",
    )
    context: Optional[str] = Field(default=None, examples=["This is table 1"])


# TODO: Consider renaming
class GetSQLTable(SQLTableInfo):
    tbl_uuid: uuid.UUID = Field(examples=[str(uuid.uuid4())])
    columns: list[GetSQLTableColumn]
    ignore: Optional[bool] = False
    primary_keys: Optional[list] = Field(default_factory=list)
    model_config = ConfigDict(from_attributes=True)


class DBTable(SQLTableInfo):
    tbl_id: int
    tbl_uuid: uuid.UUID = Field(examples=[str(uuid.uuid4())])
    columns: list[GetSQLTableColumn]
    model_config = ConfigDict(from_attributes=True)


class BaseUser(BaseModel):
    client_id: int
    username: str
    role: enums.UserRoles
    email_address: Optional[str] = None
    service_user_uuid: Optional[uuid.UUID] = None


class TeamFields(BaseModel):
    team_name: str = Field(examples=[constants.TEAM_NM_DESC])
    team_desc: str = Field(examples=[constants.TEAM_NM_DESC])
    model_config = ConfigDict(from_attributes=True)


class BaseTeam(TeamFields):
    client_id: int


class GetDBParamsBytes(DBParamsBytes):
    db_uuid: uuid.UUID
    model_config = ConfigDict(from_attributes=True)


class GetConn(BaseModel):
    conn_uuid: uuid.UUID = Field(examples=[str(uuid.uuid4())])
    conn_type: str = Field(examples=["SQL"])
    data_source_desc: str = Field(examples=["HR Database"])
    model_config = ConfigDict(from_attributes=True)


# TODO: Consider renaming
class GetConnMetadata(GetConn):
    db_uuid: uuid.UUID = Field(examples=[str(uuid.uuid4())])
    username: str
    schemas: Optional[list[DBSchema]] = None


# TODO: Consider renaming to GetDB
class GetDBParams(DBParamsSchema):
    db_uuid: uuid.UUID
    connections: Optional[list[GetConnMetadata]] = None


# TODO: Dup of Vector Vendor - consolidate
class VectorDBSchema(BaseModel):
    vector_database_vendor: enums.VectorVendorType
    vector_datasource_type: enums.VectorSourceType
    index_name: Optional[str] = None
    api_key: Optional[str] = None
    environment: Optional[str] = None
    model_config = ConfigDict(from_attributes=True)


class MessageQueryResult(BaseModel):
    result_uuid: Optional[uuid.UUID] = Field(
        default=None, examples=[str(uuid.uuid4())], description=constants.RESULT_UUID_DSC
    )
    sql_query: Optional[str] = Field(
        default=None, examples=["select * from account.teams"], description="The SQL query created by the AI."
    )
    result_type: Optional[enums.ResultType] = Field(
        default=None,
        examples=["metric"],
        description="""The type of result. \
Metric is a single datapoint (1 row and 1 column). A record is 1 row and multiple columns. \
A dataset is multiple rows and columns.""",
    )
    visual_result_uuid: Optional[uuid.UUID] = Field(
        default=None, examples=[str(uuid.uuid4())], description=constants.VISUAL_RESULT_UUID_DSC
    )
    visual_json: Optional[dict] = Field(default=None, description="The JSON for the plotly result.")
    visual_explanation: Optional[str] = Field(default=None, description="The AI explanation of the plotly result.")
    saved_result_uuid: Optional[uuid.UUID] = Field(
        default=None,
        examples=[str(uuid.uuid4())],
        description="Unique universal identifier to saved results. This is null if a result is not saved.",
    )
    model_config = ConfigDict(from_attributes=True)


class BaseMessage(BaseModel):
    role: MessageRole = Field(
        examples=["assistant"], description="The purpose or creator of the message, such as the user or AI."
    )
    msg_type: enums.MessageType = Field(
        default=enums.MessageType.RESPONSE,
        examples=["response"],
        description="The type of message. Used to control webhook behavior for AI thoughts.",
    )
    content: str = Field(examples=["There are 6 teams"], description="The content of the message.")
    timestamp: datetime = Field(examples=["2023-04-20T10:30:00"], description="The timestamp of the message.")
    msg_uuid: uuid.UUID = Field(
        default_factory=uuid.uuid4, examples=[str(uuid.uuid4())], description=constants.MSG_UUID_DSC
    )


class Message(BaseMessage):
    query_result: Optional[MessageQueryResult] = None


class QueryResultRows(BaseModel):
    query_result: list[sa.Row]
    model_config = ConfigDict(arbitrary_types_allowed=True)


class QueryResultBase(BaseModel):
    preview_row_ct: int
    num_rows: int
    num_cols: int
    result_type: enums.ResultType
    sql_query: str


def get_tmrw() -> datetime:
    tomorrow_date = datetime.now() + timedelta(days=1)
    return datetime(tomorrow_date.year, tomorrow_date.month, tomorrow_date.day)


class QueryResult(QueryResultBase, QueryResultRows):
    result_uuid: uuid.UUID
    ai_preview_row_ct: int
    result_file_path: str
    preview_file_path: str
    result_type: enums.ResultType
    result_exp_time: datetime = Field(default_factory=get_tmrw)
    metric_value: Optional[str] = None
    metric_value_formatted: Optional[str] = None
    aborted_upload: bool = False


class QueryResultDF(QueryResultBase, QueryResultRows):
    output_df: pd.DataFrame
    preview_output_df: pd.DataFrame


class APIMessage(BaseMessage, MessageQueryResult):
    """
    Standard message format for the API messages.
    """

    # PromptMetadata Attrs
    # TODO: Make this PromptMetadata into a sub-schema
    prompt_uuid: uuid.UUID = Field(examples=[str(uuid.uuid4())], description=constants.PROMPT_UUID_DSC)
    initial_prompt: str = Field(
        examples=["How many teams are there?"], description="The initial user prompt to the AI."
    )
    prompt_time: datetime = Field(examples=["2023-04-20T10:30:00"], description="The time of the prompt.")
    parent_msg_uuid: uuid.UUID = Field(examples=[str(uuid.uuid4())], description=constants.PARENT_MSG_UUID_DSC)
    verified: bool = False
    verified_user_role: Optional[str] = None
    verified_user_uuid: Optional[uuid.UUID] = None
    can_verify: Optional[bool] = None
    thumbs_up: Optional[bool] = Field(
        default=None,
        description=constants.THUMBS_UP_DESCR,
    )
    model_config = ConfigDict(from_attributes=True)


class ClientUserInfo(BaseModel):
    client_id: int
    client_uuid: uuid.UUID
    user_id: int
    user_uuid: uuid.UUID
    user_role: str  # UserRoles StrEnum


class PromptMetadataBase(ClientUserInfo):
    """This is to be used when there is no user chatting back and forth with the agent.
    This is purely for an agent iterating by itself and no human in the loop.
    """

    initial_prompt: str
    prompt_uuid: uuid.UUID
    prompt_id: int
    llm_type: enums.LLMType
    prompt_time: datetime
    return_visual_json: bool = False


class PromptMetadata(PromptMetadataBase):
    llama_debug: LlamaDebugHandler
    token_counter: TokenCountingHandler
    callback_manager: CallbackManager
    model_config = ConfigDict(arbitrary_types_allowed=True)


# NOTE: UUIDs need to be str since they are dumped into Redis
class SemCacheMetadata(BaseModel):
    result_uuid: str
    prompt_uuid: str
    verified_user_uuid: str
    sql_query: str
    timestamp: str
    verified_user_role: str
    conn_uuid: str


class SemCache(SemCacheMetadata):
    prompt: str
    response: str


class SemCacheResponse(SemCache):
    vector_dist: float
    can_verify: bool
    verified: bool


class ThoughtMessage(BaseModel):
    timestamp: datetime
    thought: str


class BaseModelInfo(BaseModel):
    model_name: enums.AIModelSchema
    max_tokens: int = Field(
        default=500,
        description="""Limit max_tokens to reduce hitting API limit since it's \
estimated based off of max_tokens instead of the actual tokens in the completion""",
    )


class AIEndpointInfo(BaseModel):
    endpoint: str


class AzureEndpointInfo(AIEndpointInfo):
    api_key: str
    deployment_name: str


class AWSEndpointInfo(AIEndpointInfo):
    access_key: str
    secret_access_key: str
    deployment_region: str


class ModelInfo(BaseModelInfo):
    endpoint_info: Optional[AIEndpointInfo] = None
    deployment_callback: Optional[Callable] = None
    deployment_callback_kwargs: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_deployment_config(self):
        if self.endpoint_info is None and self.deployment_callback is None:
            raise ValueError("At least one of 'endpoint_info' or 'deployment_callback' must be provided")
        elif self.deployment_callback is not None:
            if not self.deployment_callback_kwargs:
                raise ValueError("Deployment kwargs must be provided if using deployment callback")
        if self.endpoint_info is not None and self.deployment_callback is not None:
            logger.warning(
                "Endpoint info and deployment callback both have been defined. Deferring to endpoint info instead of \
deployment callback."
            )
        if not self.endpoint_info:
            assert self.deployment_callback_kwargs, "Need deployment callback kwargs if endpoint info is not defined."
            assert self.deployment_callback, "Need deployment callback if endpoint info is not defined."
            self.endpoint_info = self.deployment_callback(**self.deployment_callback_kwargs)
        return self


class AzureModelInfo(ModelInfo):
    endpoint_info: Optional[AzureEndpointInfo] = None
    api_version: str


class AWSModelInfo(ModelInfo):
    endpoint_info: Optional[AWSEndpointInfo] = None


class ChatMetadata(BaseModel):
    """The parts of the prompt that are immutable + thread-safe that can easily be passed to a background task"""

    chat_id: int
    chat_uuid: uuid.UUID
    vector_id: int
    index_name: str
    team_id: int
    team_uuid: uuid.UUID
    curr_chat_history: list[APIMessage] = Field(default_factory=list)
    curr_thought_history: list[ThoughtMessage] = Field(default_factory=list)
    parent_msg_uuid: uuid.UUID
    reset_parent_msg_uuid: bool = False
    send_message: bool = False
    webhook_url: Optional[str] = None
    webhook_headers: Optional[dict] = None
    return_sql_in_thoughts: bool = False
    chat_in_index: bool = False
    semcache_response: Optional[SemCacheResponse] = None
    verify_mode: enums.VerifyMode = enums.VerifyMode.EXPLORE
    vector_store: BasePydanticVectorStore
    embedding_model_info: AzureModelInfo


class TokenCountSchema(BaseModel):
    prompt_id: int
    prompt: str
    client_id: int
    ai_model_provider: str
    ai_model_nm: str
    cost_per_1k_tokens_input: Decimal
    cost_per_1k_tokens_output: Decimal
    total_embedding_token_count: int
    prompt_llm_token_count: int
    completion_llm_token_count: int
    total_llm_token_count: int


class UpdateVector(BaseModel):
    vector_metadata: dict
    timestamp: datetime


class CreateDBConn(DBConnSchema):
    schemas: Optional[list[DBSchema]] = None


class CallbackMgrs(BaseModel):
    token_counter: TokenCountingHandler
    callback_manager: CallbackManager
    llama_debug: LlamaDebugHandler
    model_config = ConfigDict(arbitrary_types_allowed=True)


# NOTE: WARNING - Since this is part of models, if these are made lower case it will break
# the tables without an alembic script to make the update


class BaseClient(BaseModel):
    client_name: str = Field(examples=["ABC Company"])
    description: Optional[str] = None


class CreateClient(BaseClient):
    hashed_client_secret: str
    client_type: enums.ClientType


class NewClientBase(CreateClient):
    client_id: int
    client_uuid: uuid.UUID


class NewClient(NewClientBase):
    client_secret_uuid: uuid.UUID


class SQLTimeoutError(TimeoutError):
    pass


class Alias(BaseModel):
    alias_name: str
    alias_number: int


class SendSolution(BaseModel):
    db: AsyncSession
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ChartType(BaseModel):
    chart_type: Literal["pie", "bar", "line", "area", "scatter"]


class TrustScore(BaseModel):
    total_messages: int
    thumbs_down_count: int
    trust_score: float


class SQLTable(SQLTableSchema):
    """Lightweight representation of a SQL table"""

    columns: list[SQLTableColumn] = []
    tbl_uuid: Optional[uuid.UUID] = None  # type:ignore
    ignore: Optional[bool] = False
    primary_keys: Optional[list] = None
    new: bool = Field(default=False, description="Whether or not the table is a new table.", exclude=True)


class IndexedTables(BaseModel):
    index_name: str
    vector_uuid: uuid.UUID
    tables: list[SQLTable]
    number_of_days: Optional[int] = None


class UploadResult(BaseModel):
    result_uuid: uuid.UUID
    s3_file_key: str
