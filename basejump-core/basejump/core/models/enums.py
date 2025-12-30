from decimal import Decimal
from enum import Enum, StrEnum

from sqlglot.dialects.dialect import Dialects


class ConnectionType(StrEnum):
    """Connection type"""

    SQL = "sql_db"
    VECTOR = "vector_db"


class ClientType(StrEnum):
    DEMO = "DEMO"
    TEST = "TEST"
    CLIENT = "CLIENT"
    # An account not to be deleted that is used internally e.g. for copying demo acct information
    INTERNAL = "INTERNAL"


class VectorVendorType(StrEnum):
    REDIS = "REDIS"


class VectorSourceType(StrEnum):
    TABLE = "TABLE"
    CHAT = "CHAT"


class DatabaseType(StrEnum):
    """Database type"""

    POSTGRES = "postgres"
    MYSQL = "mysql"
    REDSHIFT = "redshift"
    SQL_SERVER = "sql_server"
    SNOWFLAKE = "snowflake"
    ATHENA = "athena"
    ORACLE = "oracle"  # unsupported currently adding for uppercase default dbs
    DB2 = "db2"  # unsupported currently adding for uppercase default dbs


UPPERCASE_DEFAULT_DB = [DatabaseType.SNOWFLAKE.value, DatabaseType.ORACLE.value, DatabaseType.DB2.value]


# Taken from: https://github.com/tobymao/sqlglot/blob/main/sqlglot/dialects/__init__.py#L97
DB_TYPE_TO_SQLGLOT_DIALECT_LKUP = {
    DatabaseType.ATHENA: Dialects.ATHENA.value,
    DatabaseType.POSTGRES: Dialects.POSTGRES.value,
    DatabaseType.MYSQL: Dialects.MYSQL.value,
    DatabaseType.REDSHIFT: Dialects.REDSHIFT.value,
    DatabaseType.SQL_SERVER: Dialects.TSQL.value,
    DatabaseType.SNOWFLAKE: Dialects.SNOWFLAKE.value,
}


class DBDriverName(StrEnum):
    """Database driver type"""

    ATHENA = "awsathena+rest"
    POSTGRES = "postgresql"
    MYSQL = "mysql+pymysql"
    REDSHIFT = "postgresql+psycopg2"
    SQL_SERVER = "mssql+pyodbc"
    SNOWFLAKE = "snowflake"


DRIVER_LKUP = {
    DatabaseType.ATHENA: DBDriverName.ATHENA,
    DatabaseType.POSTGRES: DBDriverName.POSTGRES,
    DatabaseType.MYSQL: DBDriverName.MYSQL,
    DatabaseType.REDSHIFT: DBDriverName.REDSHIFT,
    DatabaseType.SQL_SERVER: DBDriverName.SQL_SERVER,
    DatabaseType.SNOWFLAKE: DBDriverName.SNOWFLAKE,
}


class DBAsyncDriverName(Enum):
    """Async database driver type"""

    POSTGRES = "postgresql+asyncpg"
    MYSQL = "mysql+asyncmy"
    REDSHIFT = None  # "Redshift has no async driver and is not compatible with SQLAlchemy 2"


class SSLModes(StrEnum):
    REQUIRE = "require"
    VERIFY_CA = "verify-ca"
    VERIFY_FULL = "verify-full"


class UserRoles(StrEnum):
    OWNER = "OWNER"
    ADMIN = "ADMIN"
    MEMBER = "MEMBER"
    SERVICE_MEMBER = "SERVICE_MEMBER"
    SUB_MEMBER = "SUB_MEMBER"


class APIUserRoles(StrEnum):
    INTERNAL = "INTERNAL"  # For internal use only, not for clients


class AllUserRoles(StrEnum):
    OWNER = UserRoles.OWNER.value
    ADMIN = UserRoles.ADMIN.value
    MEMBER = UserRoles.MEMBER.value
    SERVICE_MEMBER = UserRoles.SERVICE_MEMBER.value
    SUB_MEMBER = UserRoles.SUB_MEMBER.value
    INTERNAL = APIUserRoles.INTERNAL.value


USER_ROLES_LVL_LKUP = {
    AllUserRoles.INTERNAL.value: 4,
    AllUserRoles.OWNER.value: 3,
    AllUserRoles.ADMIN.value: 2,
    AllUserRoles.MEMBER.value: 1,
    AllUserRoles.SERVICE_MEMBER.value: 1,
    AllUserRoles.SUB_MEMBER.value: 1,
}


class VerifyMode(StrEnum):
    STRICT = "STRICT"
    EXPLORE = "EXPLORE"


class MessageType(StrEnum):
    """Webhook response type"""

    # The RESPONSE and ERROR types will appear in the chat message the user sees
    RESPONSE = "response"
    ERROR = "error"
    # The THOUGHT type will appear in a temporary thought bubble that the user sees
    THOUGHT = "thought"
    # Used to send back status updates (will not be returned to the user - for developer use only)
    STATUS = "status"
    # This should only have a blank response and is only to indicate completion of a thought block
    SOLUTION = "solution"
    INIT = "initial"


class ResultType(StrEnum):
    METRIC = "metric"
    RECORD = "record"
    DATASET = "dataset"


class LLMType(StrEnum):
    MERMAID_AGENT = "MERMAID_AGENT"
    DATA_AGENT = "DATA_AGENT"
    SIMPLE_AGENT = "SIMPLE_AGENT"


class AIModelSchema(StrEnum):
    GROQ = "llama3-70b-8192"
    GPT4o = "gpt-4o"
    GPT41 = "gpt-41"
    GPT4 = "gpt-4"
    GPT4oMINI = "gpt-4o-mini"
    GPT35_TURBO_AZURE = "gpt-3.5-turbo"
    GPT35_TURBO_OPENAI = "gpt-3.5-turbo-0613"
    ADA = "text-embedding-ada-002"
    ADA3_LARGE = "text-embedding-3-large"
    ADA3_SMALL = "text-embedding-3-small"
    SONNET35 = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    SONNET37 = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"


class AIModelSchemaClientOptions(StrEnum):
    GPT4o = AIModelSchema.GPT4o.value
    GPT41 = AIModelSchema.GPT41.value
    SONNET35 = AIModelSchema.SONNET35.value
    SONNET37 = AIModelSchema.SONNET37.value


class AIModelProvider(StrEnum):
    AZURE_OPENAI = "Azure OpenAI"
    GROQ = "Groq"
    AWS = "AWS Bedrock"


class AzurePricingQueries(StrEnum):
    GPT4o_input = (
        """productName eq 'Azure OpenAI' and location eq 'US East' and skuName eq 'gpt-4o-Input-regional'"""  # noqa
    )
    GPT4o_output = (
        """productName eq 'Azure OpenAI' and location eq 'US East' and skuName eq 'gpt-4o-Output-regional'"""  # noqa
    )
    ADA = """productName eq 'Azure OpenAI' and location eq 'US East' and skuName eq 'embedding-ada-regional'"""


# TODO: Use an API to automatically get the prices
# Prices here: https://wow.groq.com/
class DefaultTokenPrices(Enum):
    GROQ_70B_llama_input = Decimal(0.59 / 1000)
    GROQ_70B_llama_output = Decimal(0.79 / 1000)
    GPT4o_input = Decimal(0.005)
    GPT4o_output = Decimal(0.015)
    GPT41_input = Decimal(0.002)
    GPT41_output = Decimal(0.008)
    ADA = Decimal(0.0001)
    ADA3_SMALL = Decimal(0.000022)


class AIModelType(StrEnum):
    LLM = "llm"
    EMBEDDING = "embedding"


class RedisHashKeys(StrEnum):
    DB_REINDEX_STATUS_KEY = "db_reindex_status"  # Key to use for when DB is currently indexing
    DB_INDEX_STATUS_KEY = "db_index_status"  # Key to use for when DB is currently indexing
    DB_INDEX_UPDATE_STATUS_KEY = "Updating DB index"
    CORS_ALLOWED_DOMAINS = "CORS Allowed Domains"


class RedisValues(StrEnum):
    RUNNING_DB_INDEX = "Running DB index"
    QUEUED_INDEX = "Index in queue"
    ERROR_RUNNING_DB_INDEX = "Error running DB index"
    NO_TABLES_ERR = "No tables found for the provided database"
    NO_PERMITTED_TABLES_ERR = "No permitted tables to select found for the provided database"
    ERROR_UPDATING_DB_INDEX = "Error while updating the redis index"
    RUNNING_DB_REINDEX = "Running DB reindex"
    ERROR_REINDEXING_DB = "Error while reindexing the DB"


class SubPlanType(StrEnum):
    STARTER = "starter"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    DEMO = "demo"
    FREE = "free"
    DEVELOPER = "developer"


class TransactionType(StrEnum):
    CREDITS_PURCHASE = "credits_purchase"
    SUBSCRIPTION = "subscription_renewal"


class SQLSimilarityLabel(StrEnum):
    IDENTICAL = "identical"
    EQUIVALENT = "equivalent"
    SIMILAR = "similar"
    DIFFERENT = "different"

    def __init__(self, value):
        self._value_ = value
        self.description = ""


SQLSimilarityLabel.IDENTICAL.description = "There are no differences"
SQLSimilarityLabel.EQUIVALENT.description = (
    "The output and tables are the same, but queries are structured differently"
)
SQLSimilarityLabel.SIMILAR.description = "The queries share some or all tables, but the output is different"
SQLSimilarityLabel.DIFFERENT.description = "The queries share no tables"


class ConnType(StrEnum):
    DATABASE = "database"
    CONNECTION = "connection"


class TokenType(StrEnum):
    ACCESS_TOKEN_TYPE = "access_token"
    REFRESH_TOKEN_TYPE = "refresh_token"


class SortByOptions(StrEnum):
    """Result sort by options"""

    TITLE = "title"
    DESCRIPTION = "description"
    RESULT_TYPE = "result_type"


class HTTPMethod(StrEnum):
    """Supported HTTP methods"""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    OPTIONS = "OPTIONS"
