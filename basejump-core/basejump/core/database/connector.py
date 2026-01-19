import copy
import json
import os
import ssl
import uuid
from abc import ABC, abstractmethod
from typing import ClassVar, Optional, Union

import boto3
import psycopg2
import sqlalchemy as sa
from cryptography.fernet import Fernet
from sqlalchemy import URL, create_engine, text
from sqlalchemy.engine import Engine, Row
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool

from basejump.core.common.config.logconfig import set_logging
from basejump.core.common.config.settings import get_encryption_key
from basejump.core.database.inspector import (
    athena,
    base,
    mysql,
    postgres,
    redshift,
    snowflake,
    sql_server,
)
from basejump.core.database.ssl import (
    AthenaSSLParams,
    MSSQLSSLParams,
    MySQLSSLParams,
    PostgresSSLParams,
    SnowflakeSSLParams,
    SSLEngine,
)
from basejump.core.models import constants, enums, errors
from basejump.core.models import schemas as sch
from basejump.core.models.models import DBConn, DBParams

logger = set_logging(handler_option="stream", name=__name__)


# TODO: Get these set possibly as property / class attributes instead
POOL_SIZE = 4
MAX_OVERFLOW = 4  # Number of connections that can be opened beyond the pool_size
POOL_RECYCLE = 3600  # Recycle connections after 1 hour
POOL_TIMEOUT = 60 * 3  # Raise an exception after 3 minutes if no connection is available from the pool


class Connector(ABC):
    database_type: ClassVar[enums.DatabaseType]  # Each subclass must define this

    def __init__(self, conn_params: sch.SQLDBSchema, echo: bool = False):
        self.conn_params = conn_params
        self.echo = echo
        # TODO: Make this more robust
        try:
            assert conn_params.database_type == self.database_type
        except AssertionError:
            raise TypeError("Connector class used does not match database type passed in conn_params")
        except AttributeError:
            raise AttributeError("Connector subclasses must have a database_type defined as a class attribute.")

    @abstractmethod
    def get_ssl_args(self) -> tuple:
        pass

    @abstractmethod
    def get_conn_uri(self, hide_password: bool = False) -> str:
        pass

    @abstractmethod
    def create_ssl_engine(self, engine: Engine, ssl_cert_path) -> SSLEngine:
        pass

    @abstractmethod
    def get_inspector(self, conn: sa.Connection) -> base.BaseInspector:
        pass

    @property
    def conn_params_bytes(self) -> sch.DBParamsBytes:
        conn_params_dict = self.conn_params.dict()
        # Copy all of the non encrypted values
        include_default_schema = copy.copy(conn_params_dict["include_default_schema"])
        table_filter_string = copy.copy(conn_params_dict["table_filter_string"])
        include_views = copy.copy(conn_params_dict["include_views"])
        include_materialized_views = copy.copy(conn_params_dict["include_materialized_views"])
        include_partitioned_tables = copy.copy(conn_params_dict["include_partitioned_tables"])
        ssl_mode = copy.copy(conn_params_dict["ssl_mode"])
        ssl_root_cert = copy.copy(conn_params_dict["ssl_root_cert"])
        ssl = copy.copy(conn_params_dict["ssl"])
        # Remove from dict so it is not encrypted
        del conn_params_dict["include_default_schema"]
        del conn_params_dict["table_filter_string"]
        del conn_params_dict["include_views"]
        del conn_params_dict["include_materialized_views"]
        del conn_params_dict["include_partitioned_tables"]
        del conn_params_dict["ssl_mode"]
        del conn_params_dict["ssl_root_cert"]
        del conn_params_dict["ssl"]
        # Update other variables fields
        conn_params_dict.pop("database_name_alias_number", None)
        conn_params_dict["database_type"] = conn_params_dict["database_type"].value
        conn_params_dict["drivername"] = conn_params_dict["drivername"].value
        # Encrypt the fields
        db_params = self.encrypt_db(dict_to_encrypt=conn_params_dict)
        return sch.DBParamsBytes(
            **db_params,
            include_default_schema=include_default_schema,
            table_filter_string=table_filter_string,
            include_views=include_views,
            include_materialized_views=include_materialized_views,
            include_partitioned_tables=include_partitioned_tables,
            ssl_mode=ssl_mode,
            ssl_root_cert=ssl_root_cert,
            ssl=ssl,
        )

    @staticmethod
    def get_database_to_connect(conn_params: sch.SQLDBSchema) -> "Connector":
        if conn_params.database_type == enums.DatabaseType.ATHENA:
            return AthenaConnector(conn_params=conn_params)
        elif conn_params.database_type == enums.DatabaseType.POSTGRES:
            return PostgresConnector(conn_params=conn_params)
        elif conn_params.database_type == enums.DatabaseType.MYSQL:
            return MySQLConnector(conn_params=conn_params)
        elif conn_params.database_type == enums.DatabaseType.REDSHIFT:
            return RedshiftConnector(conn_params=conn_params)
        elif conn_params.database_type == enums.DatabaseType.SQL_SERVER:
            return SQLServerConnector(conn_params=conn_params)
        elif conn_params.database_type == enums.DatabaseType.SNOWFLAKE:
            return SnowflakeConnector(conn_params=conn_params)
        else:
            raise NotImplementedError("Database type not implemented.")

    @staticmethod
    def decrypt_db(dict_to_decrypt: dict) -> dict:
        # Decrypt the sensitive information
        try:
            encryption_key = get_encryption_key()
            f = Fernet(encryption_key)
        except KeyError:
            raise errors.MissingEnvironmentVariable("Missing the ENCRYPTION_KEY environment variable.")
        conn_params = {}
        for key, value in dict_to_decrypt.items():
            if key in [
                "include_default_schema",
                "table_filter_string",
                "include_views",
                "include_materialized_views",
                "include_partitioned_tables",
                "ssl_mode",
                "ssl_root_cert",
                "ssl",
            ]:
                # Not encrypted so just keep as is
                conn_params[key] = value
                continue
            # Get the bytes value
            bytes_value = f.decrypt(value) if value else None
            # Convert from bytes to string
            # TODO: Use an StrEnum or something more robust than this
            if key in ["query", "schemas"]:
                assert bytes_value, "Value needs to not be None"
                json_value = bytes_value.decode("UTF-8")
                new_value = json.loads(json_value)
            elif key == "port":
                new_value = int.from_bytes(bytes_value, byteorder="big") if bytes_value else None
            else:
                assert bytes_value, "Value should not be None"
                new_value = bytes_value.decode("UTF-8")

            conn_params[key] = new_value

        return conn_params

    @staticmethod
    def encrypt_db(dict_to_encrypt: dict) -> dict:
        # Encrypt the sensitive information
        try:
            encryption_key = get_encryption_key()
            f = Fernet(encryption_key)
        except KeyError:
            raise errors.MissingEnvironmentVariable("Missing the ENCRYPTION_KEY environment variable.")
        conn_params_byte = {}
        for key, value in dict_to_encrypt.items():
            # Convert to binary
            if key in [
                "include_default_schema",
                "table_filter_string",
                "include_views",
                "include_materialized_views",
                "include_partitioned_tables",
                "ssl_mode",
                "ssl_root_cert",
                "ssl",
                "schema_maps",
            ]:
                continue
            if key in ["query", "schemas"]:
                json_value = json.dumps(value)
                value = json_value.encode("UTF-8")
            elif key == "port":
                value = value.to_bytes(2, byteorder="big") if value else None
            else:
                value = value.encode("UTF-8")
            # Encrypt and add to dictionary
            if value:
                conn_params_byte[key] = f.encrypt(value)
            else:
                conn_params_byte[key] = None  # type: ignore

        return conn_params_byte

    # TODO: Maybe change to 'from_db_conn' to be more in line with typical naming conventions
    @classmethod
    async def get_db_conn(cls, db_conn: Union[DBConn, Row], db_params: DBParams):
        db_params_bytes = sch.DBParamsBytes.from_orm(db_params)
        conn_params_byte = sch.SQLDBBytesSchema(
            **db_params_bytes.dict(),
            username=db_conn.username,
            password=db_conn.password,
        )
        conn_params = cls.decrypt_db(conn_params_byte.dict())
        # HACK
        if db_conn.schemas:
            schemas = db_conn.schemas
            if isinstance(schemas, str):
                schemas = json.loads(schemas)
            conn_params["schemas"] = [
                schema.dict() if isinstance(schemas, sch.DBSchema) else schema for schema in schemas  # type:ignore
            ]
        else:
            conn_params["schemas"] = []
        # BC v0.27.0 TODO: Need to fix all old schemas saved in the improper format using alembic
        # HACK
        conn_params["schemas"] = [
            sch.DBSchema(schema_nm=schema).dict() if not isinstance(schema, dict) else schema
            for schema in conn_params["schemas"]
        ]
        for schema in conn_params["schemas"]:
            if not isinstance(schema, dict):
                raise TypeError("Schemas should be a dictionary at this point")
        try:
            conn_params_obj = sch.SQLDBSchema(**conn_params, data_source_desc=db_conn.data_source_desc)
        except Exception as e:
            logger.error("Here are the params")
            logger.error("Here is the query: %s", conn_params["query"])
            logger.error("Here is the schema: %s", conn_params["schemas"])
            logger.error("Error in get_db_conn %s", str({**conn_params}))
            logger.warning(str(e))
            raise e
        return cls.get_database_to_connect(conn_params=conn_params_obj)

    @classmethod
    async def get_db_conn_from_schema(cls, db_params: DBParams, db_conn_schema: sch.DBConnSchema):
        db_params_bytes = sch.DBParamsBytes.from_orm(db_params)
        conn_params = cls.decrypt_db(db_params_bytes.dict())
        # BC v0.27.1 Added since schemas used to allow None - need to update all prior schemas in DB to empty
        # array to remove this
        if not conn_params["schemas"]:
            conn_params["schemas"] = []
        conn_params_obj = sch.SQLDBSchema(
            **conn_params,
            username=db_conn_schema.username,
            password=db_conn_schema.password,
            data_source_desc=db_conn_schema.data_source_desc,
        )
        return cls.get_database_to_connect(conn_params=conn_params_obj)

    def _create_async_connection_uri(self) -> str:
        """Create a database URI"""
        uri = URL.create(
            drivername=self.conn_params.drivername.value,
            username=self.conn_params.username,
            password=self.conn_params.password,
            host=self.conn_params.host,
            port=self.conn_params.port,
            database=self.conn_params.database_name,
        )
        return uri.render_as_string(hide_password=False)

    def _create_ssl_engine(self, engine: Engine, ssl_cert_path) -> SSLEngine:
        return SSLEngine(original_engine=engine, ssl_cert_path=ssl_cert_path)

    def _get_conn_uri(self, hide_password: bool = False) -> str:
        """Create a database URI"""
        uri_obj = URL.create(
            drivername=self.conn_params.drivername.value,
            username=self.conn_params.username,
            password=self.conn_params.password,
            host=self.conn_params.host,
            port=self.conn_params.port,
            database=self.conn_params.database_name,
            query=self.conn_params.query,  # type: ignore
        )
        return uri_obj.render_as_string(hide_password=hide_password)

    def connect_async_db(self) -> AsyncEngine:
        """Connect to a database
        WARNING: Be sure to dispose after connecting since that is not explicitly called here
        """
        uri = self._create_async_connection_uri()
        # SSL mode always on by default
        # Not doing verify-full since it cause latency overhead + we are using a VPC
        my_ssl_ctx = ssl.create_default_context()
        my_ssl_ctx.check_hostname = False
        my_ssl_ctx.verify_mode = ssl.CERT_NONE
        ssl_args = {}
        if self.conn_params.ssl:
            ssl_args = {"ssl": my_ssl_ctx}
        engine = create_async_engine(
            uri,
            echo=self.echo,
            connect_args={**ssl_args, "timeout": 120},
            pool_pre_ping=True,
            pool_size=POOL_SIZE,  # Number of connections to keep open in the pool
            pool_recycle=POOL_RECYCLE,  # Recycle connections after 1 hour
            max_overflow=MAX_OVERFLOW,  # Number of connections that can be opened beyond the pool_size
            pool_timeout=POOL_TIMEOUT,  # Raise an exception after 2 minutes if no connection is available
            # from the pool
        )
        return engine

    def connect_db(self) -> SSLEngine:
        """Connect to a database (typically used for external client connections)
        WARNING: You must remember to dispose the database to close connections
        """
        uri = self.get_conn_uri()
        ssl_args, ssl_cert_path = self.get_ssl_args()
        if not self.conn_params.ssl:
            ssl_args = {}
            ssl_cert_path = None
        if self.database_type == enums.DatabaseType.REDSHIFT:
            ssl_args = {"sslmode": self.conn_params.ssl_mode.value}
        engine = create_engine(
            uri,
            connect_args=ssl_args,
            echo=self.echo,
            poolclass=NullPool,
        )
        return self.create_ssl_engine(engine=engine, ssl_cert_path=ssl_cert_path)

    def verify_client_connection(self):
        engine = self.connect_db()
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
                logger.info("Connection successfully verified")
        except (Exception, sa.exc.OperationalError, psycopg2.OperationalError) as e:
            logger.error("Error in verify_client_connection %s", str(e))
            raise errors.ConnectorError("Database credentials are incorrect")
        finally:
            engine.dispose()


class AthenaConnector(Connector):
    database_type = enums.DatabaseType.ATHENA

    def __init__(self, conn_params: sch.SQLDBSchema, echo: bool = False):
        super().__init__(conn_params=conn_params, echo=echo)

    def get_conn_uri(self, hide_password: bool = False) -> str:
        query = self.conn_params.query or {}
        try:
            assert query[constants.ATHENA_STAGING_DIR_NAME]
        except (KeyError, AssertionError):
            # TODO: Add a specific error here instead of general exception
            raise Exception("To connect to Athena, the s3_staging_dir query argument must be provided.")
        try:
            aws_role_arn = query.pop(constants.AWS_ROLE_ARN_NAME)
            session = self.get_aws_session(aws_role_arn)
            credentials = session.get_credentials()
            if not credentials:
                raise ValueError("Unable to retrieve credentials")
            creds = credentials.get_frozen_credentials()
            assert creds.access_key, "Unable to retrieve AWS access key"
            assert creds.secret_key, "Unable to retrieve AWS secret key"
            self.conn_params.username = creds.access_key
            self.conn_params.password = creds.secret_key
            self.conn_params.query = {**query, "aws_session_token": creds.token}
            logger.debug("Using AWS role ARN to create AWS session for Athena connection.")
        except KeyError:
            logger.debug("Not using AWS role ARN to create AWS session for Athena connection.")
        return super()._get_conn_uri(hide_password=hide_password)

    def get_ssl_args(self) -> tuple:
        ssl_params = AthenaSSLParams(
            ssl_mode=self.conn_params.ssl_mode,
            ssl_root_cert=self.conn_params.ssl_root_cert,
        )
        ssl_args, ssl_cert_path = ssl_params.get_ssl_args()
        return ssl_args, ssl_cert_path

    def create_ssl_engine(self, engine: Engine, ssl_cert_path) -> SSLEngine:
        return super()._create_ssl_engine(engine=engine, ssl_cert_path=ssl_cert_path)

    def get_aws_session(self, aws_role_arn: str, session_uuid: Optional[uuid.UUID] = None) -> boto3.Session:
        """Get an AWS client session

        Parameters
        ----------
        uuid
            A unique UUID for the session.
        role_arn
            The AWS role ARN
        """
        if not session_uuid:
            session_uuid = uuid.uuid4()
        session_name = f"session_{session_uuid}"
        logger.debug("Created AWS session: %s", session_name)

        sts = boto3.client("sts")
        import time

        time.sleep(5)
        # Assume the client-specific role
        assumed_role = sts.assume_role(
            RoleArn=aws_role_arn,
            RoleSessionName=session_name,
        )

        credentials = assumed_role["Credentials"]
        return boto3.Session(
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
        )

    def get_inspector(self, conn: sa.Connection) -> athena.AthenaInspector:
        return athena.AthenaInspector.inspect(conn=conn)


class SnowflakeConnector(Connector):
    database_type = enums.DatabaseType.SNOWFLAKE

    def __init__(self, conn_params: sch.SQLDBSchema, echo: bool = False):
        super().__init__(conn_params=conn_params, echo=echo)

    def get_conn_uri(self, hide_password: bool = False) -> str:
        password = "*****" if hide_password else self.conn_params.password
        uri = "{driver}://{user}:{password}@{account}/{database}".format(
            driver=self.conn_params.drivername.value,
            user=self.conn_params.username,
            password=password,
            account=self.conn_params.host,
            database=self.conn_params.database_name,
        )
        return uri

    def get_ssl_args(self) -> tuple:
        ssl_params = SnowflakeSSLParams(
            ssl_mode=self.conn_params.ssl_mode,
            ssl_root_cert=self.conn_params.ssl_root_cert,
        )
        ssl_args, ssl_cert_path = ssl_params.get_ssl_args()
        return ssl_args, ssl_cert_path

    def create_ssl_engine(self, engine: Engine, ssl_cert_path) -> SSLEngine:
        return super()._create_ssl_engine(engine=engine, ssl_cert_path=ssl_cert_path)

    def get_inspector(self, conn: sa.Connection) -> snowflake.SnowflakeInspector:
        return snowflake.SnowflakeInspector.inspect(conn=conn)


class SQLServerConnector(Connector):
    database_type = enums.DatabaseType.SQL_SERVER

    def __init__(self, conn_params: sch.SQLDBSchema, echo: bool = False):
        super().__init__(conn_params=conn_params, echo=echo)

    def get_conn_uri(self, hide_password: bool = False) -> str:
        if self.conn_params.query is None:
            self.conn_params.query = {}
        self.conn_params.query["driver"] = os.getenv("SQL_SERVER_ODBC_DRIVER") or "ODBC Driver 18 for SQL Server"
        return super()._get_conn_uri(hide_password=hide_password)

    def get_ssl_args(self) -> tuple:
        ssl_params = MSSQLSSLParams(
            ssl_mode=self.conn_params.ssl_mode,
            ssl_root_cert=self.conn_params.ssl_root_cert,
        )
        ssl_args, ssl_cert_path = ssl_params.get_ssl_args()
        return ssl_args, ssl_cert_path

    def create_ssl_engine(self, engine: Engine, ssl_cert_path) -> SSLEngine:
        return super()._create_ssl_engine(engine=engine, ssl_cert_path=ssl_cert_path)

    def get_inspector(self, conn: sa.Connection) -> sql_server.MSSQLServerInspector:
        return sql_server.MSSQLServerInspector.inspect(conn=conn)


class RedshiftConnector(Connector):
    database_type = enums.DatabaseType.REDSHIFT

    def __init__(self, conn_params: sch.SQLDBSchema, echo: bool = False):
        super().__init__(conn_params=conn_params, echo=echo)

    def get_conn_uri(self, hide_password: bool = False) -> str:
        return super()._get_conn_uri(hide_password=hide_password)

    def get_ssl_args(self) -> tuple:
        ssl_params = PostgresSSLParams(
            ssl_mode=self.conn_params.ssl_mode,
            ssl_root_cert=self.conn_params.ssl_root_cert,
        )
        ssl_args, ssl_cert_path = ssl_params.get_ssl_args()
        return ssl_args, ssl_cert_path

    def create_ssl_engine(self, engine: Engine, ssl_cert_path) -> SSLEngine:
        if hasattr(engine.dialect, "_set_backslash_escapes"):
            engine.dialect._set_backslash_escapes = lambda _: None
        return SSLEngine(original_engine=engine, ssl_cert_path=ssl_cert_path)

    def get_inspector(self, conn: sa.Connection) -> redshift.RedshiftInspector:
        return redshift.RedshiftInspector.inspect(conn=conn)


class PostgresConnector(Connector):
    database_type = enums.DatabaseType.POSTGRES

    def __init__(self, conn_params: sch.SQLDBSchema, echo: bool = False):
        super().__init__(conn_params=conn_params, echo=echo)

    def get_conn_uri(self, hide_password: bool = False) -> str:
        return super()._get_conn_uri(hide_password=hide_password)

    def get_ssl_args(self) -> tuple:
        ssl_params = PostgresSSLParams(
            ssl_mode=self.conn_params.ssl_mode,
            ssl_root_cert=self.conn_params.ssl_root_cert,
        )
        ssl_args, ssl_cert_path = ssl_params.get_ssl_args()
        return ssl_args, ssl_cert_path

    def create_ssl_engine(self, engine: Engine, ssl_cert_path) -> SSLEngine:
        return super()._create_ssl_engine(engine=engine, ssl_cert_path=ssl_cert_path)

    def get_inspector(self, conn: sa.Connection) -> postgres.PostgresInspector:
        return postgres.PostgresInspector.inspect(conn=conn)


class MySQLConnector(Connector):
    database_type = enums.DatabaseType.MYSQL

    def __init__(self, conn_params: sch.SQLDBSchema, echo: bool = False):
        super().__init__(conn_params=conn_params, echo=echo)

    def get_conn_uri(self, hide_password: bool = False) -> str:
        return super()._get_conn_uri(hide_password=hide_password)

    def get_ssl_args(self) -> tuple:
        ssl_params = MySQLSSLParams(
            ssl_mode=self.conn_params.ssl_mode,
            ssl_root_cert=self.conn_params.ssl_root_cert,
        )
        ssl_args, ssl_cert_path = ssl_params.get_ssl_args()
        return ssl_args, ssl_cert_path

    def create_ssl_engine(self, engine: Engine, ssl_cert_path) -> SSLEngine:
        return super()._create_ssl_engine(engine=engine, ssl_cert_path=ssl_cert_path)

    def get_inspector(self, conn: sa.Connection) -> mysql.MySQLInspector:
        return mysql.MySQLInspector.inspect(conn=conn)
