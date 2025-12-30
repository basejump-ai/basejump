"""Get the database connection"""

import asyncio
import copy
import io
import json
import os
import re
import ssl
import tempfile
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, Type, Union

import psycopg2
import ruamel.yaml
import sqlalchemy as sa
from cryptography.fernet import Fernet
from jinja2 import Environment, TemplateSyntaxError, meta
from sqlalchemy import URL, Engine, create_engine, text
from sqlalchemy.engine import Row
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.sql.elements import quoted_name

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.inspector import (
    athena,
    base,
    mysql,
    postgres,
    redshift,
    snowflake,
    sql_server,
)
from basejump.core.models import constants, enums, errors
from basejump.core.models import schemas as sch
from basejump.core.models.models import Base, DBConn, DBParams

TABLE_PROFILING_TIME_LIMIT = 60 * 3

# Set constants
logger = set_logging(handler_option="stream", name=__name__)

# TODO: Get these set possibly as property / class attributes instead
SHARED_SCHEMAS = ["account"]
POOL_SIZE = 4
MAX_OVERFLOW = 4  # Number of connections that can be opened beyond the pool_size
POOL_RECYCLE = 3600  # Recycle connections after 1 hour
POOL_TIMEOUT = 60 * 3  # Raise an exception after 3 minutes if no connection is available from the pool


def get_table_schemas() -> list:
    return list(set([table.split(".")[0] for table in Base.metadata.tables.keys() if len(table.split(".")) > 1]))


def get_table_names() -> list[tuple]:
    return list(
        set(
            [
                ((table.split(".")[0], table.split(".")[1]) if len(table.split(".")) > 1 else (None, table))
                for table in Base.metadata.tables.keys()
            ]
        )
    )


class TableManager:
    def __init__(self, conn_params: sch.SQLDBSchema, schemas: Optional[list[sch.DBSchema]] = None):
        self.conn_params = conn_params
        self.db_type = conn_params.database_type
        self.schemas = schemas or conn_params.schemas
        self.include_default_schema = conn_params.include_default_schema
        self.conn_db = ConnectDB(conn_params=conn_params)
        self.engine = self.conn_db.connect_db(echo=False)

    @staticmethod
    def sanitize_jinja_schema_input(jinja_values: dict) -> None:
        for key, value in jinja_values.items():
            pattern = "^[a-zA-Z0-9_]+$"
            match = bool(re.match(pattern, value))
            if value == "":
                logger.warning("Missing jinja values.")
            elif not match or len(value) > 63:
                logger.debug("Here are the jinja values that can't be rendered: %s", jinja_values)

    @staticmethod
    def render_query_jinja(jinja_str: str, schemas: list[sch.DBSchema]):
        """Render the jinja in the SQL query string"""
        jinja_env = Environment(autoescape=True)
        for schema in schemas:
            if schema.jinja_values:
                # Putting this here out an abundance of caution - inputs were already sanitized
                # but it can't hurt to be too careful
                TableManager.sanitize_jinja_schema_input(jinja_values=schema.jinja_values)
                try:
                    template = jinja_env.from_string(jinja_str)
                    jinja_str = template.render(**schema.jinja_values)
                except TemplateSyntaxError as e:
                    logger.error("Error resolving schema jinja template: %s", e)
                    raise e
        return jinja_str

    @staticmethod
    async def arender_query_jinja(jinja_str: str, schemas: list[sch.DBSchema]):
        """Render the jinja in the SQL query string"""
        jinja_env = Environment(autoescape=True, enable_async=True)
        for schema in schemas:
            if schema.jinja_values:
                # Putting this here out an abundance of caution - inputs were already sanitized
                # but it can't hurt to be too careful
                TableManager.sanitize_jinja_schema_input(jinja_values=schema.jinja_values)
                try:
                    template = jinja_env.from_string(jinja_str)
                    jinja_str = await template.render_async(**schema.jinja_values)
                except TemplateSyntaxError as e:
                    logger.error("Error resolving schema jinja template: %s", e)
                    raise e
        return jinja_str

    @classmethod
    def get_rendered_schema(cls, schema: sch.DBSchema) -> str:
        rendered_schema = cls.render_query_jinja(schema.schema_nm, schemas=[schema])
        # HACK: Sometimes the rendered schema is already there, but no jinja values
        if schema.schema_nm_rendered is not None and "{{" in rendered_schema and "{{" not in schema.schema_nm_rendered:
            return schema.schema_nm_rendered
        return rendered_schema

    @staticmethod
    def get_full_table_name(table_name: str, schema: Optional[str] = None) -> str:
        return f"{schema}.{table_name}" if schema else table_name

    @property
    def schema_mapping(self):
        """Find the mapping to any templated schema"""
        return {schema.schema_nm_rendered: schema.schema_nm for schema in self.schemas}

    def dispose_engine(self):
        self.engine.dispose()

    def inspector_factory(self, conn: sa.Connection) -> base.BaseInspector:
        if self.db_type == enums.DatabaseType.REDSHIFT:
            return redshift.RedshiftInspector.inspect(conn=conn)
        elif self.db_type == enums.DatabaseType.POSTGRES:
            return postgres.PostgresInspector.inspect(conn=conn)
        elif self.db_type == enums.DatabaseType.MYSQL:
            return mysql.MySQLInspector.inspect(conn=conn)
        elif self.db_type == enums.DatabaseType.SQL_SERVER:
            return sql_server.MSSQLServerInspector.inspect(conn=conn)
        elif self.db_type == enums.DatabaseType.SNOWFLAKE:
            return snowflake.SnowflakeInspector.inspect(conn=conn)
        elif self.db_type == enums.DatabaseType.ATHENA:
            return athena.AthenaInspector.inspect(conn=conn)
        else:
            raise NotImplementedError(
                """A dialect specific inspector class needs to be made since the current \
selection is not supported."""
            )

    def get_tables_names(self, inspector_callable: Callable, schema: sch.DBSchema) -> list[sch.SQLTable]:
        schema_nm_rendered = self.get_rendered_schema(schema=schema)
        schema_nm = schema.schema_nm
        logger.debug("Using the following rendered schema for inspector: %s", schema_nm_rendered)
        tables = inspector_callable(
            schema=schema_nm_rendered,
            include_views=self.conn_params.include_views,
            include_materialized_views=self.conn_params.include_materialized_views,
            include_partitioned_tbls=self.conn_params.include_partitioned_tables,
        )
        tables_list = []
        for table in tables:
            if self.conn_params.table_filter_string:
                if self.conn_params.table_filter_string in table:
                    continue
            full_table_name = self.get_full_table_name(table_name=table, schema=schema_nm)
            tables_list += [
                sch.SQLTable(
                    table_name=table,
                    table_schema=schema_nm,
                    table_schema_rendered=schema_nm_rendered,
                    full_table_name=full_table_name,
                )
            ]
        return tables_list

    def get_schema_table_names(self, inspector_callable: Callable) -> list[sch.SQLTable]:
        assert self.schemas
        schema_tables = []
        for schema in self.schemas:
            logger.debug("Getting schema table names for the following schema: %s", schema)
            schema_tables += self.get_tables_names(inspector_callable=inspector_callable, schema=schema)
        return schema_tables

    def ingest_table_names(self, permitted_only: bool = False) -> list[sch.SQLTable]:
        """Returns a list of the names of the tables in the client database"""
        # Get tables not in a schema
        if not self.include_default_schema and not self.schemas:
            raise ValueError(errors.INVALID_SCHEMA_ARGS)
        tbl_names = []
        with self.engine.connect() as conn:
            inspector = self.inspector_factory(conn=conn)
            inspector_callable = inspector.get_permitted_table_names if permitted_only else inspector.get_table_names
            if self.include_default_schema:
                # Remove the default schema from schemas to avoid dups
                # HACK: Setting as public
                try:
                    default_schema = inspector.inspector.default_schema_name  # type:ignore
                except Exception as e:
                    logger.warning("Default schema property not implemented. Here is the error: %s", str(e))
                    default_schema = "public"
                self.schemas = [schema for schema in self.schemas if schema.schema_nm_rendered != default_schema]
                # Get default schema table names
                tbl_names += self.get_tables_names(inspector_callable=inspector_callable, schema=default_schema)
            if self.schemas:
                tbl_names += self.get_schema_table_names(inspector_callable=inspector_callable)
        return tbl_names

    def get_table_info(self, table: sch.SQLTable) -> sch.SQLTable:
        try:
            with self.engine.connect() as conn:
                inspector = self.inspector_factory(conn=conn)
                table_info = self.get_single_table_info_wrapper(table=table, inspector=inspector)
        except Exception as e:
            logger.error("Error in get_table_info %s", str(e))
            raise e

        return table_info

    def get_single_table_info_wrapper(self, table: sch.SQLTable, inspector: base.BaseInspector) -> sch.SQLTable:
        """Use this when using an Async Engine. Get table info for a single table."""
        table = self.get_single_table_info(table=table, inspector=inspector)
        table_info = self.format_table_info(table=table)
        table.table_info = table_info
        return table

    @staticmethod
    def format_table_info(table: sch.SQLTable) -> str:
        # Create a dictionary
        table_dict = table.dict(exclude_none=True, exclude_defaults=True, exclude={"primary_key"})
        # NOTE: Calling this description instead. Don't want to add if it doesn't exist
        table_dict["table_name"] = table_dict["full_table_name"]
        table_dict.pop("full_table_name", None)
        table_dict.pop("tbl_uuid", None)
        table_dict.pop("conn_uuid", None)
        table_dict.pop("table_schema", None)
        table_dict.pop("ignore", None)
        table_dict.pop("table_schema_rendered", None)
        try:
            table_dict["description"] = table_dict.pop("context_str", None)
            if not table_dict["description"]:
                del table_dict["description"]
        except KeyError:
            logger.debug("No description defined for table")
        try:
            table_dict["primary_keys"] = table_dict.pop("primary_keys")
            if not table_dict["primary_keys"]:
                del table_dict["primary_keys"]
        except Exception:
            logger.debug("No primary keys defined for table")
        # Have columns go last
        table_dict["columns"] = table_dict.pop("columns", None)
        # Create a YAML instance
        yaml = ruamel.yaml.YAML()
        # Dump to a string with block style
        yaml.indent(mapping=2, sequence=2, offset=0)  # Adjust indentation if needed
        yaml.default_flow_style = False  # Set to False to use block style
        stream = io.StringIO()
        yaml.dump(table_dict, stream)
        # Get the string value
        table_info = stream.getvalue()
        return table_info

    def is_column_case_sensitive(self, column_name):
        """
        Determines if a column name is case sensitive in Snowflake.
        In Snowflake, column names enclosed in double quotes are case sensitive.

        Args:
            column_name (str): The column name to check

        Returns:
            bool: True if the column name is case sensitive, False otherwise
        """

        # If column name is all uppercase or all lowercase, it's likely not quoted
        if column_name.isupper() or column_name.islower():
            return False

        # If it contains spaces or special characters, it would need quotes
        import re

        if re.search(r"[^a-zA-Z0-9_]", column_name):
            return True

        # If it's mixed case (contains both upper and lower), it needed quotes
        if not (column_name.isupper() or column_name.islower()):
            return True

        return False

    def get_single_table_info(self, table: sch.SQLTable, inspector: base.BaseInspector) -> sch.SQLTable:
        """Get table info for a single table.

        Notes
        -----
        Originally taken from llama index sql_wrapper.py
        """
        # Create a dictionary from the current column information
        logger.debug("Getting info for table: %s", table)
        logger.debug("Rendered schema: %s", table.table_schema_rendered)
        table_columns = {}
        for tbl_column in table.columns:
            table_columns[tbl_column.column_name] = tbl_column.dict()
        try:
            # try to retrieve table comment
            logger.debug("Here is the table name: %s", table.table_name)
            logger.debug("Here is the schema name: %s", table.table_schema_rendered)
            try:
                table_comment = inspector.get_table_comment(
                    table_name=table.table_name, schema=table.table_schema_rendered
                )["text"]
            except Exception as e:
                logger.warning("Exception when getting table comment: %s", str(e))
                table_comment = ""
            if table_comment and not table.description:
                table.description = table_comment
        except NotImplementedError:
            logger.warning("Not implemented error for dialect not supporting comments")
            # get_table_comment raises NotImplementedError for a dialect that does not support comments.
            pass
        columns = {}
        if not table.table_schema_rendered:
            raise Exception("There must be a rendered schema defined to avoid matching on only table name.")
        for column in inspector.get_columns(table_name=table.table_name, schema=table.table_schema_rendered):
            # if quoted then preserve casing
            logger.debug("Column: %s", column)
            logger.debug(
                "Here is the case sensitivity: %s",
                (self.is_column_case_sensitive(column["name"]) or isinstance(column["name"], quoted_name)),
            )
            logger.debug("Here is the name: %s", column["name"])
            if self.is_column_case_sensitive(column["name"]) or isinstance(column["name"], quoted_name):
                column_name = str(column["name"])
            # SQLAlchemy returns lower case by default must uppercase for dbs that use default uppercase
            elif self.db_type in enums.UPPERCASE_DEFAULT_DB:
                column_name = str(column["name"]).upper()
            else:
                column_name = str(column["name"])
            logger.debug("Column name: %s", column_name)
            columns[column["name"]] = sch.SQLTableColumn(
                column_name=column["name"],
                column_type=str(column["type"]),
                description=str(column.get("comment")),
                quoted=(self.is_column_case_sensitive(column["name"]) or isinstance(column["name"], quoted_name)),
            )
        # TODO: Get the schema included in these definitions as well
        for foreign_key in inspector.get_foreign_keys(table_name=table.table_name, schema=table.table_schema_rendered):
            for column_name, foreign_key_col_nm in zip(
                foreign_key["constrained_columns"], foreign_key["referred_columns"]
            ):
                if self.conn_params.table_filter_string:
                    if self.conn_params.table_filter_string in foreign_key["referred_table"]:
                        continue
                col_info = columns[column_name]
                foreign_tbl_nm = (
                    ".".join([foreign_key["referred_schema"], foreign_key["referred_table"]])
                    if foreign_key["referred_schema"]
                    else foreign_key["referred_table"]
                )
                # If there is schema templated, the schema needs to be updated to use the template
                foreign_tbl_schema = foreign_tbl_nm.split(".")[0] if len(foreign_tbl_nm.split(".")) > 1 else None
                if foreign_tbl_schema:
                    logger.debug("Here is the foreign_tbl_schema: %s", foreign_tbl_schema)
                    logger.debug("Here is the schema mapping: %s", self.schema_mapping)
                    foreign_tbl_schema = self.schema_mapping.get(foreign_tbl_schema)
                    if foreign_tbl_schema:
                        tbl_nm = foreign_tbl_nm.split(".")[1]
                        foreign_tbl_nm = f"{foreign_tbl_schema}.{tbl_nm}"
                        col_info.foreign_key_table_name = foreign_tbl_nm
                        col_info.foreign_key_column_name = foreign_key_col_nm
        # Overwrite column information if it already exists
        # TODO: Make this more elegant, probably can use .dict() similar to how tables are being handled
        # with the pydantic schema
        for key, value in table_columns.items():
            if value["foreign_key_table_name"]:
                columns[value["column_name"]].foreign_key_table_name = value["foreign_key_table_name"]
            if value["foreign_key_column_name"]:
                columns[value["column_name"]].foreign_key_column_name = value["foreign_key_column_name"]
            if value["description"]:
                columns[value["column_name"]].description = value["description"]
            if value["distinct_values"]:
                columns[value["column_name"]].distinct_values = value["distinct_values"]
            if value["ignore"]:
                del columns[value["column_name"]]
        # HACK: Reinstantiating new objects is only done to preserve ordering
        table.columns = [value for key, value in columns.items()]
        return table

    async def get_tables_info(self, tables: list[sch.SQLTable]) -> list[sch.SQLTable]:
        table_results = []
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            futures = [loop.run_in_executor(pool, self.get_table_info, table) for table in tables]
            for future in asyncio.as_completed(futures, timeout=TABLE_PROFILING_TIME_LIMIT):
                try:
                    result = await future
                    logger.debug("Table profiling result: %s", result)
                except Exception as exc:
                    logger.error("Error when running table profiling in threads: %s", str(exc))
                    raise exc
                else:
                    table_results.append(result)

        return table_results

    async def get_db_tables(self) -> list[sch.SQLTable]:
        """Helper function to retrieve client database information"""
        # Get the tables from the client database
        tables_base = await asyncio.to_thread(self.ingest_table_names)
        logger.debug("Here is tables base: %s", tables_base)
        tables = await self.get_tables_info(tables=tables_base)
        return tables


class SSLParams(ABC):
    def __init__(self, drivername: enums.DBDriverName, ssl_mode: enums.SSLModes, ssl_root_cert: Optional[str]):
        self.drivername = drivername
        self.ssl_mode = ssl_mode
        self.ssl_root_cert = ssl_root_cert

    def get_cert(self):
        if not self.ssl_root_cert:
            raise errors.SSLConfigError
        # Create a temporary file for the root cert
        named_file = tempfile.NamedTemporaryFile(suffix=".crt", delete=False)
        named_file.write(bytes(self.ssl_root_cert.encode()))
        named_file.close()
        return named_file.name

    @abstractmethod
    def get_require(self) -> tuple:
        """Ensure there is SSL/TSL encryption, but don't verify the certificate"""
        pass

    @abstractmethod
    def get_verify_ca(self) -> tuple:
        """Ensure there is SSL/TSL encryption and verify the certificate, but don't verify the hostname"""
        pass

    @abstractmethod
    def get_verify_full(self) -> tuple:
        """Ensure there is SSL/TSL encryption, verify the certificate, and verify the hostname"""
        pass


class BaseSSLContextParams(SSLParams):
    def __init__(self, drivername: enums.DBDriverName, ssl_mode: enums.SSLModes, ssl_root_cert: Optional[str]):
        super().__init__(drivername=drivername, ssl_mode=ssl_mode, ssl_root_cert=ssl_root_cert)

    def get_require(self) -> tuple:
        """Ensure there is SSL/TSL encryption, but don't verify the certificate"""
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        return {"ssl": ssl_context}, None

    def get_verify_ca(self) -> tuple:
        """Ensure there is SSL/TSL encryption and verify the certificate, but don't verify the hostname"""
        ssl_cert_path = self.get_cert()
        ssl_context = ssl.create_default_context(cafile=ssl_cert_path)
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        return {"ssl": ssl_context}, ssl_cert_path

    def get_verify_full(self) -> tuple:
        """Ensure there is SSL/TSL encryption, verify the certificate, and verify the hostname"""
        ssl_cert_path = self.get_cert()
        ssl_context = ssl.create_default_context(cafile=ssl_cert_path)
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        return {"ssl": ssl_context}, ssl_cert_path


class MySQLSSLParams(BaseSSLContextParams):
    pass


class SnowflakeSSLParams(BaseSSLContextParams):
    pass


class AthenaSSLParams(BaseSSLContextParams):
    pass


class PostgresSSLParams(SSLParams):
    def __init__(self, drivername: enums.DBDriverName, ssl_mode: enums.SSLModes, ssl_root_cert: Optional[str]):
        super().__init__(drivername=drivername, ssl_mode=ssl_mode, ssl_root_cert=ssl_root_cert)

    def get_require(self):
        """Ensure there is SSL/TSL encryption, but don't verify the certificate"""
        return {"sslmode": self.ssl_mode.value}, None

    def get_verify_ca(self):
        """Ensure there is SSL/TSL encryption and verify the certificate, but don't verify the hostname"""
        ssl_cert_path = self.get_cert()
        return {"sslmode": self.ssl_mode.value, "sslrootcert": ssl_cert_path}, ssl_cert_path

    def get_verify_full(self):
        """Ensure there is SSL/TSL encryption, verify the certificate, and verify the hostname"""
        ssl_cert_path = self.get_cert()
        return {"sslmode": self.ssl_mode.value, "sslrootcert": ssl_cert_path}, ssl_cert_path


class MSSQLSSLParams(SSLParams):
    def __init__(self, drivername: enums.DBDriverName, ssl_mode: enums.SSLModes, ssl_root_cert: Optional[str]):
        super().__init__(drivername=drivername, ssl_mode=ssl_mode, ssl_root_cert=ssl_root_cert)

    def get_require(self) -> tuple:
        """Ensure there is SSL/TSL encryption, but don't verify the certificate"""
        return {"encrypt": "YES", "trust_server_certificate": "YES"}, None

    def get_verify_ca(self) -> tuple:
        """Ensure there is SSL/TSL encryption and verify the certificate, but don't verify the hostname"""
        # NOTE: Use verify full since no distinction to not check hostname
        return self.get_verify_full()

    def get_verify_full(self) -> tuple:
        """Ensure there is SSL/TSL encryption, verify the certificate, and verify the hostname"""
        # NOTE: MS Certs are already trusted and installed so no path is necessary
        return {"encrypt": "YES", "trust_server_certificate": "No"}, None


def ssl_param_factory(
    drivername: enums.DBDriverName, ssl_mode: enums.SSLModes, ssl_root_cert: Optional[str] = None
) -> tuple:
    """Retrieves the relevant ssl args based on the driver and ssl mode"""
    ssl_driver_lkup: dict[enums.DBDriverName, Type[SSLParams]] = {
        enums.DBDriverName.ATHENA: AthenaSSLParams,
        enums.DBDriverName.POSTGRES: PostgresSSLParams,
        enums.DBDriverName.MYSQL: MySQLSSLParams,
        enums.DBDriverName.REDSHIFT: PostgresSSLParams,
        enums.DBDriverName.SQL_SERVER: MSSQLSSLParams,
        enums.DBDriverName.SNOWFLAKE: SnowflakeSSLParams,
    }
    ssl_driver = ssl_driver_lkup.get(drivername)
    if not ssl_driver:
        raise NotImplementedError(
            f"""Drivername '{drivername.value}' not found as a SSL supported driver. \
Update the code to support the driver using SSL if needed."""
        )
    ssl_params = ssl_driver(drivername=drivername, ssl_mode=ssl_mode, ssl_root_cert=ssl_root_cert)
    if ssl_mode == enums.SSLModes.REQUIRE:
        return ssl_params.get_require()
    elif ssl_mode == enums.SSLModes.VERIFY_CA:
        return ssl_params.get_verify_ca()
    elif ssl_mode == enums.SSLModes.VERIFY_FULL:
        return ssl_params.get_verify_full()
    else:
        raise NotImplementedError(f"SSL mode '{ssl_mode.value}' is not supported")


class SSLEngine(Engine):
    """Light wrapper to handle SSL certificate cleanup"""

    def __init__(self, original_engine: Engine, ssl_cert_path: Optional[str]):
        # Copy all attributes from the original engine
        self.__dict__.update(original_engine.__dict__)
        self._original_engine = original_engine
        self.ssl_cert_path = ssl_cert_path

    def __del__(self):
        """Ensure cleanup of temp file if context manager wasn't used"""
        if self.ssl_cert_path:
            try:
                os.remove(self.ssl_cert_path)
            except FileNotFoundError as e:
                logger.warning("File not found %s", str(e))
                pass


class ConnectDB:
    def __init__(self, conn_params: sch.SQLDBSchema):
        self.conn_params = conn_params

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
        return cls(conn_params=conn_params_obj)

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
        return cls(conn_params=conn_params_obj)

    @staticmethod
    def decrypt_db(dict_to_decrypt: dict) -> dict:
        # Decrypt the sensitive information

        f = Fernet(os.environ["ENCRYPTION_KEY"])
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
        f = Fernet(os.environ["ENCRYPTION_KEY"])
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
            echo=False,
            connect_args={**ssl_args, "timeout": 120},
            pool_pre_ping=True,
            pool_size=POOL_SIZE,  # Number of connections to keep open in the pool
            pool_recycle=POOL_RECYCLE,  # Recycle connections after 1 hour
            max_overflow=MAX_OVERFLOW,  # Number of connections that can be opened beyond the pool_size
            pool_timeout=POOL_TIMEOUT,  # Raise an exception after 2 minutes if no connection is available
            # from the pool
        )
        return engine

    def get_conn_uri(self, hide_password: bool = False) -> str:
        """Create a database URI"""
        query = self.conn_params.query or {}
        if self.conn_params.database_type == enums.DatabaseType.SQL_SERVER:
            query["driver"] = os.environ["SQL_SERVER_ODBC_DRIVER"]
        elif self.conn_params.database_type == enums.DatabaseType.ATHENA:
            try:
                assert query[constants.ATHENA_STAGING_DIR_NAME]
            except (KeyError, AssertionError):
                # TODO: Add a specific error here instead of general exception
                raise Exception("To connect to Athena, the s3_staging_dir query parameter must be provided.")
        elif self.conn_params.database_type == enums.DatabaseType.SNOWFLAKE:
            password = "*****" if hide_password else self.conn_params.password
            uri = "{driver}://{user}:{password}@{account}/{database}".format(
                driver=self.conn_params.drivername.value,
                user=self.conn_params.username,
                password=password,
                account=self.conn_params.host,
                database=self.conn_params.database_name,
            )
            return uri
        uri_obj = URL.create(
            drivername=self.conn_params.drivername.value,
            username=self.conn_params.username,
            password=self.conn_params.password,
            host=self.conn_params.host,
            port=self.conn_params.port,
            database=self.conn_params.database_name,
            query=query,
        )
        return uri_obj.render_as_string(hide_password=hide_password)

    def connect_db(self, echo: bool = False) -> SSLEngine:
        """Connect to a database (typically used for external client connections)
        WARNING: You must remember to dispose the database to close connections
        """
        uri = self.get_conn_uri()
        if not self.conn_params.ssl:
            ssl_args = {}
            ssl_cert_path = None
        else:
            assert isinstance(self.conn_params.drivername, enums.DBDriverName)
            ssl_args, ssl_cert_path = ssl_param_factory(
                drivername=self.conn_params.drivername,
                ssl_mode=self.conn_params.ssl_mode,
                ssl_root_cert=self.conn_params.ssl_root_cert,
            )
        engine = create_engine(
            uri,
            connect_args=ssl_args,
            echo=echo,
            poolclass=NullPool,
        )
        if self.conn_params.database_type == enums.DatabaseType.REDSHIFT:
            if hasattr(engine.dialect, "_set_backslash_escapes"):
                engine.dialect._set_backslash_escapes = lambda _: None
        return SSLEngine(original_engine=engine, ssl_cert_path=ssl_cert_path)

    def verify_client_connection(self):
        engine = self.connect_db()
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
                logger.info("Connection successfully verified")
        except (Exception, sa.exc.OperationalError, psycopg2.OperationalError) as e:
            logger.error("Error in verify_client_connection %s", str(e))
            raise errors.ConnectDBError("Database credentials are incorrect")
        finally:
            engine.dispose()

    def _verify_schemas(self, schemas: set[str]):
        mng_tables = TableManager(conn_params=self.conn_params)
        try:
            with mng_tables.engine.connect() as connection:
                inspector = mng_tables.inspector_factory(conn=connection)
                permitted_schemas = inspector.get_permitted_schema_names()
                schema_diff = schemas - set(permitted_schemas)
                if schema_diff:
                    non_perm_schemas = ", ".join(list(schema_diff))
                    invalid_schemas = errors.InvalidSchemas(non_perm_schemas=non_perm_schemas)
                    logger.error("Invalid schemas %s", str(invalid_schemas))
                    raise invalid_schemas
                logger.info("The following schemas successfully verified: %s", schemas)
        finally:
            mng_tables.dispose_engine()

    # TODO: Could likely use regex instead
    @classmethod
    def validate_jinja_braces(cls, string_to_validate: str, initial_pass: bool = True) -> bool:
        """Validate the jinja in the string is formatted correctly"""
        if initial_pass:
            if string_to_validate.count("{") != string_to_validate.count("}"):
                raise errors.InvalidJinjaBraceCount
        curly_brace_starting_idx = string_to_validate.find("{")
        if curly_brace_starting_idx != -1:
            try:
                assert string_to_validate[curly_brace_starting_idx + 1] == "{"
            except (AssertionError, IndexError):
                raise errors.InvalidJinjaStartingBrace
            try:
                assert string_to_validate[curly_brace_starting_idx + 2] not in [
                    "{",
                    "}",
                ]
            except (AssertionError, IndexError):
                raise errors.InvalidJinjaContent
            cls.validate_jinja_braces(string_to_validate[curly_brace_starting_idx + 2 :], False)  # noqa
        curly_brace_ending_idx = string_to_validate.find("}")
        if curly_brace_ending_idx != -1:
            try:
                assert string_to_validate[curly_brace_ending_idx + 1] == "}"
            except (AssertionError, IndexError):
                raise errors.InvalidJinjaEndingBrace
            cls.validate_jinja_braces(string_to_validate[curly_brace_ending_idx + 2 :], False)  # noqa
        return True

    def validate_schema_keys(self, schemas: list[sch.DBSchema]):
        # Create a Jinja environment
        env = Environment(autoescape=True)
        for schema in schemas:
            if not schema.jinja_values:
                continue
            # Define the template string
            template_string = schema.schema_nm
            # Parse the template
            parsed_content = env.parse(template_string)
            # Find the variables used in the template
            variables = meta.find_undeclared_variables(parsed_content)
            # Assert all the variables are defined
            try:
                assert len(set(schema.jinja_values.keys()) & variables) == len(variables)
            except AssertionError:
                raise errors.MissingJinjaKey

    async def validate_schemas(self) -> list[sch.DBSchema]:
        # Validate the jinja is correctly formatted
        assert self.conn_params.schemas
        schema_nms = " ".join([schema.schema_nm for schema in self.conn_params.schemas])
        self.validate_jinja_braces(string_to_validate=schema_nms)
        # Validate all of the keys exist
        self.validate_schema_keys(schemas=self.conn_params.schemas)
        # Render the schema names
        for schema in self.conn_params.schemas:
            schema.schema_nm_rendered = TableManager.get_rendered_schema(schema=schema)
        # Verify the user has access to all schemas that were provided/schemas provided exist
        await asyncio.to_thread(
            self._verify_schemas,
            schemas=set([schema.schema_nm_rendered for schema in self.conn_params.schemas]),  # type: ignore
        )
        return self.conn_params.schemas


class LocalSession:
    def __init__(self, client_id: int, engine: AsyncEngine, include_dummy_tables: bool = False):
        self._engine = engine
        self.session = None
        self.client_id = client_id
        self.include_dummy_tables = include_dummy_tables

    @property
    def base_schemas(self):
        # Get the schemas from the DB models
        schemas = set(table.schema for table in Base.metadata.tables.values() if table.schema)
        return schemas

    @property
    def schemas(self) -> list:
        return list(self.schema_map.values())

    @property
    def schema_map(self) -> dict:
        schema_map = {}
        for schema_base in self.base_schemas:
            schema = self.get_client_schema(
                client_id=self.client_id,
                schema=str(schema_base),
                include_dummy_tables=self.include_dummy_tables,
            )
            schema_map[schema_base] = schema

        return schema_map

    @staticmethod
    def get_client_schema(client_id: int, schema: str, include_dummy_tables: bool = False) -> str:
        if include_dummy_tables:
            return schema + str(client_id)
        return schema + str(client_id) if schema not in SHARED_SCHEMAS else schema

    async def engine(self):
        if self.client_id == 0:
            return self._engine
        engine_mapped = self._engine.execution_options(schema_translate_map=self.schema_map)
        return engine_mapped

    async def create_schemas(self):
        session = async_sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
        async with session.begin() as conn:
            for schema in get_table_schemas():
                # Create schemas without client IDs if they don't exist
                await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
            for schema in self.schemas:
                # Create schemas with client IDs if they don't exist
                await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))

    async def _manage_views(self, create: bool = False):
        session = async_sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
        async with session.begin() as conn:
            for schema, table_name in get_table_names():
                full_tbl_name = f"{schema}.{table_name}" if schema else table_name
                for client_schema in self.schemas:
                    if client_schema in SHARED_SCHEMAS:
                        continue
                    if schema in client_schema:
                        client_full_tbl_name = f"{client_schema}.{table_name}"
                        if create:
                            stmt = f"""\
    CREATE VIEW {client_full_tbl_name} WITH(security_invoker=TRUE) AS \
    SELECT * FROM {full_tbl_name} \
    WHERE client_id = {self.client_id}"""
                            logger.debug(f"Creating view: {client_full_tbl_name}")
                        else:
                            stmt = f"DROP VIEW IF EXISTS {client_full_tbl_name}"
                            logger.debug(f"Dropping view: {client_full_tbl_name}")
                        await conn.execute(text(stmt))
                        break

    async def create_views(self):
        await self._manage_views(create=True)

    async def delete_views(self):
        await self._manage_views(create=False)

    async def get_session(self):
        """Get a database session"""
        engine = await self.engine()
        session = async_sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=engine)
        return session

    async def open(self):
        """Get a database session"""
        session = await self.get_session()
        self.session = session()
        return self.session

    async def close(self):
        assert self.session
        await self.session.close()

    async def delete_schemas(self, delete_shared=False):
        session = await self.get_session()
        async with session.begin() as conn:
            for schema in self.schemas:
                if delete_shared:
                    await conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
                elif schema not in SHARED_SCHEMAS:
                    await conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
            await conn.commit()
