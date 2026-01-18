import asyncio
import io
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import ruamel.yaml
from jinja2 import Environment, TemplateSyntaxError, meta
from sqlalchemy.sql.elements import quoted_name

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.connector import Connector
from basejump.core.database.inspector import base
from basejump.core.models import enums, errors
from basejump.core.models import schemas as sch

logger = set_logging(handler_option="stream", name=__name__)

TABLE_PROFILING_TIME_LIMIT = 60 * 60


class TableManager:
    def __init__(
        self,
        conn_params: sch.SQLDBSchema,
        schemas: Optional[list[sch.DBSchema]] = None,
        verbose: bool = False,
    ):
        self.conn_params = conn_params
        self.db_type = conn_params.database_type
        self.schemas = schemas or conn_params.schemas
        self.include_default_schema = conn_params.include_default_schema
        self.client_db = Connector.get_database_to_connect(conn_params=conn_params)
        self.engine = self.client_db.connect_db()
        self.verbose = verbose

    @staticmethod
    def sanitize_jinja_schema_input(jinja_values: dict) -> None:
        for key, value in jinja_values.items():
            pattern = "^[a-zA-Z0-9_]+$"
            match = bool(re.match(pattern, value))
            if value == "":
                logger.warning("Missing jinja values.")
            elif not match or len(value) > 63:
                logger.debug("Here are the jinja values that can't be rendered: %s", jinja_values)

    @classmethod
    def render_query_jinja(cls, jinja_str: str, schemas: list[sch.DBSchema]):
        """Render the jinja in the SQL query string"""
        jinja_env = Environment(autoescape=True)
        for schema in schemas:
            if schema.jinja_values:
                # Putting this here out an abundance of caution - inputs were already sanitized
                # but it can't hurt to be too careful
                cls.sanitize_jinja_schema_input(jinja_values=schema.jinja_values)
                try:
                    template = jinja_env.from_string(jinja_str)
                    jinja_str = template.render(**schema.jinja_values)
                except TemplateSyntaxError as e:
                    logger.error("Error resolving schema jinja template: %s", e)
                    raise e
        return jinja_str

    @classmethod
    async def arender_query_jinja(cls, jinja_str: str, schemas: list[sch.DBSchema]):
        """Render the jinja in the SQL query string"""
        jinja_env = Environment(autoescape=True, enable_async=True)
        for schema in schemas:
            if schema.jinja_values:
                # Putting this here out an abundance of caution - inputs were already sanitized
                # but it can't hurt to be too careful
                cls.sanitize_jinja_schema_input(jinja_values=schema.jinja_values)
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

    def get_tables_names(self, inspector_callable: Callable, schema: sch.DBSchema) -> list[sch.SQLTable]:
        schema_nm_rendered = self.get_rendered_schema(schema=schema)
        schema_nm = schema.schema_nm
        logger.debug("Using the following rendered schema for inspector: %s", str(schema_nm_rendered))
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
            inspector = self.client_db.get_inspector(conn=conn)
            inspector_callable = inspector.get_permitted_table_names if permitted_only else inspector.get_table_names
            if self.include_default_schema:
                # Remove the default schema from schemas to avoid dups
                # HACK: Setting as public
                try:
                    default_schema_name = inspector.inspector.default_schema_name  # type:ignore
                    if default_schema_name is None:
                        raise Exception("default schema name returned None")
                    else:
                        default_schema = sch.DBSchema(
                            schema_nm=default_schema_name, schema_nm_rendered=default_schema_name
                        )
                except Exception as e:
                    logger.warning("Default schema property not implemented. Here is the error: %s", str(e))
                    default_schema_name = "public"
                    logger.info(f"Defaulting to {default_schema_name} as the schema name.")
                    default_schema = sch.DBSchema(
                        schema_nm=default_schema_name, schema_nm_rendered=default_schema_name
                    )
                self.schemas = [schema for schema in self.schemas if schema.schema_nm_rendered != default_schema_name]
                # Get default schema table names
                tbl_names += self.get_tables_names(inspector_callable=inspector_callable, schema=default_schema)
            if self.schemas:
                tbl_names += self.get_schema_table_names(inspector_callable=inspector_callable)
        return tbl_names

    def get_table_info(self, table: sch.SQLTable) -> sch.SQLTable:
        try:
            with self.engine.connect() as conn:
                inspector = self.client_db.get_inspector(conn=conn)
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
            # TODO: Make this an instance function and then use verbose so this debug statement isn't used
            # logger.debug("No primary keys defined for table")
            pass
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
        if self.verbose:
            logger.debug("Getting info for table: %s", table)
            logger.debug("Rendered schema: %s", table.table_schema_rendered)
        table_columns = {}
        for tbl_column in table.columns:
            table_columns[tbl_column.column_name] = tbl_column.dict()
        try:
            # try to retrieve table comment
            if self.verbose:
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
            if self.verbose:
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
            if self.verbose:
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
                    if self.verbose:
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
                    if self.verbose:
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
        logger.info("Retrieving database tables")
        tables_base = await asyncio.to_thread(self.ingest_table_names)
        if self.verbose:
            logger.debug("Here are the tables: %s", tables_base)
        tables = await self.get_tables_info(tables=tables_base)
        logger.info("Finishing retrieving database tables")
        return tables

    def _verify_schemas(self, schemas: set[str]):
        try:
            with self.engine.connect() as connection:
                inspector = self.client_db.get_inspector(conn=connection)
                permitted_schemas = inspector.get_permitted_schema_names()
                schema_diff = schemas - set(permitted_schemas)
                if schema_diff:
                    non_perm_schemas = ", ".join(list(schema_diff))
                    invalid_schemas = errors.InvalidSchemas(non_perm_schemas=non_perm_schemas)
                    logger.error("Invalid schemas %s", str(invalid_schemas))
                    raise invalid_schemas
                logger.info("The following schemas were successfully verified: %s", schemas)
        finally:
            self.dispose_engine()

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
            schema.schema_nm_rendered = self.get_rendered_schema(schema=schema)
        # Verify the user has access to all schemas that were provided/schemas provided exist
        await asyncio.to_thread(
            self._verify_schemas,
            schemas=set([schema.schema_nm_rendered for schema in self.conn_params.schemas]),  # type: ignore
        )
        return self.conn_params.schemas
