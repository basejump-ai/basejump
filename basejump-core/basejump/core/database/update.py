import uuid
from typing import Sequence

from redis.asyncio import Redis as RedisAsync
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database import db_utils
from basejump.core.database.crud import crud_connection, crud_table, crud_utils
from basejump.core.database.db_connect import ConnectDB, TableManager
from basejump.core.database.index import DBTableIndexer
from basejump.core.models import errors, models
from basejump.core.models import schemas as sch

logger = set_logging(handler_option="stream", name=__name__)


class DBManager:
    """Manage and update database connections and save them in the
    database for future use as a connection string"""

    def __init__(
        self,
        db: AsyncSession,
        connections: Sequence[models.DBConn],
        db_params: sch.DBParamsSchema,
        database: models.DBParams,
        client_user: sch.ClientUserInfo,
        db_id: int,
        db_uuid: uuid.UUID,
        embedding_model_info: sch.AzureModelInfo,
        small_model_info: sch.ModelInfo,
        redis_client_async: RedisAsync,
        sql_engine: AsyncEngine,
    ):
        self.db = db
        self.db_params = db_params
        self.database = database
        self.client_user = client_user
        self.db_id = db_id
        self.db_uuid = db_uuid
        self.embedding_model_info = embedding_model_info
        self.small_model_info = small_model_info
        self.redis_client_async = redis_client_async
        self.connections = connections

    async def validate_db(self):
        # Verify the updated params using a random connection
        conn_db = await crud_connection.get_conndb_from_connection(
            db_params=self.db_params, connection=self.connections[0]
        )
        # NOTE: Not validating schemas since we are doing it manually in the next step.
        # If very schemas was set to true, that would mean
        await crud_connection.verify_connection(conn_db=conn_db)

    # TODO: Create a master connection which has access to everything needed so you don't have to iterate over all
    # connections to find if the schema is able to connect
    async def validate_new_db_schema(self, schema: sch.DBSchema) -> None:
        """Finds if at least 1 connection is valid for a new schema."""
        valid_schema = False
        for connection in self.connections:
            logger.debug("Checking this connection: %s", connection)
            logger.debug("Checking this schema: %s", schema)
            conn_db = await crud_connection.get_conndb_from_connection(db_params=self.db_params, connection=connection)
            conn_db.conn_params.schemas = [schema]
            # NOTE: Only 1 schema needs to be valid to add the schema to the database list of accepted schemas
            try:
                await crud_connection.verify_connection(conn_db=conn_db)
                logger.debug("Schema is valid for this connection")
                valid_schema = True
                break
            except errors.InvalidSchemas:
                pass
        if not valid_schema:
            logger.error("Not one connection in the database can connect to the schema: %s", schema)
            db_schemas = set([schema.schema_nm for schema in self.db_params.schemas])
            invalid_schemas = errors.InvalidSchemas(str(db_schemas))
            raise invalid_schemas
        logger.info("Able to connect to new schema: %s", schema)

    async def validate_db_alias(self):
        # Ensure no duplicate alias names excluding the current alias in database
        client_dbs = await crud_connection.get_client_dbs(db=self.db, client_id=self.client_user.client_id)
        for client_db in client_dbs:
            if client_db.db_id == self.database.db_id:
                continue
            other_db_alias = ConnectDB.decrypt_db({"database_name_alias": client_db.database_name_alias})[
                "database_name_alias"
            ]
            logger.debug("New DB Alias = %s", self.db_params.database_name_alias)
            logger.debug("Other DB Alias = %s", other_db_alias)
            if self.db_params.database_name_alias == other_db_alias:
                raise errors.DBAliasConflict

    async def get_index_db_tables(self):
        db_vector = await crud_connection.get_vector_connection_from_id(db=self.db, vector_id=self.database.vector_id)
        return DBTableIndexer(
            client_id=self.client_user.client_id,
            client_uuid=self.client_user.client_uuid,
            db_uuid=self.db_uuid,
            vector_uuid=db_vector.vector_uuid,
            embedding_model_info=self.embedding_model_info,
        )

    async def update_schemas(self, fetch_latest_tables: bool = False) -> list[str]:
        """Determine if there are new schemas. If there are and there is a schema map, update
        connections with matching schema maps.
        """

        if fetch_latest_tables:
            database = await crud_connection.get_database_params(db=self.db, db_uuid=self.db_uuid, get_tables=True)
        else:
            database = self.database
        tables = [sch.GetSQLTable.from_orm(table) for table in database.tables]
        tables_formatted = await db_utils.process_db_tables(tables=tables)
        schemas = {table.table_schema for table in tables_formatted}
        logger.debug("Here are the schemas: %s", schemas)
        db_schemas = set([schema.schema_nm for schema in self.db_params.schemas])
        logger.debug("Here are the DB schemas: %s", db_schemas)
        new_schemas = db_schemas - schemas
        updated_schemas = False
        for new_schema in new_schemas:
            for schema_map in self.db_params.schema_maps:
                if new_schema == schema_map.new_schema:
                    updated_schemas = True
        if updated_schemas:
            logger.info("Updated schemas detected: %s", new_schemas)
        index_db_tables = await self.get_index_db_tables()
        logger.debug("Updating the index for: %s", str(index_db_tables.vector_uuid))
        # Verify each schema + connection
        # TODO: If there was a master connection, then only 1 connection would need to be checked
        connections = await crud_connection.get_db_conns(db=self.db, db_id=self.db_id)

        if updated_schemas:
            # Map schemas to existing schemas
            if self.db_params.schema_maps:
                await crud_connection.update_connection_schemas(
                    db_params=self.db_params, schema_maps=self.db_params.schema_maps, connections=list(connections)
                )
            # Update existing table schemas
            await self.update_table_schemas(
                tables=tables_formatted,
                index_db_tables=index_db_tables,
            )
        return list(new_schemas)

    async def update_db(self) -> sch.GetDBParams:
        if not self.db_params.database_name_alias:
            self.db_params.database_name_alias = ConnectDB.decrypt_db(
                {"database_name_alias": self.database.database_name_alias}
            )["database_name_alias"]
        conn_db = await crud_connection.get_conndb_from_connection(
            db_params=self.db_params, connection=self.connections[0]
        )
        for key, value in conn_db.conn_params_bytes.dict().items():
            # TODO: Reference the db models directly since the names could change and hard coded is not best practice
            # Skip values not in SQLDB params
            if key in ["username", "password", "data_source_desc", "schema_maps"]:
                pass
            else:
                setattr(self.database, key, value)
        await self.db.commit()
        await self.db.refresh(self.database)
        get_db_params = crud_utils.helper_decrypt_db(database=self.database)
        logger.info("Client engine updated and saved in database")
        # TODO: Raise an error if schema maps don't match anything
        return get_db_params

    async def get_connection_params(self) -> list[sch.SQLConnSchema]:
        # Get connections to update
        connections_w_params = []
        for connection in self.connections:
            conn_db_to_update = await crud_connection.get_conndb_from_connection(
                db_params=self.db_params, connection=connection
            )
            sql_conn = sch.SQLConnSchema(
                conn_params=conn_db_to_update.conn_params,
                conn_id=connection.conn_id,
                conn_uuid=str(connection.conn_uuid),
                db_id=self.db_id,
                vector_id=connection.database_params.vector_id,
                db_uuid=str(self.db_uuid),
            )
            connections_w_params.append(sql_conn)
        return connections_w_params

    async def update_table_schemas(
        self,
        tables: list[sch.SQLTable],
        index_db_tables: DBTableIndexer,
    ) -> None:
        # Retrieve the tables from the DB
        tables_w_new_schema = []
        tbl_uuids = [table.tbl_uuid for table in tables if table.tbl_uuid is not None]
        db_tables = await crud_table.get_tables_from_uuid(db=self.db, tbl_uuids=tbl_uuids, include_cols=True)
        # Update the table schemas
        for db_table in db_tables:
            for table in tables:
                if str(db_table.tbl_uuid) != str(table.tbl_uuid):
                    continue
                for schema_map in self.db_params.schema_maps:
                    if table.table_schema == schema_map.old_schema:
                        # Update the table schema
                        full_table_name = TableManager.get_full_table_name(
                            table_name=table.table_name, schema=schema_map.new_schema
                        )
                        table.table_schema = schema_map.new_schema
                        table.full_table_name = full_table_name
                        tables_w_new_schema.append(table)
                        # Update the db table as well
                        db_table.table_name = full_table_name
                    # Update the database table columns
                    for db_column in db_table.columns:
                        if db_column.foreign_key_table_name:
                            foreign_key_table_name = db_utils.get_table_name(db_column.foreign_key_table_name)
                            db_column.foreign_key_table_name = TableManager.get_full_table_name(
                                table_name=foreign_key_table_name, schema=schema_map.new_schema
                            )
                    for column in table.columns:
                        if column.foreign_key_table_name:
                            foreign_key_table_name = db_utils.get_table_name(column.foreign_key_table_name)
                            column.foreign_key_table_name = TableManager.get_full_table_name(
                                table_name=foreign_key_table_name, schema=schema_map.new_schema
                            )

        # Update the schema table columns
        # Index any new schemas in the vector database
        # TODO: Submit this to the background
        for table in tables_w_new_schema:
            table_info = TableManager.format_table_info(table=table)
            table.table_info = table_info
        logger.info("Here are the tables w/new schemas: %s", tables_w_new_schema)
        await index_db_tables.update_index_from_tables(
            tables=tables_w_new_schema, redis_client_async=self.redis_client_async
        )

    async def check_for_updated_tables(
        self, connections: list[sch.SQLConnSchema], fetch_latest_tables: bool = False
    ) -> list[sch.SQLTable]:
        if fetch_latest_tables:
            database = await crud_connection.get_database_params(db=self.db, db_uuid=self.db_uuid, get_tables=True)
        else:
            database = self.database
        # Identify new tables
        prior_tables_base = [sch.GetSQLTable.from_orm(table) for table in database.tables]
        logger.debug("Here are the prior_tables_base: %s", prior_tables_base)
        prior_tables = await db_utils.process_db_tables(tables=prior_tables_base)
        distinct_prior_table_names = {table.full_table_name.lower() for table in prior_tables}
        prior_tables_dict = {table.full_table_name.lower(): table for table in prior_tables}
        logger.debug("Here are the prior table names: %s", distinct_prior_table_names)
        # HACK: Assuming the first connection is a 'master' connection
        # TODO: Update connections so there is a master connection and all other connections
        # have a subset of the master connection's access
        connections = connections[:1]
        for connection in connections:
            # Get current table names
            mng_tbls = TableManager(conn_params=connection.conn_params, schemas=connection.conn_params.schemas)
            current_tables = await mng_tbls.get_db_tables()
            # Check for new columns
            for current_table in current_tables:
                prior_table = prior_tables_dict.get(current_table.full_table_name.lower())
                if prior_table:
                    prior_table_columns_names = {column.column_name for column in prior_table.columns}
                    for current_table_column in current_table.columns:
                        if current_table_column.column_name not in prior_table_columns_names:
                            current_table_column.new = True
                            prior_table.columns.append(current_table_column)
            current_tables_dict = {table.full_table_name.lower(): table for table in current_tables}
            distinct_current_table_names = {table.full_table_name.lower() for table in current_tables}
            logger.debug("Here are the current table names: %s", distinct_current_table_names)
            new_table_names = distinct_current_table_names - distinct_prior_table_names
            for new_table_name in new_table_names:
                # Update sets and lists with new tables
                new_table = current_tables_dict[new_table_name]
                new_table.new = True
                prior_tables.append(new_table)
                distinct_prior_table_names.add(new_table_name)
        # The prior tables being returned now has new columns and new tables being returned
        return prior_tables
