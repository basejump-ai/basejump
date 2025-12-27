"""Create vector indexes from the database"""

import asyncio
import uuid
from typing import Optional

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database import db_utils
from basejump.core.database.aicatalog import AICatalog
from basejump.core.database.crud import crud_connection, crud_table
from basejump.core.database.db_connect import LocalSession, TableManager
from basejump.core.database.vector_utils import get_index_name
from basejump.core.models import constants, enums
from basejump.core.models import schemas as sch
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.vector_stores.redis import RedisVectorStore
from redis.asyncio import Redis as RedisAsync
from redisvl.schema import IndexSchema
from sqlalchemy.ext.asyncio import AsyncEngine

logger = set_logging(handler_option="stream", name=__name__)


class DBTableIndexer:
    """Index SQL table information from the database"""

    def __init__(
        self,
        client_id: int,
        client_uuid: uuid.UUID,
        db_uuid: uuid.UUID,
        embedding_model_info: sch.AzureModelInfo,
        index_name: Optional[str] = None,
        vector_uuid: Optional[uuid.UUID] = None,
        vector_database_vendor: enums.VectorVendorType = enums.VectorVendorType.REDIS,  # noqa
    ):
        # Setup variables
        self.client_id = client_id
        self.client_uuid = client_uuid
        self.db_uuid = db_uuid
        self.embedding_model_info = embedding_model_info
        self.vector_uuid = vector_uuid or uuid.uuid4()
        assert vector_database_vendor
        self.vector_database_vendor = vector_database_vendor
        self.vector_datasource_type = enums.VectorSourceType.TABLE
        self.index_name = get_index_name(client_id=self.client_id) if not index_name else index_name

    async def to_nodes_from_tables(self, tables: list[sch.SQLTable]) -> list[TextNode]:
        # Originally taken from the llama_index table_node_mapping.py module
        nodes = []
        for table in tables:
            table_text = f"Schema of table {table.full_table_name}:\n" f"{table.table_info}\n"

            metadata = {
                "name": table.full_table_name,
                "client_uuid": str(self.client_uuid),
                "db_uuid": str(self.db_uuid),
                "vector_type": self.vector_datasource_type.value,
            }

            if table.context_str is not None:
                table_text += f"Context of table {table.full_table_name}:\n"
                table_text += table.context_str
                metadata["context"] = table.context_str

            metadata["table_info"] = table.table_info  # type:ignore
            node = TextNode(
                text=table_text,
                metadata=metadata,
                excluded_embed_metadata_keys=["name", "context", "table_info", "client_uuid", "db_uuid"],
                excluded_llm_metadata_keys=["context", "table_info", "client_uuid", "chat_uuid", "db_uuid"],
            )
            node.id_ = str(table.tbl_uuid)
            nodes.append(node)
        return nodes

    async def create_index(self, tables: list[sch.SQLTable], redis_client_async: RedisAsync) -> None:
        """Creating and update use the same process, this function is simply here for completeness or
        those looking for a create index function"""
        await self.update_index_from_tables(tables=tables, redis_client_async=redis_client_async)

    async def update_index_from_tables(self, tables: list[sch.SQLTable], redis_client_async: RedisAsync) -> None:
        """Create an index with embeddings
        for the tables in the database"""
        # Create the nodes
        # TODO: Move embedding to its own function
        # logger.debug("Here is the list of tables: %s", tables)
        logger.debug("Creating nodes...")
        nodes = await self.to_nodes_from_tables(tables)
        # Add embedding for each node
        # TODO: Add progress bar for embedding step
        logger.debug("Creating node embeddings...")
        ai_catalog = AICatalog()
        embed_model = ai_catalog.get_embedding_model(model_info=self.embedding_model_info)
        for node in nodes:
            node.embedding = await embed_model.aget_text_embedding(node.get_content(metadata_mode=MetadataMode.EMBED))
        await self._update_index(nodes=list(nodes), tables=tables, redis_client_async=redis_client_async)

    async def _update_index(
        self, nodes: list[TextNode], tables: list[sch.SQLTable], redis_client_async: RedisAsync
    ) -> None:
        schema = IndexSchema.from_dict(
            {
                "index": {"name": self.index_name, "prefix": self.index_name + "/vector"},
                "fields": [
                    # Required fields
                    {"name": "id", "type": "tag"},
                    {"name": "doc_id", "type": "tag"},
                    {"name": "text", "type": "text"},
                    {"name": "vector", "type": "vector", "attrs": {"dims": 1536, "algorithm": "flat"}},
                    *constants.VECTOR_FILTERS,
                ],
            }
        )
        vector_store = RedisVectorStore(redis_client_async=redis_client_async, schema=schema, legacy_filters=True)
        logger.debug("Deleting overlapping nodes for %s documents...", len(nodes))
        update_nodes_error = False
        try:
            await vector_store.adelete_nodes(node_ids=[node.node_id for node in nodes])
        except Exception as e:
            logger.warning("Deleting threw error: %s", e)
            update_nodes_error = True
        logger.debug("Upserting into Redis...")
        # Find tables to ignore (if any) and omit them
        try:
            ignore_tables_uuid = [str(table.tbl_uuid) for table in tables if table.ignore]
            final_nodes = [node for node in nodes if node.node_id not in ignore_tables_uuid]
            await vector_store.async_add(final_nodes)  # type: ignore
            logger.info("Finished indexing")
        except Exception as e:
            logger.error("Error inn _update_index %s", str(e))
            update_nodes_error = True
        if update_nodes_error:
            # TODO: This seems backwards, the hash key should likely be provided first
            await redis_client_async.hset(  # type: ignore
                str(self.vector_uuid),
                enums.RedisHashKeys.DB_INDEX_UPDATE_STATUS_KEY.value,
                enums.RedisValues.ERROR_UPDATING_DB_INDEX.value,
            )
            raise Exception("Error when updating Redis index")


async def index_db(
    index_db_tables: DBTableIndexer,
    conn_params: sch.SQLDBSchema,
    client_user: sch.ClientUserInfo,
    db_id: int,
    db_uuid: uuid.UUID,
    conn_id: int,
    small_model_info: sch.ModelInfo,
    redis_client_async: RedisAsync,
    sql_engine: AsyncEngine,
    schemas: Optional[list[sch.DBSchema]] = None,
    tables: Optional[list[sch.SQLTable]] = None,
    update_only: bool = False,
) -> sch.IndexedTables:
    try:
        if update_only:
            try:
                assert tables
            except AssertionError:
                raise Exception("Need tables provided if update_only = True")
        # Upload tables
        mng_tbls = TableManager(conn_params=conn_params, schemas=schemas)
        if not tables:
            tables = await mng_tbls.get_db_tables()
        # logger.debug("All tables found: %s", tables)
        permitted_tables = await asyncio.to_thread(mng_tbls.ingest_table_names, permitted_only=True)
        logger.debug("Permitted tables found: %s", permitted_tables)
        await asyncio.to_thread(mng_tbls.dispose_engine)
        # NOTE: This is called in a task, so it needs its own AsyncSession
        session = LocalSession(client_id=client_user.client_id, engine=sql_engine)
        db = await session.open()
        try:
            tables = await crud_table.upload_table_names(
                db=db,
                client_id=client_user.client_id,
                db_id=db_id,
                conn_id=conn_id,
                tables=tables,
                permitted_tables=permitted_tables,
                update_only=update_only,
            )
            # Create index
            await index_db_tables.create_index(tables=tables, redis_client_async=redis_client_async)
            logger.info("Database index was successful")
        except Exception as e:
            await db.rollback()
            logger.error("Upload tables error!")
            logger.error(e)
            raise e
        finally:
            await session.close()
    except AssertionError as e:
        logger.error("Error indexing database: %s", str(e))
        raise e
        # TODO: This can likely be improved with pub/sub or something else
    except Exception as e:
        logger.error("Error indexing database: %s", str(e))
        raise e
    return sch.IndexedTables(
        index_name=index_db_tables.index_name, vector_uuid=index_db_tables.vector_uuid, tables=tables
    )


async def reindex_db(
    client_user: sch.ClientUserInfo,
    conn_params: sch.SQLDBSchema,
    vector_uuid: uuid.UUID,
    db_id: int,
    db_uuid: uuid.UUID,
    conn_id: int,
    index_name: str,
    redis_client_async: RedisAsync,
    small_model_info: sch.ModelInfo,
    embedding_model_info: sch.AzureModelInfo,
    sql_engine: AsyncEngine,
) -> sch.IndexedTables:
    try:
        # Check if tables already exist in the DB
        session = LocalSession(client_id=client_user.client_id, engine=sql_engine)
        db = await session.open()
        try:
            db_tables = await crud_table.get_tables_using_db_id(db=db, db_id=db_id, get_columns=True)
            tables_base = [sch.GetSQLTable.from_orm(table) for table in db_tables]
            tables = await db_utils.process_db_tables(tables=tables_base)
            index_db_tables = DBTableIndexer(
                client_id=client_user.client_id,
                client_uuid=client_user.client_uuid,
                vector_uuid=vector_uuid,
                db_uuid=db_uuid,
                index_name=index_name,
                embedding_model_info=embedding_model_info,
            )
            if not db_tables:
                logger.info("No tables found. Creating tables and reindexing.")
                await index_db(
                    index_db_tables=index_db_tables,
                    conn_params=conn_params,
                    client_user=client_user,
                    db_id=db_id,
                    conn_id=conn_id,
                    db_uuid=db_uuid,
                    small_model_info=small_model_info,
                    redis_client_async=redis_client_async,
                    sql_engine=sql_engine,
                )
            else:
                logger.info("Started reindexing the DB")
                for table in tables:
                    table_info = TableManager.format_table_info(table=table)
                    table.table_info = table_info
                await index_db_tables.create_index(tables=tables, redis_client_async=redis_client_async)
        except Exception as e:
            logger.error(str(e))
            raise e
        finally:
            await session.close()
    except Exception as e:
        logger.error(str(e))
    return sch.IndexedTables(
        index_name=index_db_tables.index_name, vector_uuid=index_db_tables.vector_uuid, tables=tables
    )


class IndexUpdater:
    def __init__(
        self,
        connections: list[sch.SQLConnSchema],
        index_db_tables: DBTableIndexer,
        client_user: sch.ClientUserInfo,
        db_id: int,
        db_uuid: uuid.UUID,
        small_model_info: sch.ModelInfo,
    ):
        self.connections = connections
        self.index_db_tables = index_db_tables
        self.client_user = client_user
        self.db_id = db_id
        self.db_uuid = db_uuid
        self.small_model_info = small_model_info

    async def index_new_schemas(
        self, redis_client_async: RedisAsync, sql_engine: AsyncEngine, new_schemas: list[sch.DBSchema]
    ):
        logger.info("New schemas detected: %s", new_schemas)
        await self._update_index(redis_client_async=redis_client_async, sql_engine=sql_engine, new_schemas=new_schemas)

    async def index_new_tables(
        self,
        redis_client_async: RedisAsync,
        sql_engine: AsyncEngine,
        new_tables: list[sch.SQLTable],
        update_only: bool = False,
    ):
        logger.info("New tables detected: %s", new_tables)
        await self._update_index(
            redis_client_async=redis_client_async,
            sql_engine=sql_engine,
            new_tables=new_tables,
            update_only=update_only,
        )

    async def _update_index(
        self,
        redis_client_async: RedisAsync,
        sql_engine: AsyncEngine,
        new_schemas: Optional[list[sch.DBSchema]] = None,
        new_tables: Optional[list[sch.SQLTable]] = None,
        update_only: bool = False,
    ):
        """Update an existing index"""
        if not new_schemas and not new_tables:
            raise Exception("Either new schemas or new tables needs to be provided.")
        # Update connections
        for idx, connection in enumerate(self.connections, start=1):
            if idx == 1:
                # Only need to index the DB tables for the first connection
                await index_db(
                    index_db_tables=self.index_db_tables,
                    conn_params=connection.conn_params,
                    client_user=self.client_user,
                    db_id=self.db_id,
                    db_uuid=self.db_uuid,
                    conn_id=connection.conn_id,
                    small_model_info=self.small_model_info,
                    redis_client_async=redis_client_async,
                    sql_engine=sql_engine,
                    schemas=new_schemas,
                    tables=new_tables,
                    update_only=update_only,
                )
            elif not update_only:
                # Otherwise just update the connection relationships to the tables
                await crud_connection.setup_connection_assoc_table(
                    client_id=self.client_user.client_id,
                    conn_id=connection.conn_id,
                    conn_params=connection.conn_params,
                    sql_engine=sql_engine,
                    new_tables=new_tables,
                    db_id=self.db_id,
                )
