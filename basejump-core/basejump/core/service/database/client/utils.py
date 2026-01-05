import asyncio
import copy
import uuid
from asyncio import Task

from redis.asyncio import Redis as RedisAsync
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from basejump.core.database.client.index import DBTableIndexer
from basejump.core.database.crud import crud_connection
from basejump.core.database.db_connect import ConnectDB
from basejump.core.database.vector_utils import get_index_name
from basejump.core.models import schemas as sch


async def setup_vector(db: AsyncSession, client_id: int, index_db_tables: DBTableIndexer) -> int:
    vectordb_schema = sch.VectorDBSchema(
        vector_database_vendor=index_db_tables.vector_database_vendor.value,
        vector_datasource_type=index_db_tables.vector_datasource_type.value,
    )
    vectordb_schema.index_name = get_index_name(client_id=client_id)
    vector_db = await crud_connection.save_vector_store_info(
        db=db,
        client_id=client_id,
        vector_uuid=index_db_tables.vector_uuid,
        vectordb_schema=vectordb_schema,
    )
    return vector_db.vector_id


async def create_alias_name(db: AsyncSession, conn_params: sch.SQLDBSchema):
    if not conn_params.database_name_alias:
        conn_params.database_name_alias = conn_params.database_name
    alias_list = await crud_connection.get_db_aliases(db=db)
    for alias in alias_list:
        if conn_params.database_name_alias == alias.alias_name:
            alias_num_list = [
                alias.alias_number for alias in alias_list if conn_params.database_name_alias in alias.alias_name
            ]
            conn_params.database_name_alias_number = max(alias_num_list) + 1
            conn_params.database_name_alias = f"{alias.alias_name} ({conn_params.database_name_alias_number})"
            break


async def setup_connection(
    db: AsyncSession,
    client_id: int,
    conn_params: sch.SQLDBSchema,
    db_id: int,
    login_params: sch.CreateDBConn,
    sql_engine: AsyncEngine,
) -> sch.GetSQLConn:
    # Verify the connection
    conn_db = ConnectDB(conn_params=conn_params)
    await asyncio.to_thread(conn_db.verify_client_connection)
    # Create the connection
    db_login = await crud_connection.create_db_conn(
        db=db,
        db_id=db_id,
        login_params=login_params,
        client_id=client_id,
        data_source_desc=conn_params.data_source_desc,
    )
    # Add to connection association table in the background
    background_tasks = set()
    task: Task = asyncio.create_task(
        crud_connection.setup_connection_assoc_table(
            client_id=client_id,
            conn_id=copy.copy(db_login.conn_id),
            conn_params=conn_params,
            sql_engine=sql_engine,
            db_id=db_id,
        )
    )
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    db_params = await crud_connection.get_database_params_from_id(db=db, db_id=db_login.db_id)
    assert db_params
    return sch.GetSQLConn(
        conn_uuid=db_login.conn_uuid,
        db_uuid=copy.copy(db_params.db_uuid),
    )


async def setup_db(
    db: AsyncSession,
    client_user: sch.ClientUserInfo,
    conn_params: sch.SQLDBSchema,
    redis_client_async: RedisAsync,  # TODO: Looks like this can be removed
    embedding_model_info: sch.AzureModelInfo,
) -> tuple[sch.SQLConn, DBTableIndexer]:
    # Verify the connection
    conn_db = ConnectDB(conn_params=conn_params)
    await asyncio.to_thread(conn_db.verify_client_connection)
    if conn_params.schemas:
        conn_params.schemas = await conn_db.validate_schemas()
    # Create the alias name if it doesn't exist
    await create_alias_name(db=db, conn_params=conn_params)
    assert conn_params.database_name_alias
    # Save the vector db connection
    db_uuid = uuid.uuid4()
    # TODO: See if index_db_tables schema is necessary, seems like it isn't needed
    index_db_tables = DBTableIndexer(
        client_id=client_user.client_id,
        client_uuid=client_user.client_uuid,
        db_uuid=db_uuid,
        embedding_model_info=embedding_model_info,
    )
    vector_id = await setup_vector(db=db, client_id=client_user.client_id, index_db_tables=index_db_tables)
    # Save the db connection
    db_creds = await crud_connection.save_db_connection(
        db=db,
        client_id=client_user.client_id,
        db_uuid=db_uuid,
        conn_params=conn_params,
        vector_id=vector_id,
    )
    conn_uuid = str(copy.copy(db_creds.db_login.conn_uuid))
    db_id = copy.copy(db_creds.db_params.db_id)
    conn_id = copy.copy(db_creds.db_login.conn_id)
    sql_conn = sch.SQLConn(
        conn_uuid=conn_uuid,
        db_uuid=copy.copy(db_creds.db_params.db_uuid),
        database_name_alias=conn_params.database_name_alias,
        db_id=db_id,
        conn_id=conn_id,
    )
    return sql_conn, index_db_tables


async def create_database_from_existing_connection(
    db: AsyncSession,
    client_id: int,
    db_id: int,
    login_params: sch.CreateDBConn,
    sql_engine: AsyncEngine,
) -> sch.GetSQLConn:
    """Use existing database credentials to create a new connection"""

    # Get the database parameters
    database = await crud_connection.get_database_params_from_id(db=db, db_id=db_id)
    db_params = sch.DBParamsBytes.from_orm(database)
    decrypted_db_params = ConnectDB.decrypt_db(db_params.dict())

    # Get the connection parameters
    db_conn = sch.DBConnSchema.parse_obj(login_params)
    db_conn_dict = db_conn.dict()
    del db_conn_dict["schemas"]  # HACK: Improve schema management between database and connections
    conn_params = sch.SQLDBSchema(**decrypted_db_params, **db_conn_dict)

    # Set up a new connection
    return await setup_connection(
        db=db,
        client_id=client_id,
        conn_params=conn_params,
        login_params=login_params,
        db_id=db_id,
        sql_engine=sql_engine,
    )
