"""
Functions to interact with the database for tables related to the connection.py endpoint module
and connection-related tables
"""

import asyncio
import copy
import json
import uuid
from typing import Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.orm import Session, joinedload, selectinload

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.crud import crud_table, crud_utils
from basejump.core.database.db_connect import ConnectDB, LocalSession, TableManager
from basejump.core.models import errors, models
from basejump.core.models import schemas as sch

logger = set_logging(handler_option="stream", name=__name__)


async def get_client_dbs(db: AsyncSession, client_id: int) -> list[models.DBParams]:
    stmt = select(models.DBParams).filter_by(client_id=client_id)
    results = await db.execute(stmt)
    return list(results.scalars().all())


async def get_database_params(
    db: AsyncSession, db_uuid: uuid.UUID, get_tables: bool = False
) -> Optional[models.DBParams]:
    if get_tables:
        stmt = (
            select(models.DBParams)
            .filter_by(db_uuid=db_uuid)
            .options(selectinload(models.DBParams.tables).selectinload(models.DBTables.columns))
            .options(selectinload(models.DBParams.connections))
        )
    else:
        stmt = select(models.DBParams).filter_by(db_uuid=db_uuid)
    database = await db.execute(stmt)
    return database.scalar_one_or_none()


async def get_database_params_from_id(db: AsyncSession, db_id: int) -> Optional[models.DBParams]:
    database = await db.execute(select(models.DBParams).filter_by(db_id=db_id))
    return database.scalar_one_or_none()


async def get_db_conn(db: AsyncSession, conn_uuid: uuid.UUID, get_tables: bool = False) -> Optional[models.DBConn]:
    if get_tables:
        stmt = (
            select(models.DBConn)
            .filter_by(conn_uuid=conn_uuid)
            .options(
                selectinload(models.DBConn.tables_assoc)
                .joinedload(models.ConnTableAssociation.tables)
                .selectinload(models.DBTables.columns)
            )
        )
    else:
        stmt = select(models.DBConn).filter_by(conn_uuid=conn_uuid)
    db_conn_base = await db.execute(stmt)
    return db_conn_base.scalar_one_or_none()


async def get_db_conns(db: AsyncSession, db_id: int) -> Sequence[models.DBConn]:
    database = await db.execute(select(models.DBConn).filter_by(db_id=db_id))

    return database.scalars().all()


async def get_db_conn_from_id(db: AsyncSession, conn_id: int) -> Optional[models.DBConn]:
    database = await db.execute(select(models.DBConn).filter_by(conn_id=conn_id))
    return database.scalar_one_or_none()


async def create_db_conn(
    db: AsyncSession,
    db_id: int,
    login_params: sch.CreateDBConn,
    client_id: int,
    data_source_desc: str,
) -> models.DBConn:
    # Encrypt the username and password
    dict_to_encrypt = {"username": login_params.username, "password": login_params.password}
    encrypted_dict = ConnectDB.encrypt_db(dict_to_encrypt=dict_to_encrypt)
    username = encrypted_dict["username"]
    password = encrypted_dict["password"]
    db_login = models.DBConn(
        db_id=db_id,
        username=username,
        password=password,
        schemas=[schema.dict() for schema in login_params.schemas] if login_params.schemas else [],
        client_id=client_id,
        data_source_desc=data_source_desc,
    )
    db.add(db_login)
    await db.commit()
    await db.refresh(db_login)
    return db_login


async def save_db_connection(
    db: AsyncSession, conn_params: sch.SQLDBSchema, client_id: int, db_uuid: uuid.UUID, vector_id: int
) -> models.DBCredentials:
    """Save the connection to the database"""
    # Add the fields to SQLDB
    conn_db = ConnectDB(conn_params=conn_params)
    db_params = conn_db.conn_params_bytes.dict()
    database = models.DBParams(
        **db_params,
        db_uuid=db_uuid,
        vector_id=vector_id,
        client_id=client_id,
        database_name_alias_number=conn_params.database_name_alias_number,
    )
    # Save the database connection
    db.add(database)
    await db.commit()
    await db.refresh(database)
    database = copy.deepcopy(database)

    # Save the connection
    login_params = sch.CreateDBConn(
        username=conn_params.username,
        password=conn_params.password,
        schemas=conn_params.schemas,
        data_source_desc=conn_params.data_source_desc,
    )
    db_login = await create_db_conn(
        db=db,
        db_id=database.db_id,
        login_params=login_params,
        client_id=client_id,
        data_source_desc=conn_params.data_source_desc,
    )

    return models.DBCredentials(db_login=db_login, db_params=database)


async def get_conndb_from_connection(db_params: sch.DBParamsSchema, connection: models.DBConn) -> ConnectDB:
    db_params_dict = db_params.dict()
    db_params_dict["username"] = ConnectDB.decrypt_db({"username": connection.username})["username"]
    db_params_dict["password"] = ConnectDB.decrypt_db({"password": connection.password})["password"]
    # HACK
    if isinstance(connection.schemas, str):
        connection_schemas = json.loads(connection.schemas)
    else:
        connection_schemas = connection.schemas
    db_params_dict["schemas"] = connection_schemas  # Use connection schemas and not DB schemas
    conn_params_all = sch.SQLDBSchema(**db_params_dict, data_source_desc="")
    return ConnectDB(conn_params=conn_params_all)


async def verify_connection(conn_db: ConnectDB) -> None:
    try:
        await asyncio.to_thread(conn_db.verify_client_connection)
        if conn_db.conn_params.schemas:
            await conn_db.validate_schemas()
    except (
        errors.InvalidSchemas,
        errors.InvalidJinjaBraceCount,
        errors.InvalidJinjaContent,
        errors.InvalidJinjaStartingBrace,
        errors.InvalidJinjaEndingBrace,
    ) as e:
        logger.error("Invalid schemas %s", str(e))
        raise e
    except Exception as e:
        logger.error("DB connection error %s", str(e))
        raise errors.ConnectDBError


async def update_connection_schemas(
    db_params: sch.DBParamsSchema, schema_maps: list[sch.SchemaMap], connections: list[models.DBConn]
) -> None:
    """Update existing connection schemas
    Find a matching schemas in login schemas and then update it with a new name if there is one
    For every login, compare all db schemas to login schemas"""
    updated_schemas = []
    for connection in connections:
        if connection.schemas:
            connection_schemas = connection.schemas
            if isinstance(connection_schemas, str):
                connection_schemas = json.loads(copy.copy(connection_schemas))
            for connection_schema in connection_schemas:
                logger.debug("Here is the connection schema %s", connection_schema)
                for schema_map in schema_maps:
                    if schema_map.old_schema == connection_schema["schema_nm"]:
                        connection_schema["schema_nm"] = schema_map.new_schema
                        connection_schema["jinja_values"] = None  # More secure to set all jinja values to None
                        logger.info(
                            f"Prior schema '{schema_map.old_schema}' updated to new schema '{schema_map.new_schema}'"
                        )

                updated_schemas.append(connection_schema)
        logger.debug("Here is the connection type:", str(type(connection)))
        connection.schemas = updated_schemas


async def get_connection(db: AsyncSession, conn_uuid: uuid.UUID) -> Optional[models.Connection]:
    conn = await db.execute(select(models.Connection).filter_by(conn_uuid=conn_uuid))
    return conn.scalar_one_or_none()


async def get_vector_connection(db: AsyncSession, vector_uuid: uuid.UUID) -> models.DBVector:
    stmt = select(models.DBVector).filter_by(vector_uuid=vector_uuid)
    conn = await db.execute(stmt)
    return conn.scalar_one()


async def get_vector_connection_from_id(db: AsyncSession, vector_id: int) -> models.DBVector:
    conn = await db.execute(select(models.DBVector).filter_by(vector_id=vector_id))
    return conn.scalar_one()


async def get_connections(db: AsyncSession, team_id: int, user_id: int) -> list[models.Connection]:
    """Returns a list of db models comprised of various connection types"""
    # NOTE: Do not remove UserTeamAssociation - it is what prevents unauthorized access to team information and limits
    # access to only those teams a user should have access to
    stmt = (
        select(models.DBConn)
        .join(models.ConnTeamAssociation)
        .join(models.UserTeamAssociation, models.ConnTeamAssociation.team_id == models.UserTeamAssociation.team_id)
        .filter(models.UserTeamAssociation.team_id == team_id, models.UserTeamAssociation.user_id == user_id)
        .options(joinedload(models.DBConn.database_params))
    )
    connections_base = await db.execute(stmt)
    connections = connections_base.scalars().all()
    if not connections:
        logger.warning("No connections found for this client")
    return list(connections)


async def get_user_connections(db: AsyncSession, user_uuid: uuid.UUID) -> list:
    stmt = (
        select(models.ConnTeamAssociation.conn_id)
        .select_from(models.User)
        .join(models.UserTeamAssociation, models.User.user_id == models.UserTeamAssociation.user_id)
        .join(models.ConnTeamAssociation, models.UserTeamAssociation.team_id == models.ConnTeamAssociation.team_id)
        .filter(models.User.user_uuid == user_uuid)
    )
    results = await db.execute(stmt)
    return list(results.scalars().all())


# TODO Update this to accepting a schema instead
async def save_vector_store_info(
    db: AsyncSession,
    client_id: int,
    vector_uuid: uuid.UUID,
    vectordb_schema: sch.VectorDBSchema,
    vector_metadata: Optional[list[dict]] = None,
):
    """Save the vector store information to the database"""
    metadata_json = json.dumps(vector_metadata)
    vector_db = models.DBVector(
        client_id=client_id,
        vector_uuid=vector_uuid,
        vector_metadata=metadata_json,
        vector_database_vendor=vectordb_schema.vector_database_vendor.value,
        vector_datasource_type=vectordb_schema.vector_datasource_type.value,
        index_name=vectordb_schema.index_name,
    )
    db.add(vector_db)
    await db.commit()
    await db.refresh(vector_db)

    return vector_db


async def get_db_aliases(db: AsyncSession) -> list[sch.Alias]:
    stmt = select(models.DBParams.database_name_alias, models.DBParams.database_name_alias_number)
    result = await db.execute(stmt)
    aliases = result.all()
    alias_list = []
    for alias in aliases:
        alias_list.append(
            sch.Alias(
                alias_name=ConnectDB.decrypt_db(dict_to_decrypt={"database_name_alias": alias.database_name_alias})[
                    "database_name_alias"
                ],
                alias_number=alias.database_name_alias_number,
            )
        )
    return alias_list


async def get_demo_tbl_info(db: AsyncSession, vector_id: int):
    stmt = select(models.DemoVectorAssociation).filter_by(vector_id=vector_id)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def setup_connection_assoc_table(
    client_id: int,
    conn_id: int,
    conn_params: sch.SQLDBSchema,
    sql_engine: AsyncEngine,
    db_id: int,
    new_tables: Optional[list[sch.SQLTable]] = None,
):
    # TODO: When there is a new schema, this should only add tables from the new schema
    # This needs to be checked since it seems this code will add code from prior schemas as well
    # unless the conn_params only includes the new schema
    mng_tbls = TableManager(conn_params=conn_params)
    permitted_tables = await asyncio.to_thread(mng_tbls.ingest_table_names, permitted_only=True)
    await asyncio.to_thread(mng_tbls.dispose_engine)
    permitted_table_names = [table.full_table_name for table in permitted_tables]
    if new_tables:
        new_table_names = [table.full_table_name for table in new_tables]
        table_names = [table_name for table_name in permitted_table_names if table_name in new_table_names]
    else:
        table_names = permitted_table_names
    session = LocalSession(client_id=client_id, engine=sql_engine)
    db = await session.open()
    try:
        # TODO: Assess if this needs to be filtered by database
        tables = await crud_table.get_tables_from_nms(db=db, table_names=table_names, db_id=db_id)
        for table in tables:
            conn_table = models.ConnTableAssociation(client_id=client_id, conn_id=conn_id, tbl_id=table.tbl_id)
            db.add(conn_table)
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error("Create connection association tables error: %s", str(e))
        raise e
    finally:
        await session.close()


async def update_vector_connection(db: AsyncSession, update_vector: sch.UpdateVector, vector_uuid: uuid.UUID) -> None:
    logger.debug("Here is the vector_uuid: %s", vector_uuid)
    vector_conn = await get_vector_connection(db=db, vector_uuid=vector_uuid)
    crud_utils.update_model(schema=update_vector, db_model=vector_conn)
    await db.commit()


def get_client_active_storage_conn_stmt(client_id: int):
    return (
        select(models.ClientStorageConnection)
        .where(models.ClientStorageConnection.client_id == client_id)
        .where(models.ClientStorageConnection.active.is_(True))
    )


async def get_client_active_storage_conn(db: AsyncSession, client_id: int) -> Optional[models.ClientStorageConnection]:
    stmt = get_client_active_storage_conn_stmt(client_id=client_id)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


def get_client_active_storage_conn_sync(db: Session, client_id: int) -> Optional[models.ClientStorageConnection]:
    stmt = get_client_active_storage_conn_stmt(client_id=client_id)
    result = db.execute(stmt)
    return result.scalar_one_or_none()


async def get_vector_from_connection(db: AsyncSession, db_uuid: uuid.UUID) -> Optional[models.DBVector]:
    stmt = select(models.DBVector).join(models.DBParams).filter(models.DBParams.db_uuid == db_uuid)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()
