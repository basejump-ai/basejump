"""CRUD functions for the table module + for retrieving information related to relational tables"""

import uuid
from typing import Sequence

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.crud import crud_utils
from basejump.core.database.db_connect import TableManager
from basejump.core.models import constants
from basejump.core.models import schemas as sch
from basejump.core.models.models import (
    ConnTableAssociation,
    DBConn,
    DBTableColumns,
    DBTables,
)
from sqlalchemy import Row, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import load_only, selectinload

logger = set_logging(handler_option="stream", name=__name__)


async def get_columns_by_name(
    db: AsyncSession, columns: list[sch.DBColumn], conn_id: int, schemas: list[sch.DBSchema]
) -> list[sch.DBColumn]:
    column_names = [column.column_name.lower() for column in columns]
    logger.debug(f"Column names: {column_names}")
    stmt = (
        select(DBTableColumns, DBTables)
        .join(DBTables)
        .join(ConnTableAssociation)
        .where(
            ConnTableAssociation.conn_id == conn_id,
            func.lower(DBTableColumns.column_name).in_(column_names),
        )
        .options(
            load_only(
                DBTableColumns.column_name,
                DBTableColumns.column_type,
                DBTableColumns.distinct_values,
            )
        )
        .options(load_only(DBTables.table_name))
    )
    result = await db.execute(stmt)
    result_columns = result.all()
    final_columns = []
    for result_col, result_tbl in result_columns:
        # Render the table
        rendered_tbl = await TableManager.arender_query_jinja(jinja_str=result_tbl.table_name, schemas=schemas)
        for column in columns:
            split_table = rendered_tbl.split(".")
            if len(split_table) == 2:
                schema_name = split_table[0]
                table_name = split_table[1]
            else:
                schema_name = None
                table_name = split_table[0]
            if (
                column.table_name.lower() == table_name.lower()
                and result_col.column_name.lower() == column.column_name.lower()
                and (column.schema_name.lower() if column.schema_name else None)
                == (schema_name.lower() if schema_name else None)
            ):
                db_col = sch.DBColumn(
                    column_name=column.column_name,
                    table_name=column.table_name,
                    schema_name=column.schema_name,
                    filters=result_col.distinct_values if result_col.distinct_values else [],
                    column_type=result_col.column_type,
                )
                final_columns.append(db_col)
    return final_columns


async def get_tables_using_db_id(db: AsyncSession, db_id: int, get_columns: bool = False) -> list[DBTables]:
    if get_columns:
        stmt = select(DBTables).filter_by(db_id=db_id).options(selectinload(DBTables.columns))
    else:
        select(DBTables).filter_by(db_id=db_id)
    table = await db.execute(stmt)
    return list(table.scalars().all())


async def get_tables_from_uuid(
    db: AsyncSession, tbl_uuids: list[uuid.UUID], include_cols: bool = False
) -> list[DBTables]:
    if include_cols:
        stmt = select(DBTables).filter(DBTables.tbl_uuid.in_(tbl_uuids)).options(selectinload(DBTables.columns))
    else:
        stmt = select(DBTables).filter(DBTables.tbl_uuid.in_(tbl_uuids))
    table = await db.execute(stmt)
    return list(table.scalars().all())


async def get_conn_tables(db: AsyncSession, conn_id: int) -> Sequence[DBTables] | None:
    """Get the permitted tables for a specific connection"""
    stmt = (
        select(DBConn)
        .filter_by(conn_id=conn_id)
        .options(selectinload(DBConn.tables_assoc).joinedload(ConnTableAssociation.tables))
    )
    result_base = await db.execute(stmt)
    result = result_base.scalar_one_or_none()
    tables = result.tables if result else []
    return_tables = [table for table in tables if not table.ignore]
    return return_tables


async def get_tables_from_nms(db: AsyncSession, table_names: list[str], db_id: int) -> list[DBTables]:
    stmt = select(DBTables).filter(DBTables.table_name.in_(table_names), DBTables.db_id == db_id)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def upload_table_names(
    db: AsyncSession,
    client_id: int,
    db_id: int,
    conn_id: int,
    tables: list[sch.SQLTable],
    permitted_tables: list[sch.SQLTable],
    update_only: bool = False,
) -> list[sch.SQLTable]:
    """Uploads the names of the tables in the client database
    to the client_tables table in the basejump database"""
    # Update the table names
    # TODO: Raise this error into an HTTP Exception
    # logger.debug("Uploading tables: %s", tables)
    permitted_tables_dict = {table.full_table_name.lower(): table for table in permitted_tables}
    try:
        assert tables
    except AssertionError:
        logger.error(constants.NO_TABLES)
        raise AssertionError(constants.NO_TABLES)

    found_permitted_table = False
    for table in tables:
        logger.debug("Table name: %s", table.full_table_name)
        if update_only:
            retrieved_tables = await get_tables_from_nms(db=db, table_names=[table.full_table_name], db_id=db_id)
            retrieved_table = retrieved_tables[0]
            tbl_id = retrieved_table.tbl_id
        else:
            tbl_id = await crud_utils.get_next_val(db=db, full_table_nm=str(DBTables.__table__), column_nm="tbl_id")
            sql_table = DBTables(
                tbl_id=tbl_id,
                tbl_uuid=uuid.uuid4(),
                db_id=db_id,
                client_id=client_id,
                table_name=table.full_table_name,
                context=table.context_str,
            )
            db.add(sql_table)
            table.tbl_uuid = sql_table.tbl_uuid
            if permitted_tables_dict.get(table.full_table_name.lower()):
                found_permitted_table = True
                conn_table = ConnTableAssociation(client_id=client_id, conn_id=conn_id, tbl_id=tbl_id)
                db.add(conn_table)
        for column in table.columns:
            if update_only and not column.new:
                continue
        for column in table.columns:
            table_cols = DBTableColumns(
                client_id=client_id,
                tbl_id=tbl_id,
                column_name=column.column_name,
                column_type=column.column_type,
                description=column.description,
                foreign_key_column_name=column.foreign_key_column_name,
                foreign_key_table_name=column.foreign_key_table_name,
                quoted=column.quoted,
            )
            db.add(table_cols)
    await db.commit()
    try:
        assert found_permitted_table
    except AssertionError:
        logger.warning(constants.NO_PERMITTED_TABLES)
    return tables


async def get_all_tables(db: AsyncSession) -> list[Row]:
    stmt = select(DBTables.table_name, DBTables.ignore)
    result = await db.execute(stmt)
    return list(result.all())


async def get_all_columns(db: AsyncSession, conn_id: int) -> list[Row]:
    stmt = (
        select(DBTables.table_name, DBTableColumns.column_name, DBTableColumns.ignore, DBTableColumns.quoted)
        .join(DBTables)
        .join(ConnTableAssociation, DBTables.tbl_id == ConnTableAssociation.tbl_id)
    ).filter(ConnTableAssociation.conn_id == conn_id)
    result = await db.execute(stmt)
    return list(result.all())
