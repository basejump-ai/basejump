import pytest

from basejump.core.database.crud import crud_table


@pytest.mark.table
async def test_get_tables(db_session):
    """Test getting tables"""
    # Get all table columns
    await crud_table.get_all_columns(db=db_session.db, conn_id=db_session.conn_id)

    # Get all tables
    await crud_table.get_all_tables(db=db_session.db)

    # Get all tables for a connection
    await crud_table.get_conn_tables(db=db_session.db, conn_id=db_session.conn_id)

    # Get all tables for a database
    tables = await crud_table.get_tables_using_db_id(
        db=db_session.db, db_id=db_session.db_id, get_columns=True
    )

    # Get table information for a given set of tables
    await crud_table.get_tables_from_uuid(
        db=db_session.db, tbl_uuids=[table.tbl_uuid for table in tables]
    )
