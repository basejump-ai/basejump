import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine import Engine

from basejump.core.common.config.logconfig import set_logging
from basejump.core.common.config.settings import settings
from basejump.core.database.connector import PostgresConnector
from basejump.core.database.db_utils import get_table_schemas
from basejump.core.models import enums
from basejump.core.models import schemas as sch

logger = set_logging(handler_option="stream", name=__name__)

# Set up default database
description = "Useful for finding information about clients, teams, and users."
conn_params = sch.SQLDBSchema(
    database_type=enums.DatabaseType.POSTGRES,
    drivername=enums.DBAsyncDriverName.POSTGRES,
    # NOTE: These settings should be defined in an .env file with the BASEJUMP_ prefix
    username=settings.db_user,
    password=settings.db_password.get_secret_value(),
    host=settings.db_host,
    port=settings.db_port,
    database_name=settings.db_name,
    query={},
    schemas=[sch.DBSchema(schema_nm="account")],
    database_desc=description,
    data_source_desc=description,
    include_default_schema=False,
    ssl=settings.ssl,
)
conn_params_noasync = sch.SQLDBSchema(**conn_params.dict())
conn_params_noasync.drivername = enums.DBDriverName.POSTGRES
postgres_db = PostgresConnector(conn_params=conn_params_noasync)


def gen_client_id_suffixes(engine: Engine, reverse: bool = False, local_env: bool = True) -> list:
    """Used for generation of client suffixes in alembic

    Notes
    -----
    Don't forget to dispose the engine after using!
    """
    if local_env:
        # Using a shortened list for convenience when running locally
        return ["", "0"] if not reverse else ["0", ""]
    schemas = get_table_schemas()
    schemas_str = "|".join(schemas)
    with engine.connect() as conn:
        # TODO: All shared schemas need to be excluded from this since they only have dummy
        # tables that don't get dropped in a cascade. Use a constant for shared schemas here.
        query_str = f"""select distinct table_schema from information_schema.tables where \
            table_schema ~ '{schemas_str}' \
            and table_schema not like 'account%'"""
        result = conn.execute(sa.text(query_str))
        db_schemas = result.scalars().all()
    my_set = {
        db_schema.split(schema)[1]
        for schema in schemas
        for db_schema in db_schemas
        if len(db_schema.split(schema)) > 1
    }
    with engine.connect() as conn:
        query_str = """select client_id from account.client"""
        result = conn.execute(sa.text(query_str))
        client_ids = result.scalars().all()
    my_client_id_set = set([str(client_id) for client_id in client_ids])
    final_set = my_client_id_set & my_set
    return sorted(list(final_set), reverse=reverse)


def refresh_views(tables: list, local_env: bool = True):
    """This is needed when adding/deleting columns since views only include columns based on when they were created."""
    basejump_engine_noasync = postgres_db.connect_db()
    suffixes = gen_client_id_suffixes(engine=basejump_engine_noasync, reverse=False, local_env=local_env)
    # HACK: Using print since logconfig propagate is set to False
    # TODO: Refactor the logconfig so user handles the logging instead of the library
    print("Running suffixes...", suffixes)
    print("Dropping views...")
    for suffix in suffixes:
        if not suffix or suffix == "0":
            continue
        for table in tables:
            table_name = f"{table.split('.')[0] + suffix}.{table.split('.')[1]}"
            # Drop the view
            # HACK: Using print since logconfig propagate is set to False
            print("Dropping view", table_name)
            op.execute(f"DROP VIEW IF EXISTS {table_name}")
    # HACK: Using print since logconfig propagate is set to False
    print("Adding views...")
    for suffix in suffixes:
        if not suffix or suffix == "0":
            continue
        # Create views
        for table in tables:
            table_name_w_suffix = f"{table.split('.')[0] + suffix}.{table.split('.')[1]}"
            op.execute(
                f"""\
        CREATE VIEW {table_name_w_suffix} WITH(security_invoker=TRUE) AS \
        SELECT * FROM {table} \
        WHERE client_id = {suffix}"""
            )
    basejump_engine_noasync.dispose()
