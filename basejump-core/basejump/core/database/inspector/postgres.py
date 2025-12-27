from typing import Optional

import sqlalchemy as sa
from basejump.core.database.inspector.base import StandardInspector


class PostgresInspector(StandardInspector):
    def __init__(self, conn: sa.Connection):
        super().__init__(conn=conn)

    def get_permitted_table_names(
        self,
        schema: Optional[str] = None,
        include_views: bool = False,
        include_materialized_views: bool = False,
        include_partitioned_tbls: bool = False,
    ):
        table_types = ["'r'"]  # for regular tables
        if include_views:
            table_types.append("'v'")  # v for views
        if include_materialized_views:
            table_types.append("'m'")  # m for materialized views
        if include_partitioned_tbls:
            table_types.append("'p'")
        table_types_str = ", ".join(table_types)
        # NOTE: Could just use the information schema to simplify this code
        schema_txt = """ AND n.nspname = :schema """ if schema else ""
        get_perm_tbl_sql = f"""SELECT
            n.nspname AS schema_name,
            c.relname AS table_name,
            r.rolname AS role_name
        FROM
            pg_catalog.pg_class c
        JOIN
            pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        JOIN
            pg_catalog.pg_roles r ON r.rolname \
NOT IN ('pg_read_all_data', 'pg_write_all_data', 'pg_monitor', 'pg_signal_backend')
        WHERE
            c.relkind IN ({table_types_str})  -- 'r' for regular tables, 'v' for views, 'p' for partitions
            and r.rolname = current_user
            AND has_table_privilege(r.rolname, c.oid, 'SELECT')
            AND c.relispartition = false  -- Exclude child partitions
            {schema_txt}
        ORDER BY
            schema_name, table_name, role_name
        """
        schema_dict = {"schema": schema} if schema else {}
        result = self.conn.execute(sa.text(get_perm_tbl_sql), schema_dict)
        return [row.table_name for row in result.fetchall()]

    def get_permitted_schema_names(self) -> list[str]:
        get_schema_nm_sql = """
        SELECT
            n.nspname as "schema"
        FROM pg_catalog.pg_namespace n
        WHERE 1=1
            AND n.nspname !~ '^pg_'
            AND has_schema_privilege(current_user, n.nspname, 'USAGE');"""
        result = self.conn.execute(sa.text(get_schema_nm_sql))
        return [row.schema for row in result.fetchall()]
