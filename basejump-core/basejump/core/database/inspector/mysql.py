from typing import Optional

import sqlalchemy as sa
from basejump.core.database.inspector.base import StandardInspector


class MySQLInspector(StandardInspector):
    def __init__(self, conn: sa.Connection):
        super().__init__(conn=conn)

    def get_permitted_table_names(
        self,
        schema: Optional[str] = None,
        include_views: bool = False,
        include_materialized_views: bool = False,
        include_partitioned_tbls: bool = False,
    ):
        schema_txt = """ AND t.TABLE_SCHEMA = :schema """ if schema else ""
        table_types = ["'BASE TABLE'"]
        if include_views:
            table_types.append("'VIEW'")
        if include_materialized_views:
            pass  # materialized views not implemented in mysql
        table_types_str = ", ".join(table_types)
        partition_tbl_str = "" if include_partitioned_tbls else " AND p.PARTITION_NAME IS NULL "
        get_perm_tbl_sql = f"""
SELECT
    t.TABLE_SCHEMA AS schema_name,
    t.TABLE_NAME AS table_name
FROM
    information_schema.tables t
LEFT JOIN
    information_schema.partitions p
    ON t.TABLE_SCHEMA = p.TABLE_SCHEMA
    AND t.TABLE_NAME = p.TABLE_NAME
WHERE
    t.TABLE_TYPE IN ({table_types_str})  -- Include both tables and views
    AND t.TABLE_SCHEMA NOT IN ('information_schema', 'performance_schema', 'mysql', 'sys')  -- Exclude system schemas
    {partition_tbl_str}
    {schema_txt}
ORDER BY
    schema_name, table_name;
        """
        schema_dict = {"schema": schema} if schema else {}
        result = self.conn.execute(sa.text(get_perm_tbl_sql), schema_dict)
        return [row.table_name for row in result.fetchall()]

    def get_permitted_schema_names(self) -> list[str]:
        get_perm_sch_sql = """SELECT DISTINCT
            TABLE_SCHEMA AS schema_name
        FROM
            information_schema.tables
        WHERE
            TABLE_TYPE IN ('BASE TABLE', 'VIEW')  -- Include both tables and views
            AND TABLE_SCHEMA NOT IN ('information_schema', 'performance_schema', 'mysql', 'sys')
        ;"""
        result = self.conn.execute(sa.text(get_perm_sch_sql))
        return [row.schema_name for row in result.fetchall()]
