from typing import Optional

import sqlalchemy as sa
from basejump.core.database.inspector.base import StandardInspector


class MSSQLServerInspector(StandardInspector):
    def __init__(self, conn: sa.Connection):
        super().__init__(conn=conn)

    # TODO: Need to excluded tables that are part of a partition: https://github.com/Basejump-AI/Basejump/issues/1233
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
            pass  # materialized views not implemented in sql_server
        table_types_str = ", ".join(table_types)
        partition_tbl_str = "" if include_partitioned_tbls else " AND p.partition_number = 1 "
        get_perm_tbl_sql = f"""SELECT DISTINCT
                t.TABLE_SCHEMA AS schema_name,
                t.TABLE_NAME AS table_name
            FROM
                information_schema.tables t
            LEFT JOIN sys.indexes i ON OBJECT_ID(t.TABLE_SCHEMA + '.' + t.TABLE_NAME) = i.object_id AND i.index_id <= 1
            LEFT JOIN sys.partitions p ON i.object_id = p.object_id AND i.index_id = p.index_id
            WHERE 1=1
                {schema_txt}
                AND t.TABLE_TYPE IN ({table_types_str})
                {partition_tbl_str}
                """
        schema_dict = {"schema": schema} if schema else {}
        result = self.conn.execute(sa.text(get_perm_tbl_sql), schema_dict)
        return [row.table_name for row in result.fetchall()]

    def get_permitted_schema_names(self) -> list[str]:
        get_perm_sch_sql = """
        SELECT DISTINCT
            TABLE_SCHEMA AS schema_name
        FROM
            information_schema.tables
        WHERE
            TABLE_TYPE IN ('BASE TABLE', 'VIEW')  -- Include both tables and views
            AND TABLE_SCHEMA NOT IN ('sys', 'information_schema')
        ;"""
        result = self.conn.execute(sa.text(get_perm_sch_sql))
        return [row.schema_name for row in result.fetchall()]
