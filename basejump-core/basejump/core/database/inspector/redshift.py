from typing import Optional

import sqlalchemy as sa
from basejump.core.database.db_utils import process_foreign_key_definition
from basejump.core.database.inspector.postgres import PostgresInspector


class RedshiftInspector(PostgresInspector):
    def __init__(self, conn: sa.Connection):
        self.conn = conn

    @classmethod
    def inspect(cls, conn: sa.Connection):
        return cls(conn=conn)

    def get_table_names(
        self,
        schema: Optional[str] = None,
        include_views: bool = False,
        include_materialized_views: bool = False,
        include_partitioned_tbls: bool = False,
    ) -> list[str]:
        schema_txt = """ AND schema = :schema """ if schema else ""
        table_types = ["'r'"]  # for regular tables
        if include_views:
            table_types.append("'v'")  # v for views
        if include_materialized_views:
            table_types.append("'m'")  # m for materialized views
        if include_partitioned_tbls:
            table_types.append("'p'")
        table_types_str = ", ".join(table_types)
        get_table_name_sql = (
            f"""
        SELECT
            c.relkind,
            n.oid as "schema_oid",
            n.nspname as "schema",
            c.oid as "rel_oid",
            c.relname,
            CASE c.reldiststyle
            WHEN 0 THEN 'EVEN' WHEN 1 THEN 'KEY' WHEN 8 THEN 'ALL' END
            AS "diststyle",
            c.relowner AS "owner_id",
            u.usename AS "owner_name",
            TRIM(TRAILING ';' FROM pg_catalog.pg_get_viewdef(c.oid, true))
            AS "view_definition",
            pg_catalog.array_to_string(c.relacl, '
        ') AS "privileges"
        FROM pg_catalog.pg_class c
                LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                JOIN pg_catalog.pg_user u ON u.usesysid = c.relowner
        WHERE c.relkind IN ({table_types_str})
            AND n.nspname !~ '^pg_'"""
            + schema_txt
            + """
        UNION
        SELECT
            'r' AS "relkind",
            s.esoid AS "schema_oid",
            s.schemaname AS "schema",
            null AS "rel_oid",
            t.tablename AS "relname",
            null AS "diststyle",
            s.esowner AS "owner_id",
            u.usename AS "owner_name",
            null AS "view_definition",
            null AS "privileges"
        FROM
            svv_external_tables t
            JOIN svv_external_schemas s ON s.schemaname = t.schemaname
            JOIN pg_catalog.pg_user u ON u.usesysid = s.esowner
        where 1"""
            + schema_txt
            + """ ORDER BY "relkind", "schema_oid", "schema" """
        )
        schema_dict = {"schema": schema} if schema else {}
        result = self.conn.execute(sa.text(get_table_name_sql), schema_dict)
        return [row.relname for row in result.fetchall()]

    def get_permitted_table_names(
        self,
        schema: Optional[str] = None,
        include_views: bool = False,
        include_materialized_views: bool = False,
        include_partitioned_tbls: bool = False,
    ) -> list[str]:
        schema_txt = """ AND schema = :schema """ if schema else ""
        table_types = ["'r'"]  # for regular tables
        if include_views:
            table_types.append("'v'")  # v for views
        if include_materialized_views:
            table_types.append("'m'")  # m for materialized views
        if include_partitioned_tbls:
            table_types.append("'p'")
        table_types_str = ", ".join(table_types)
        get_perm_table_name_sql = f"""SELECT
                n.nspname as "schema",
                c.relname AS "relname"
            FROM pg_catalog.pg_class c
            LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
            JOIN pg_catalog.pg_user u ON u.usesysid = c.relowner
            WHERE c.relkind IN ({table_types_str})
                AND n.nspname !~ '^pg_'
                AND has_table_privilege(current_user, n.nspname || '.' || c.relname, 'SELECT')
                {schema_txt}
            UNION
            SELECT
                s.schemaname AS "schema",
                t.tablename AS "relname"
            FROM svv_external_tables t
            JOIN svv_external_schemas s ON s.schemaname = t.schemaname
            WHERE has_table_privilege(current_user, s.schemaname || '.' || t.tablename, 'SELECT')
                {schema_txt}
                """
        schema_dict = {"schema": schema} if schema else {}
        result = self.conn.execute(sa.text(get_perm_table_name_sql), schema_dict)
        return [row.relname for row in result.fetchall()]

    def get_table_comment(self, table_name: str, schema: Optional[str] = None) -> dict:
        schema_txt = " AND nsp.nspname = :schema " if schema else ""
        get_oid_sql = (
            """
        SELECT
            tbl.oid

        FROM pg_namespace nsp

        INNER JOIN pg_class  tbl
            ON nsp.oid = tbl.relnamespace """
            + schema_txt
            + """ WHERE tbl.relname = :table_name;
        """
        )
        schema_dict = {"schema": schema}
        table_dict = {"table_name": table_name}
        keys = table_dict | schema_dict if schema else table_dict
        result = self.conn.execute(sa.text(get_oid_sql), keys)
        table_oid = result.scalar_one()
        get_table_comment_sql = """
        SELECT
            pgd.description as table_comment
        FROM
            pg_catalog.pg_description pgd
        WHERE
            pgd.objsubid = 0 AND
            pgd.objoid = :table_oid
        """
        result = self.conn.execute(sa.text(get_table_comment_sql), {"table_oid": table_oid})
        # TODO: Sometimes can return multiple rows, this only supports one
        table_comment = result.scalar_one_or_none()
        return {"text": table_comment}

    def get_columns(self, table_name: str, schema: Optional[str] = None) -> list[dict]:
        schema_txt = """ AND schema = :schema """ if schema else ""
        get_columns_sql = (
            """SELECT
            n.nspname as "schema",
            c.relname as "table_name",
            att.attname as "name",
            format_encoding(att.attencodingtype::integer) as "encode",
            format_type(att.atttypid, att.atttypmod) as "type",
            att.attisdistkey as "distkey",
            att.attsortkeyord as "sortkey",
            att.attnotnull as "notnull",
            pg_catalog.col_description(att.attrelid, att.attnum)
            as "comment",
            adsrc,
            attnum,
            pg_catalog.format_type(att.atttypid, att.atttypmod),
            pg_catalog.pg_get_expr(ad.adbin, ad.adrelid) AS DEFAULT,
            n.oid as "schema_oid",
            c.oid as "table_oid"
        FROM pg_catalog.pg_class c
        LEFT JOIN pg_catalog.pg_namespace n
            ON n.oid = c.relnamespace
        JOIN pg_catalog.pg_attribute att
            ON att.attrelid = c.oid
        LEFT JOIN pg_catalog.pg_attrdef ad
            ON (att.attrelid, att.attnum) = (ad.adrelid, ad.adnum)
        WHERE n.nspname !~ '^pg_'
            AND att.attnum > 0
            AND NOT att.attisdropped """
            + schema_txt
            + """ AND table_name = :table_name
        UNION
        SELECT
            view_schema as "schema",
            view_name as "table_name",
            col_name as "name",
            null as "encode",
            col_type as "type",
            null as "distkey",
            0 as "sortkey",
            null as "notnull",
            null as "comment",
            null as "adsrc",
            null as "attnum",
            col_type as "format_type",
            null as "default",
            null as "schema_oid",
            null as "table_oid"
        FROM pg_get_late_binding_view_cols() cols(
            view_schema name,
            view_name name,
            col_name name,
            col_type varchar,
            col_num int)
        WHERE 1 """
            + schema_txt
            + """ AND table_name = :table_name
        UNION
        SELECT c.schemaname AS "schema",
            c.tablename AS "table_name",
            c.columnname AS "name",
            null AS "encode",
            -- Spectrum represents data types differently.
            -- Standardize, so we can infer types.
            CASE
                WHEN c.external_type = 'int' THEN 'integer'
                WHEN c.external_type = 'float' THEN 'real'
                WHEN c.external_type = 'double' THEN 'double precision'
                WHEN c.external_type = 'timestamp'
                THEN 'timestamp without time zone'
                WHEN c.external_type ilike 'varchar%%'
                THEN replace(c.external_type, 'varchar', 'character varying')
                WHEN c.external_type ilike 'decimal%%'
                THEN replace(c.external_type, 'decimal', 'numeric')
                ELSE
                replace(
                replace(
                    replace(c.external_type, 'decimal', 'numeric'),
                    'char', 'character'),
                'varchar', 'character varying')
                END
                AS "type",
            false AS "distkey",
            0 AS "sortkey",
            null AS "notnull",
            null as "comment",
            null AS "adsrc",
            c.columnnum AS "attnum",
            CASE
                WHEN c.external_type = 'int' THEN 'integer'
                WHEN c.external_type = 'float' THEN 'real'
                WHEN c.external_type = 'double' THEN 'double precision'
                WHEN c.external_type = 'timestamp'
                THEN 'timestamp without time zone'
                WHEN c.external_type ilike 'varchar%%'
                THEN replace(c.external_type, 'varchar', 'character varying')
                WHEN c.external_type ilike 'decimal%%'
                THEN replace(c.external_type, 'decimal', 'numeric')
                ELSE
                replace(
                replace(
                    replace(c.external_type, 'decimal', 'numeric'),
                    'char', 'character'),
                'varchar', 'character varying')
                END
                AS "format_type",
            null AS "default",
            s.esoid AS "schema_oid",
            null AS "table_oid"
        FROM svv_external_columns c
        JOIN svv_external_schemas s ON s.schemaname = c.schemaname
        WHERE 1 """
            + schema_txt
            + """ AND table_name = :table_name
        ORDER BY "schema", "table_name", "attnum";"""
        )
        schema_dict = {"schema": schema}
        table_dict = {"table_name": table_name}
        keys = table_dict | schema_dict if schema else table_dict
        result = self.conn.execute(sa.text(get_columns_sql), keys)
        return [dict(zip(result.keys(), row)) for row in result.all()]

    def get_foreign_keys(self, table_name: str, schema: Optional[str] = None) -> list[dict]:
        schema_txt = """ AND schema = :schema """ if schema else ""
        get_foreign_keys_sql = (
            """SELECT
          n.nspname as "schema",
          c.relname as "table_name",
          t.contype,
          t.conname,
          t.conkey,
          a.attnum,
          a.attname,
          pg_catalog.pg_get_constraintdef(t.oid, true)::varchar(512) as condef,
          n.oid as "schema_oid",
          c.oid as "rel_oid"
        FROM pg_catalog.pg_class c
        LEFT JOIN pg_catalog.pg_namespace n
          ON n.oid = c.relnamespace
        JOIN pg_catalog.pg_constraint t
          ON t.conrelid = c.oid
        JOIN pg_catalog.pg_attribute a
          ON t.conrelid = a.attrelid AND a.attnum = ANY(t.conkey)
        WHERE n.nspname !~ '^pg_' AND t.contype = 'f' """
            + schema_txt
            + """ AND table_name = :table_name"""
        )
        schema_dict = {"schema": schema}
        table_dict = {"table_name": table_name}
        keys = table_dict | schema_dict if schema else table_dict
        result = self.conn.execute(sa.text(get_foreign_keys_sql), keys)

        result_w_keys = [dict(zip(result.keys(), row)) for row in result.all()]
        return [process_foreign_key_definition(row["condef"]) for row in result_w_keys]
