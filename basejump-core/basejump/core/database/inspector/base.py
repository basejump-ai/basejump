from abc import ABC, abstractmethod
from typing import Optional

import sqlalchemy as sa


class BaseInspector(ABC):
    """Implements the same methods as SQLAlchemy inspector so that it can be used in
    place of the SQLAlchemy inspector for drivers which may not be compatible with
    SQLAlchemy 2
    """

    @abstractmethod
    def inspect(cls, conn: sa.Connection):
        pass

    @abstractmethod
    def get_table_names(
        self,
        schema: Optional[str] = None,
        include_views: bool = False,
        include_materialized_views: bool = False,
        include_partitioned_tbls: bool = False,
    ) -> list[str]:
        pass

    @abstractmethod
    def get_permitted_table_names(
        self,
        schema: Optional[str] = None,
        include_views: bool = False,
        include_materialized_views: bool = False,
        include_partitioned_tbls: bool = False,
    ):
        pass

    @abstractmethod
    def get_permitted_schema_names(self) -> list[str]:
        pass

    @abstractmethod
    def get_table_comment(self, table_name: str, schema: Optional[str] = None):
        pass

    @abstractmethod
    def get_columns(self, table_name: str, schema: Optional[str] = None):
        pass

    @abstractmethod
    def get_foreign_keys(self, table_name: str, schema: Optional[str] = None):
        pass


class StandardInspector(BaseInspector):
    def __init__(self, conn: sa.Connection):
        self.inspector = sa.inspect(conn)
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
        tables = self.inspector.get_table_names(schema=schema)
        tables_set = set(tables)
        if include_views:
            views = self.inspector.get_view_names(schema=schema)
            views_set = set(views)
            tables_set.update(views_set)
        if include_materialized_views:
            mat_views = self.inspector.get_materialized_view_names(schema=schema)
            mat_views_set = set(mat_views)
            tables_set.update(mat_views_set)
        if include_partitioned_tbls:
            # NOTE: This is implemented in permitted tables and therefore will be filtered out if set to False
            pass
        return list(tables_set)

    def get_permitted_table_names(
        self,
        schema: Optional[str] = None,
        include_views: bool = False,
        include_materialized_views: bool = False,
        include_partitioned_tbls: bool = False,
    ):
        raise NotImplementedError("This is not implemented in SQLAlchemy. Use a dialect specific inspector instead.")

    def get_permitted_schema_names(self) -> list[str]:
        raise NotImplementedError("This is not implemented in SQLAlchemy. Use a dialect specific inspector instead.")

    def get_table_comment(self, table_name: str, schema: Optional[str] = None) -> dict:
        return dict(self.inspector.get_table_comment(table_name=table_name, schema=schema))

    def get_columns(self, table_name: str, schema: Optional[str] = None) -> list[dict]:
        return [dict(item) for item in self.inspector.get_columns(table_name=table_name, schema=schema)]

    def get_foreign_keys(self, table_name: str, schema: Optional[str] = None) -> list[dict]:
        return [dict(item) for item in self.inspector.get_foreign_keys(table_name=table_name, schema=schema)]
