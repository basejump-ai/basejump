from typing import Optional

import sqlalchemy as sa
from basejump.core.database.inspector.base import StandardInspector


class AthenaInspector(StandardInspector):
    """An Inspector class for using Athena as the query engine"""

    def __init__(self, conn: sa.Connection):
        super().__init__(conn=conn)

    def get_permitted_table_names(
        self,
        schema: Optional[str] = None,
        include_views: bool = False,
        include_materialized_views: bool = False,
        include_partitioned_tbls: bool = False,
    ):
        # NOTE: This should be limited using AWS IAM permissions
        # In Basejump this will be the same as getting all schemas
        return self.get_table_names(
            schema=schema,
            include_views=include_views,
            include_materialized_views=include_materialized_views,
            include_partitioned_tbls=include_partitioned_tbls,
        )

    def get_permitted_schema_names(self) -> list[str]:
        # NOTE: This should be limited using AWS IAM permissions
        # In Basejump this will be the same as getting all schemas
        return self.inspector.get_schema_names()
