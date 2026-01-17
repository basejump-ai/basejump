from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from basejump.core.common.config.logconfig import set_logging
from basejump.core.database.db_utils import get_table_names, get_table_schemas
from basejump.core.models.models import Base

logger = set_logging(handler_option="stream", name=__name__)

SHARED_SCHEMAS = ["account"]


class LocalSession:
    def __init__(self, client_id: int, engine: AsyncEngine, include_dummy_tables: bool = False):
        self._engine = engine
        self.session = None
        self.client_id = client_id
        self.include_dummy_tables = include_dummy_tables

    @property
    def base_schemas(self):
        # Get the schemas from the DB models
        schemas = set(table.schema for table in Base.metadata.tables.values() if table.schema)
        return schemas

    @property
    def schemas(self) -> list:
        return list(self.schema_map.values())

    @property
    def schema_map(self) -> dict:
        schema_map = {}
        for schema_base in self.base_schemas:
            schema = self.get_client_schema(
                client_id=self.client_id,
                schema=str(schema_base),
                include_dummy_tables=self.include_dummy_tables,
            )
            schema_map[schema_base] = schema

        return schema_map

    @staticmethod
    def get_client_schema(client_id: int, schema: str, include_dummy_tables: bool = False) -> str:
        if include_dummy_tables:
            return schema + str(client_id)
        return schema + str(client_id) if schema not in SHARED_SCHEMAS else schema

    async def engine(self):
        if self.client_id == 0:
            return self._engine
        engine_mapped = self._engine.execution_options(schema_translate_map=self.schema_map)
        return engine_mapped

    async def create_schemas(self):
        session = async_sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
        async with session.begin() as conn:
            for schema in get_table_schemas():
                # Create schemas without client IDs if they don't exist
                await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
            for schema in self.schemas:
                # Create schemas with client IDs if they don't exist
                await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))

    async def _manage_views(self, create: bool = False):
        session = async_sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
        async with session.begin() as conn:
            for schema, table_name in get_table_names():
                full_tbl_name = f"{schema}.{table_name}" if schema else table_name
                for client_schema in self.schemas:
                    if client_schema in SHARED_SCHEMAS:
                        continue
                    if schema in client_schema:
                        client_full_tbl_name = f"{client_schema}.{table_name}"
                        if create:
                            stmt = f"""\
    CREATE VIEW {client_full_tbl_name} WITH(security_invoker=TRUE) AS \
    SELECT * FROM {full_tbl_name} \
    WHERE client_id = {self.client_id}"""
                            logger.debug(f"Creating view: {client_full_tbl_name}")
                        else:
                            stmt = f"DROP VIEW IF EXISTS {client_full_tbl_name}"
                            logger.debug(f"Dropping view: {client_full_tbl_name}")
                        await conn.execute(text(stmt))
                        break

    async def create_views(self):
        await self._manage_views(create=True)

    async def delete_views(self):
        await self._manage_views(create=False)

    async def get_session(self):
        """Get a database session"""
        engine = await self.engine()
        session = async_sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=engine)
        return session

    async def open(self):
        """Get a database session"""
        session = await self.get_session()
        self.session = session()
        return self.session

    async def close(self):
        assert self.session
        await self.session.close()

    async def delete_schemas(self, delete_shared=False):
        session = await self.get_session()
        async with session.begin() as conn:
            for schema in self.schemas:
                if delete_shared:
                    await conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
                elif schema not in SHARED_SCHEMAS:
                    await conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
            await conn.commit()
