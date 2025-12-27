import copy
import pytest
from collections import namedtuple

from basejump.core.models import errors
from basejump.core.models import schemas as sch
from basejump.demo import service, settings
from basejump.core.database.crud import crud_connection, crud_chat
from basejump.core.service import service_utils
from basejump.core.database.diagram import MermaidAgentManager
from basejump.core.database.vector_utils import get_index_name
from basejump.core.models import enums
from basejump.core.database.db_connect import ConnectDB


@pytest.mark.connection
async def test_invalid_db_schemas(client_session):
    """Add a database connection with an incorrect schema to ensure the correct error is thrown"""
    schema = sch.DBSchema(schema_nm="accountz")
    conn_params = sch.SQLDBSchema.parse_obj(client_session.client_conn_params)
    conn_params.schemas = [schema]
    with pytest.raises(errors.InvalidSchemas):
        await service.add_client_database(
            db=client_session.db,
            client_id=client_session.client_id,
            conn_params=conn_params,
            redis_client_async=client_session.redis_client_async,
            client_user=client_session.client_user,
            embedding_model_info=settings.embedding_model_info,
            small_model_info=settings.small_model_info,
            sql_engine=client_session.sql_engine,
        )


@pytest.mark.connection
async def test_get_connections(db_session):
    """Confirm multiple connections will be retrieved for a given team"""
    # Test getting database parameters for an existing connection
    login_params = sch.CreateDBConn(
        username=db_session.client_conn_params.username,
        password=db_session.client_conn_params.password,
        data_source_desc=db_session.client_conn_params.data_source_desc,
    )
    result = await service_utils.create_database_from_existing_connection(
        db=db_session.db,
        client_id=db_session.client_id,
        db_id=db_session.db_id,
        login_params=login_params,
        sql_engine=db_session.sql_engine,
    )

    # Get the connection
    connection = await crud_connection.get_connection(
        db=db_session.db, conn_uuid=result.conn_uuid
    )

    # Add a connection to a team
    await service.add_connection_to_team(
        db=db_session.db,
        client_id=db_session.client_id,
        team_id=db_session.team_id,
        conn_id=connection.conn_id,
    )

    # Retrieve all the connections for a single team
    connections = await crud_connection.get_connections(
        db=db_session.db, user_id=db_session.user_id, team_id=db_session.team_id
    )
    assert len(connections) == 2


@pytest.mark.connection
async def test_get_mermaid_erd_diagram(db_session):
    """Test creating an ERD diagram using MermaidJS"""
    # Retrieve variables
    db_params = await crud_connection.get_database_params(
        db=db_session.db, db_uuid=db_session.db_uuid, get_tables=True
    )
    assert db_params
    tbl_uuids = [table.tbl_uuid for table in db_params.tables if not table.ignore]
    prompt_id, prompt_uuid = await crud_chat.create_prompt_history(
        db=db_session.db,
        client_id=db_session.client_id,
        llm_type=enums.LLMType.MERMAID_AGENT,
    )
    vector_db = await crud_connection.get_vector_from_connection(
        db=db_session.db, db_uuid=db_session.db_uuid
    )

    # Set up the agent
    mermaid_agent = await service.setup_mermaid_agent(
        prompt_id=prompt_id,
        prompt_uuid=prompt_uuid,
        client_user=db_session.client_user,
        large_model_info=settings.large_model_info,
        sql_engine=db_session.sql_engine,
        redis_client_async=db_session.redis_client_async,
    )
    # Set up the mermaid agent manager
    mgn_mermaid = MermaidAgentManager(
        db=db_session.db,
        index_name=get_index_name(client_id=db_session.client_id),
        tbl_uuids=tbl_uuids,
        client_user=db_session.client_user,
        vector_uuid=vector_db.vector_uuid,
        small_model_info=settings.small_model_info,
        large_model_info=settings.large_model_info,
        redis_client_async=db_session.redis_client_async,
        sql_engine=db_session.sql_engine,
        mermaid_agent=mermaid_agent,
    )

    # Create an ERD diagram
    await mgn_mermaid.create_erd_diagram()


@pytest.mark.connection
async def test_invalid_creds(db_session):
    """Try setting up a database using an invalid password"""
    # Try to set up database incorrectly
    conn_params_local = copy.deepcopy(db_session.client_conn_params)
    wrong_password = "1234"
    conn_params_local.password = wrong_password
    with pytest.raises(errors.ConnectDBError):
        await service.add_client_database(
            db=db_session.db,
            client_id=db_session.client_id,
            conn_params=conn_params_local,
            redis_client_async=db_session.redis_client_async,
            client_user=db_session.client_user,
            embedding_model_info=settings.embedding_model_info,
            small_model_info=settings.small_model_info,
            sql_engine=db_session.sql_engine,
        )
    with pytest.raises(errors.ConnectDBError):
        login_params = sch.CreateDBConn(
            username=db_session.client_conn_params.username,
            password=wrong_password,
            data_source_desc=db_session.client_conn_params.data_source_desc,
        )
        await service_utils.create_database_from_existing_connection(
            db=db_session.db,
            client_id=db_session.client_id,
            db_id=db_session.db_id,
            login_params=login_params,
            sql_engine=db_session.sql_engine,
        )


@pytest.mark.connection
async def test_jinjafied_schemas(db_session):
    """Test setting up a database using a jinjafied schema"""
    conn_params_local = copy.deepcopy(db_session.client_conn_params)
    conn_params_local.include_views = True
    conn_params_local.schemas = [
        sch.DBSchema(
            schema_nm="connect{{client_id}}",
            jinja_values={"client_id": str(db_session.client_id)},
        )
    ]
    await service.add_client_database(
        db=db_session.db,
        client_id=db_session.client_id,
        conn_params=conn_params_local,
        redis_client_async=db_session.redis_client_async,
        client_user=db_session.client_user,
        embedding_model_info=settings.embedding_model_info,
        small_model_info=settings.small_model_info,
        sql_engine=db_session.sql_engine,
    )
    with pytest.raises(errors.InvalidSchemas):
        conn_params_local.schemas = [
            sch.DBSchema(
                schema_nm="accountz123",
            )
        ]
        await service.add_client_database(
            db=db_session.db,
            client_id=db_session.client_id,
            conn_params=conn_params_local,
            redis_client_async=db_session.redis_client_async,
            client_user=db_session.client_user,
            embedding_model_info=settings.embedding_model_info,
            small_model_info=settings.small_model_info,
            sql_engine=db_session.sql_engine,
        )


@pytest.mark.connection
async def test_validate_jinja_braces():
    texts = []
    Pairs = namedtuple("Pairs", "text bool")
    texts.append(Pairs("{{hey there}}", True))
    texts.append(Pairs("hey there", True))
    texts.append(Pairs("{hey there}}", False))
    texts.append(Pairs("hey there}}", False))
    texts.append(Pairs("{{hey there}", False))
    texts.append(Pairs("{{hey there", False))
    texts.append(Pairs("{{}}", False))
    texts.append(Pairs("}}hey there{{", False))
    for text in texts:
        try:
            ConnectDB.validate_jinja_braces(text.text)

        except (
            errors.InvalidJinjaBraceCount,
            errors.InvalidJinjaContent,
            errors.InvalidJinjaStartingBrace,
            errors.InvalidJinjaEndingBrace,
        ):
            assert text.bool is False
        else:
            assert text.bool
