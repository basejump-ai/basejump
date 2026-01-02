import asyncio

from basejump.core.common.config.logconfig import set_logging
from basejump.core.models import enums
from basejump.core.models import schemas as sch
from basejump.demo import service, settings
from basejump.demo.settings import client_conn_params

logger = set_logging(handler_option="stream", name=__name__)


async def run_main():
    async with service.run_session() as core_session:
        service_context = service.create_service_context(core_session)
        user_info = await service.create_internal_user_info(service_context)
        await service.setup_database(service_context, user_info, client_conn_params)
        await service.chat("How many users are there?", service_context, user_info)


async def run_main_full():
    # Create a client
    client_result = await service.create_client(
        sql_engine=settings.sql_engine,
        client_name="ABC Company",
        client_type=enums.ClientType.DEMO,
        description="A company that provides ABC as a service.",
    )
    logger.info(client_result)

    # Create a session
    async with service.run_session(client_id=client_result.client_id) as core_session:
        service_context = service.create_service_context(core_session)

        # Create a team
        team_result = await service.create_team(
            db=core_session.db,
            team_name="AI power users",
            client_id=client_result.client_id,
            team_desc="A team in charge of managing ABC",
        )
        logger.info(team_result)

        # Create a user
        user_result = await service.create_user(
            db=service_context.db,
            client_id=client_result.client_id,
            username="John Doe",
            email_address="john@gmail.com",
        )
        logger.info(user_result)

        # Associate a user with a team
        user_team_result = await service.add_user_to_team(
            db=service_context.db,
            username=user_result.username,
            team_name=team_result.team_name,
            user_id=user_result.user_id,
            team_id=team_result.team_id,
        )
        logger.info(user_team_result)

        # Add a client database
        # Setup variables
        client_user = sch.ClientUserInfo(
            client_id=client_result.client_id,
            client_uuid=client_result.client_uuid,
            user_id=user_result.user_id,
            user_uuid=user_result.user_uuid,
            user_role="MEMBER",
        )

        # Update the client database to a synchronous connection since not all DBs support asynch connections
        client_conn_params = sch.SQLDBSchema(**settings.conn_params.dict())
        client_conn_params.drivername = enums.DBDriverName.POSTGRES
        redis_client_async = settings.get_redis_client_async_instance()
        db_result = await service.setup_database(
            db=service_context.db,
            client_id=client_result.client_id,
            conn_params=client_conn_params,  # Using the same database here for simplicity, but feel free to update
            redis_client_async=redis_client_async,
            client_user=client_user,
            embedding_model_info=settings.embedding_model_info,
            small_model_info=settings.small_model_info,
            sql_engine=settings.sql_engine,
        )
        await redis_client_async.aclose()

        # Add a connection to a team
        await service.add_connection_to_team(
            db=service_context.db,
            client_id=client_result.client_id,
            team_id=team_result.team_id,
            conn_id=db_result.conn_id,
        )

        # Ask the AI a question
        user_info = sch.UserInfo(
            client_id=client_user.client_id,
            client_uuid=client_user.client_id,
            user_id=client_user.user_id,
            user_uuid=client_user.user_uuid,
            user_role=client_user.user_role,
            team_id=team_result.team_id,
            team_uuid=team_result.team_uuid,
        )
        chat_result = await service.chat(
            prompt="How many users are there?",
            service_context=service_context,
            user_info=user_info,
            allow_unrestricted_db_chat=False,
        )
        # Here is the LLM response
        logger.info(chat_result.content)
        # Here is the SQL query that was ran
        logger.info(chat_result.query_result.sql_query)
        # Use this to get the result in AWS S3
        logger.info(chat_result.query_result.result_uuid)


if __name__ == "__main__":
    asyncio.run(run_main())
