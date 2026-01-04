import asyncio

from basejump.core.common.config.logconfig import set_logging
from basejump.core.models import enums
from basejump.core.models import schemas as sch
from basejump.demo import service, settings
from basejump.demo.settings import client_conn_params

logger = set_logging(handler_option="stream", name=__name__)


async def run_main():
    async with service.run_session() as (core_session, db):
        service_context = service.create_service_context(core_session)
        user_info = await service.create_internal_user_info(db, service_context)
        connection = await service.setup_database(db, service_context, user_info, client_conn_params)
        await service.chat(
            db,
            "Provide a report of all clients.",
            service_context,
            user_info,
            connection,
        )


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
    async with service.run_session(client_id=client_result.client_id) as (
        core_session,
        db,
    ):
        service_context = service.create_service_context(core_session)

        # Create a team
        team_result = await service.create_team(
            db=db,
            team_name="AI power users",
            client_id=client_result.client_id,
            team_desc="A team in charge of managing ABC",
        )
        logger.info(team_result)

        # Create a user
        user_result = await service.create_user(
            db=db,
            client_id=client_result.client_id,
            username="John Doe",
            email_address="john@gmail.com",
        )
        logger.info(user_result)

        # Associate a user with a team
        user_team_result = await service.add_user_to_team(
            db=db,
            username=user_result.username,
            team_name=team_result.team_name,
            user_id=user_result.user_id,
            team_id=team_result.team_id,
        )
        logger.info(user_team_result)

        # Add a client database
        # Variable setup
        user_info = sch.UserInfo(
            client_id=client_result.client_id,
            client_uuid=client_result.client_uuid,
            user_id=user_result.user_id,
            user_uuid=user_result.user_uuid,
            user_role="MEMBER",
            team_id=team_result.team_id,
            team_uuid=team_result.team_uuid,
        )

        # Set up the database
        db_result = await service.setup_database(
            db=db,
            service_context=service_context,
            user_info=user_info,
            conn_params=client_conn_params,
        )

        # Add a connection to a team
        await service.add_connection_to_team(
            db=db,
            client_id=client_result.client_id,
            team_id=team_result.team_id,
            conn_id=db_result.conn_id,
        )

        # Ask the AI a question
        chat_result = await service.chat(
            db=db,
            prompt="Provide a report of all clients.",
            service_context=service_context,
            user_info=user_info,
        )
        # Here is the LLM response
        logger.info("LLM response: %s", chat_result.content)
        # Here is the SQL query that was ran
        logger.debug("SQL query: %s", chat_result.query_result.sql_query)  # type: ignore
        # Use this to get the result in AWS S3
        logger.debug("Result UUID: %s", chat_result.query_result.result_uuid)  # type: ignore


if __name__ == "__main__":
    asyncio.run(run_main_full())
