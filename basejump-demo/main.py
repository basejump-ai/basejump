import asyncio

from basejump.demo import service, settings
from basejump.core.models import enums
from basejump.core.common.config.logconfig import set_logging
from basejump.core.models import schemas as sch
from basejump.core.database.vector_utils import get_index_name

logger = set_logging(handler_option="stream", name=__name__)


async def run_main():

    # ==== Create a client ====
    client_result = await service.create_client(
        sql_engine=settings.sql_engine,
        client_name="ABC Company",
        client_type=enums.ClientType.DEMO,
        description="A company that provides ABC as a service.",
    )
    logger.info(client_result)

    # ==== Create a session ====
    async with service.run_session(client_id=client_result.client_id) as db:

        #  ==== Create a team ====
        team_result = await service.create_team(
            db=db,
            team_name="AI power users",
            client_id=client_result.client_id,
            team_desc="A team in charge of managing ABC",
        )
        logger.info(team_result)

        # ==== Create a user ====
        user_result = await service.create_user(
            db=db,
            client_id=client_result.client_id,
            username="John Doe",
            email_address="john@gmail.com",
        )
        logger.info(user_result)

        # ==== Associate a user with a team ====
        user_team_result = await service.add_user_to_team(
            db=db,
            username=user_result.username,
            team_name=team_result.team_name,
            user_id=user_result.user_id,
            team_id=team_result.team_id,
        )
        logger.info(user_team_result)

        # ==== Add a client database ====

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
        db_result = await service.add_client_database(
            db=db,
            client_id=client_result.client_id,
            conn_params=client_conn_params,  # Using the same database here for simplicity, but feel free to update
            redis_client_async=redis_client_async,
            client_user=client_user,
            embedding_model_info=settings.embedding_model_info,
            small_model_info=settings.small_model_info,
            sql_engine=settings.sql_engine,
        )

        # ==== Add a connection to a team ====
        await service.add_connection_to_team(
            db=db,
            client_id=client_result.client_id,
            team_id=team_result.team_id,
            conn_id=db_result.conn_id,
        )

        # ==== Create a chat ====
        create_chat_result = await service.create_chat(
            db=db,
            client_id=client_result.client_id,
            team_id=team_result.team_id,
            user_id=user_result.user_id,
        )

        # ==== Ask the AI a question ====
        redis_client_async = settings.get_redis_client_async_instance()
        chat_result = await service.chat(
            db=db,
            index_name=get_index_name(client_id=client_result.client_id),
            prompt="Give me a report of all clients.",
            chat_id=create_chat_result.chat_id,
            team_id=team_result.team_id,
            team_info=sch.TeamFields.model_validate(team_result),
            client_user=client_user,
            embedding_model_info=settings.embedding_model_info,
            sql_engine=settings.sql_engine,
            redis_client_async=redis_client_async,
            conn_params=client_conn_params,
            vector_id=create_chat_result.vector_id,
            chat_uuid=create_chat_result.chat_uuid,
            team_uuid=team_result.team_uuid,
            large_model_info=settings.large_model_info,
            small_model_info=settings.small_model_info,
            client_llm=settings.LLM,
        )
        # Here is the LLM response
        logger.info(chat_result.content)
        # Here is the SQL query that was ran
        logger.info(chat_result.query_result.sql_query)
        # Use this to get the result in AWS S3
        logger.info(chat_result.query_result.result_uuid)


if __name__ == "__main__":
    asyncio.run(run_main())
